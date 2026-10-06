"""The solver configuration that runs inside isolated evaluation workers.

A toolkit is a genome plus a list of capability artifacts. An artifact is plain
data containing generated, policy-checked source code; loading one never executes
anything beyond arithmetic over `x` and parameters (see `synth.check_source`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Sequence

from ..genome import Genome
from .discovery import DiscoveryLoop
from .synth import Abstraction, Model, SynthHypothesis, lib_leaf

ARTIFACT_INTERFACE = "hypothesis-family/v1"


class ArtifactError(ValueError):
    pass


def artifact_hash(artifact: dict) -> str:
    """Hash of the executable content of an artifact (terms only)."""
    canonical = json.dumps(artifact.get("terms", []), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_artifact(artifact: dict) -> tuple[SynthHypothesis, list[Abstraction]]:
    """Validate an artifact and turn it into (family hypothesis, building blocks)."""
    if artifact.get("interface") != ARTIFACT_INTERFACE:
        raise ArtifactError(f"unsupported interface {artifact.get('interface')!r}")
    cid = str(artifact.get("capability_id", ""))
    if not cid:
        raise ArtifactError("artifact needs a capability_id")
    if artifact.get("sha256") and artifact["sha256"] != artifact_hash(artifact):
        raise ArtifactError("artifact content does not match its hash")
    terms = artifact.get("terms") or []
    if not 1 <= len(terms) <= 2:
        raise ArtifactError("an artifact has one or two terms")
    blocks = []
    for i, t in enumerate(terms):
        try:
            blocks.append(Abstraction(
                name=f"{cid}.t{i}", source=str(t["source"]), n_params=int(t["n_params"]),
                n_freq=int(t.get("n_freq", 0)), structure=str(t.get("structure", "")),
                inits=tuple(tuple(float(v) for v in init) for init in t.get("inits", []))))
        except (KeyError, TypeError, ValueError) as e:
            raise ArtifactError(f"bad term {i}: {e}") from None
    model = Model(tuple(lib_leaf(b) for b in blocks))
    family = SynthHypothesis(model, name=f"cap:{cid}", origin="capability")
    return family, blocks


@dataclass
class Toolkit:
    genome: Genome
    capabilities: list[dict] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict) -> "Toolkit":
        return cls(Genome.from_dict(config.get("genome", {})), list(config.get("capabilities", [])))

    def to_config(self) -> dict:
        return {"genome": self.genome.to_dict(), "capabilities": self.capabilities}

    def _loaded(self):
        fams, blocks = [], []
        for art in self.capabilities:
            f, b = load_artifact(art)
            fams.append(f)
            blocks.extend(b)
        return fams, blocks

    def solve(self, instrument, *, budget: int, seed: int, eval_x: Sequence[float]) -> dict:
        families, blocks = self._loaded()       # fresh hypothesis objects per task
        loop = DiscoveryLoop(self.genome, budget=budget, seed=seed, extra_hypotheses=families,
                             library=blocks, predict_x=eval_x, separate_streams=True)
        res = loop.run(instrument)
        return {
            "answer": res.answer, "credence": res.credence, "kind": res.answer_kind,
            "description": res.best_fit.description,
            "pred": [res.predict(x) for x in eval_x],
            "n_experiments": res.n_experiments, "stop_reason": res.stop_reason,
            "inadequacy": res.inadequacy_detected, "constructed": res.constructed,
            "posterior_top": [[k, v] for k, v in list(res.posterior.items())[:5]],
            "observations": [[o.x, o.y] for o in res.observations],
        }
