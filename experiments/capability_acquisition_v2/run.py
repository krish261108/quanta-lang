"""Capability acquisition v2 (see PREREGISTRATION.md in this directory).

Identical to v1 (`experiments/capability_acquisition/run.py`) except: the base solver uses
small-sample model evidence (`evidence="gprior"`), the loop and evaluator carry the fixes
found by the v1 independent reviews, the loop gets a proposer-only view of the evaluation
service, and replications 2 and 3 (fresh seeds in every split) are used.

    python experiments/capability_acquisition_v2/run.py --replication 2 [--phase all]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "capability_acquisition" / "run.py"
spec = importlib.util.spec_from_file_location("ca_run", V1)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

from quanta.genome import Genome  # noqa: E402

run.S0 = Genome(falsification_rounds=2, likelihood="student_t", flexible_baseline=True, pool_equivalent=True,
                evidence="gprior")
run.S0_CONSTRUCT = Genome(falsification_rounds=2, likelihood="student_t", flexible_baseline=True,
                          pool_equivalent=True, evidence="gprior", construct=True)
run.SEED = 20261107
run.PROPOSER_VIEW = True
run.HERE = HERE                                  # results go to capability_acquisition_v2/results/

if __name__ == "__main__":
    if not any(a.startswith("--replication") for a in sys.argv[1:]):
        sys.exit("pass --replication 2 or --replication 3 (v1 used 0 and 1)")
    sys.exit(run.main())
