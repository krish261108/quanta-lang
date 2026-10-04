import json

from quanta.cli import main
from quanta.governance import Action, AuditLog, Governor, Risk


def test_discover_prints_full_report(capsys):
    assert main(["discover", "--seed", "30002"]) == 0
    out = capsys.readouterr().out
    for heading in ("## Result", "## Established facts", "## Inferences", "## Speculation",
                    "## Experiment trace", "## Grading", "## Reproducibility"):
        assert heading in out


def test_bench_writes_results(tmp_path, capsys):
    out_file = tmp_path / "r.json"
    assert main(["bench", "--n", "6", "--jobs", "1", "--out", str(out_file)]) == 0
    data = json.loads(out_file.read_text())
    assert len(data["results"]) == 6 and "accuracy" in json.loads(capsys.readouterr().out)


def test_audit_verify(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    Governor(audit=AuditLog(path)).authorize(Action("t", {}, Risk.LOW))
    assert main(["audit", "verify", str(path)]) == 0
    path.write_text(path.read_text().replace('"allowed": true', '"allowed": false'))
    assert main(["audit", "verify", str(path)]) == 1


def test_memory_commands(tmp_path, capsys):
    from quanta.memory import MemoryStore
    db = tmp_path / "m.db"
    with MemoryStore(db) as m:
        m.add_knowledge("chemistry", "water boils at 100 C at sea level", verified=True)
    assert main(["memory", "search", "boils", "--memory", str(db)]) == 0
    assert "water boils" in capsys.readouterr().out
    assert main(["memory", "stats", "--memory", str(db)]) == 0
