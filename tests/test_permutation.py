import json
import os
import tempfile

from scripts.build_permuted import main as permute_main


def test_permutation(tmp_path):
    src = tmp_path / "src.jsonl"
    out = tmp_path / "out.jsonl"
    src.write_text(
        '{"id":"x","question":"q","choices":["A","B"],"answer":0}\n',
        encoding="utf-8",
    )
    # call
    import sys

    sys.argv = ["", str(src), str(out), "--seed", "1"]
    permute_main()
    line = out.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert set(data["choices"]) == {"A", "B"}
    assert data["answer"] in [0, 1]
    assert data["id"].endswith("_perm1")

