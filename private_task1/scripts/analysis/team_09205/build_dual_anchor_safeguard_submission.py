from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path


def find_repo_root() -> Path:
    env = os.environ.get("UDSC2026_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "private_task1").exists():
            return p
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "private_task1").is_dir() and (p / "data").exists():
            return p
    raise RuntimeError("Cannot locate repo root. Set UDSC2026_ROOT=D:/udsc2026")


def load_zip(path: Path):
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one JSON in {path}")
        data = json.loads(zf.read(names[0]))
    out = {}
    for qid, row in data.items():
        ans = [str(x) for x in row["answer"]]
        if len(ans) != 5 or len(set(ans)) != 5:
            raise RuntimeError(f"Invalid top5 at {qid}")
        out[str(qid)] = {"answer": ans}
    return out


def resolve_v2(repo: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    hits = sorted({p.resolve() for p in repo.rglob("submission_v2.zip") if p.is_file()})
    if len(hits) != 1:
        raise RuntimeError(f"Need unique submission_v2.zip, found: {[str(p) for p in hits]}; pass --v2")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guarded", default=None)
    ap.add_argument("--v2", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    repo = find_repo_root()
    guarded_path = Path(args.guarded).resolve() if args.guarded else (
        repo / "private_task1" / "submissions" / "sprint48_guarded_direct" /
        "submission_private_guarded_direct.zip"
    )
    v2_path = resolve_v2(repo, args.v2)
    output = Path(args.output).resolve() if args.output else (
        repo / "private_task1" / "submissions" / "team_09205" /
        "submission_private_dual_anchor_guarded_rebuilt.zip"
    )
    guarded, v2 = load_zip(guarded_path), load_zip(v2_path)
    if set(guarded) != set(v2):
        raise RuntimeError("Query IDs differ")
    final = {}
    changes = 0
    for qid in guarded:
        ans = list(guarded[qid]["answer"])
        v2_r1 = v2[qid]["answer"][0]
        if v2_r1 not in ans:
            ans[4] = v2_r1
            changes += 1
        final[qid] = {"answer": ans}
    private = json.loads((repo / "private_task1" / "input" / "private-official.json").read_text(encoding="utf-8"))
    if set(final) != {str(k) for k in private}:
        raise RuntimeError("Private query-id set mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(final, ensure_ascii=False, separators=(",", ":")))
    print(f"output={output}")
    print(f"changed_queries={changes}")
    print("labels_used=false")


if __name__ == "__main__":
    main()
