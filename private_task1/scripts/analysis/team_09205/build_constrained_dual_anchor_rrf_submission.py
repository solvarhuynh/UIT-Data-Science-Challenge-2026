from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

K = 60
W_GUARDED = 0.55
W_V2 = 0.45


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


def load_submission_zip(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if len(names) != 1:
            raise RuntimeError(f"Expected exactly one JSON member in {path}, got {names}")
        data = json.loads(zf.read(names[0]))
    if not isinstance(data, dict):
        raise RuntimeError(f"Submission must be a JSON object: {path}")
    for qid, row in data.items():
        ans = row.get("answer") if isinstance(row, dict) else None
        if not isinstance(ans, list) or len(ans) != 5 or len(set(map(str, ans))) != 5:
            raise RuntimeError(f"Invalid top5 at query {qid} in {path}")
        row["answer"] = [str(x) for x in ans]
    return {str(k): v for k, v in data.items()}


def resolve_v2(repo: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(p)
    preferred = [
        repo / "private_task1" / "submissions" / "submission_v2.zip",
        repo / "outputs" / "task1" / "private" / "submission_v2.zip",
        repo / "artifacts" / "task1" / "submission_v2.zip",
        repo / "submission_v2.zip",
    ]
    hits = [p for p in preferred if p.is_file()]
    if not hits:
        hits = sorted({p.resolve() for p in repo.rglob("submission_v2.zip") if p.is_file()})
    if len(hits) != 1:
        raise RuntimeError(
            "Could not uniquely resolve submission_v2.zip. "
            f"Found {len(hits)} candidates: {[str(p) for p in hits]}. "
            "Pass --v2 explicitly."
        )
    return hits[0]


def validate_private_ids(repo: Path, preds: dict[str, dict[str, list[str]]]) -> None:
    private_path = repo / "private_task1" / "input" / "private-official.json"
    if not private_path.is_file():
        raise FileNotFoundError(private_path)
    private = json.loads(private_path.read_text(encoding="utf-8"))
    expected = {str(k) for k in private}
    actual = set(preds)
    if actual != expected:
        raise RuntimeError(
            f"Private query-id mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}"
        )


def build(guarded, v2):
    if set(guarded) != set(v2):
        raise RuntimeError("Guarded and v2 query-id sets differ")
    out = {}
    injected_v2_r1 = 0
    guarded_r1_missing_before = 0
    for qid in guarded:
        scores: dict[str, float] = {}
        for r, d in enumerate(guarded[qid]["answer"], 1):
            scores[d] = scores.get(d, 0.0) + W_GUARDED / (K + r)
        for r, d in enumerate(v2[qid]["answer"], 1):
            scores[d] = scores.get(d, 0.0) + W_V2 / (K + r)

        ranked = sorted(scores.keys(), key=lambda d: -scores[d])
        top5 = ranked[:5]
        g_r1 = guarded[qid]["answer"][0]
        v_r1 = v2[qid]["answer"][0]

        if g_r1 not in top5:
            top5[4] = g_r1
            guarded_r1_missing_before += 1
        if v_r1 not in top5:
            top5[4] = v_r1
            injected_v2_r1 += 1

        if len(top5) != 5 or len(set(top5)) != 5:
            raise RuntimeError(f"Duplicate/invalid top5 after guardrail at {qid}")
        if g_r1 not in top5 or v_r1 not in top5:
            raise RuntimeError(
                f"Guardrail collision at {qid}; refusing to silently change original semantics"
            )
        out[qid] = {"answer": top5}
    return out, guarded_r1_missing_before, injected_v2_r1


def main() -> None:
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
        "submission_private_constrained_dual_anchor_rrf_rebuilt.zip"
    )

    guarded = load_submission_zip(guarded_path)
    v2 = load_submission_zip(v2_path)
    final, g_missing, v_injected = build(guarded, v2)
    validate_private_ids(repo, final)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(final, ensure_ascii=False, separators=(",", ":"))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", payload)

    print(f"repo_root={repo}")
    print(f"guarded={guarded_path}")
    print(f"v2={v2_path}")
    print(f"output={output}")
    print(f"queries={len(final)}")
    print(f"guarded_r1_missing_before_guardrail={g_missing}")
    print(f"v2_r1_injected={v_injected}")
    print("labels_used=false")


if __name__ == "__main__":
    main()
