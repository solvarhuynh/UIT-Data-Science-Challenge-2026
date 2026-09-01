import json
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import load

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/task1"
CONTRACT = R / "step2p1_model_contract.json"
P1C = R / "step2p1c_realizability_report.json"
REFERENCE = R / "step2p1c_inner_oof_top_scores.jsonl"
OUT = R / "step2p5_p1_inner_full_action_scores.jsonl"
REPORT = R / "step2p5a_reproducibility_report.json"


def action_id(a):
    return {"incoming_doc_id": a["incoming_doc_id"], "incoming_union_rank": a["incoming_union_rank"],
            "drop_rank": a["drop_rank"], "dropped_doc_id": a["dropped_doc_id"]}


def top(items):
    return sorted(items, key=lambda z: (-z[1], -z[0]["drop_rank"], z[0]["incoming_union_rank"], z[0]["incoming_doc_id"]))[0]


def score(model, rows, columns):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    result = {}
    for query, xs in grouped.items():
        count = len(xs)
        if count < 2:
            raise RuntimeError(f"CONTRACT_ERROR single-action query {query}")
        X = np.empty((count * (count - 1), len(columns)), dtype=np.float32)
        owner = np.empty(count * (count - 1), dtype=np.int32)
        pos = 0
        for i, a in enumerate(xs):
            for j, b in enumerate(xs):
                if i == j:
                    continue
                X[pos] = [a["features"][c] - b["features"][c] for c in columns]
                owner[pos] = i
                pos += 1
        prob = model.predict_proba(X)[:, 1]
        total = np.zeros(count, dtype=np.float64)
        np.add.at(total, owner, prob)
        result[query] = [(a, float(total[i] / (count - 1))) for i, a in enumerate(xs)]
    return result


def main():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p1c = json.loads(P1C.read_text(encoding="utf-8"))
    if not (contract["status"] == "FROZEN_PRE_FIT" and len(contract["feature_columns"]) == 36 and contract["fixed_K"] == 77 and
            p1c["status"] == "PASS" and p1c["inner_models_persisted"] == 12):
        raise RuntimeError("CONTRACT_ERROR canonical P1/P1C contract")
    columns = contract["feature_columns"]
    reference = [json.loads(x) for x in REFERENCE.read_text(encoding="utf-8").splitlines() if x]
    reference_by_split = {(x["outer_held_fold"], x["inner_valid_fold"], x["query_id"]): x for x in reference}
    if len(reference) != len(reference_by_split) != 16800:
        raise RuntimeError("CONTRACT_ERROR reference top-score rows")
    paths = [Path(x) for x in p1c["inner_model_paths"]]
    if not all((ROOT / path).is_file() for path in paths):
        raise RuntimeError("BLOCKED_CANONICAL_MODEL_RECONSTRUCTION missing persisted canonical P1C model")
    ns = runpy.run_path(str(ROOT / "scripts/analysis/step2_k77_oof_policy_realizability.py"), run_name="p5a_parent")
    folds = ns["target_folds"](); records, gold = ns["load_gold"](set(folds)); baseline, _ = ns["load_baseline"](set(folds), folds); candidates, _ = ns["load_candidates"](set(folds), folds)
    actions, incoming = ns["make_actions"](folds, records, gold, baseline, candidates, columns)
    if incoming != 403322 or sum(len(actions[f]) for f in range(1, 5)) != 806644:
        raise RuntimeError("CONTRACT_ERROR K77 action universe")
    models_meta = []; mismatches = []; rows_written = 0; queries_checked = identity_matches = score_matches = 0; max_error = 0.0
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for relpath in paths:
            payload = load(ROOT / relpath)
            outer, inner = payload["outer_held_fold"], payload["inner_valid_fold"]
            expected_train = [f for f in range(1, 5) if f not in (outer, inner)]
            if payload["inner_train_folds"] != expected_train or payload["ordered_feature_columns"] != columns or payload["model_parameters"] != contract["hyperparameters"]:
                raise RuntimeError(f"CONTRACT_ERROR model provenance {relpath}")
            model = payload["model"]
            scores = score(model, actions[inner], columns)
            models_meta.append({"outer_fold": outer, "inner_fold": inner, "model_path": str(relpath).replace('\\', '/'), "training_folds": expected_train, "inner_held_out_fold": inner})
            for query, items in scores.items():
                ref = reference_by_split.get((outer, inner, query))
                if ref is None:
                    raise RuntimeError("CONTRACT_ERROR missing reference query")
                winner = top(items); stored = ref["top_action_identity"]
                exact = action_id(winner[0]) == stored
                error = abs(winner[1] - ref["top_action_score"])
                score_ok = error <= 1e-9
                queries_checked += 1; identity_matches += int(exact); score_matches += int(score_ok); max_error = max(max_error, error)
                if not (exact and score_ok):
                    mismatches.append({"outer_fold": outer, "inner_fold": inner, "query_id": query, "stored_top_action": stored,
                                       "regenerated_top_action": action_id(winner[0]), "identity_match": exact,
                                       "stored_top_score": ref["top_action_score"], "regenerated_top_score": winner[1],
                                       "abs_score_error": error, "score_match": score_ok})
                ordered = sorted(items, key=lambda z: (-z[1], -z[0]["drop_rank"], z[0]["incoming_union_rank"], z[0]["incoming_doc_id"]))
                rank = {id(a): i + 1 for i, (a, _) in enumerate(ordered)}
                for a, value in items:
                    handle.write(json.dumps({"outer_fold": outer, "inner_fold": inner, "query_id": query, "fold": inner,
                                             "action_id": action_id(a), "incoming_doc_id": a["incoming_doc_id"], "incoming_union_rank": a["incoming_union_rank"],
                                             "drop_rank": a["drop_rank"], "dropped_doc_id": a["dropped_doc_id"], "P1_score": value,
                                             "prediction_scope": "INNER_HELD_OUT", "candidate_count": len(items), "P1_rank": rank[id(a)],
                                             "is_P1_top": a is winner[0]}) + "\n")
                    rows_written += 1
    passed = queries_checked == 16800 and not mismatches
    report = {"status": "PASS" if passed else "REPRODUCTION_MISMATCH", "step": "STEP 2-P5-A — P1 Inner Full-Action Score Artifact Completion",
              "experiment_type": "ARTIFACT_COMPLETION_ONLY", "scientific_intervention": False, "P5_executed": False, "policy_modified": False,
              "fold0_or_public_used": False, "policy_oof_strict": True, "oof_violation": False, "end_to_end_selection_oof": False, "K": 77,
              "model_recovery": {"frozen_models_expected": 12, "frozen_models_found": len(models_meta), "frozen_model_primary_path_used": True,
                                 "fallback_retraining_used": False, "fallback_retraining_models": 0,
                                 "canonical_model_directory": "reports/task1/step2p1c_models/inner_fold_models", "models": models_meta},
              "score_contract": {"features": 36, "pair_input": "x(a)-x(b)", "pair_probability": "P(a outranks b)",
                                 "aggregation": "MEAN_PAIRWISE_WIN_PROBABILITY", "tie_break": "CANONICAL_P1"},
              "reference_artifact": "reports/task1/step2p1c_inner_oof_top_scores.jsonl", "reference_top_rows": 16800,
              "output_artifact": "reports/task1/step2p5_p1_inner_full_action_scores.jsonl",
              "reproduction_gate": {"identity_tolerance": "EXACT", "score_absolute_tolerance": 1e-9, "queries_expected": 16800,
                                     "queries_checked": queries_checked, "identity_matches": identity_matches, "identity_mismatches": queries_checked - identity_matches,
                                     "score_matches": score_matches, "score_mismatches": queries_checked - score_matches, "max_abs_score_error": max_error, "all_queries_pass": passed},
              "full_action_artifact": {"rows": rows_written, "queries": queries_checked, "outer_folds": [1, 2, 3, 4], "inner_models": 12,
                                       "prediction_scope": "INNER_HELD_OUT", "artifact_trusted": passed},
              "P5_status": {"fusion_hypothesis": "AUTHORIZED_BUT_NOT_RUN", "artifact_realizability": "COMPLETED_AND_REPRODUCED" if passed else "UNTRUSTED_REPRODUCTION_MISMATCH",
                            "fusion_computed": False, "threshold_selected": False, "R_computed": False, "D77_computed": False},
              "preserved_status": {"P4_global_signal": "STRONG_DESCRIPTIVE_OOF_SIGNAL", "P4Q_within_query_signal": "SUPPORTED_DESCRIPTIVELY", "relative_proxy_reranking": "AUTHORIZED",
                 "absolute_proxy_gate": "NOT_AUTHORIZED", "proxy_only_ranking": "NOT_AUTHORIZED", "P3": "CLOSED", "pair_reweighting": "NOT_AUTHORIZED", "B_vs_N_only": "NOT_AUTHORIZED",
                 "rank_feature_intervention": "NOT_JUSTIFIED", "raw_pairwise_dispersion": "DEFERRED", "feature_expansion": "NOT_JUSTIFIED", "K77": "NOT_REJECTED", "smaller_K": "DEFERRED", "workflow_B": "NOT_ACTIVE"},
              "mismatches": mismatches, "notes": ["P5 fusion was not executed.", "Frozen canonical P1C models were loaded for their own original inner-held-out splits only."]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "queries_checked": queries_checked, "identity_matches": identity_matches,
                      "score_matches": score_matches, "max_abs_score_error": max_error, "rows": rows_written}, indent=2))


if __name__ == "__main__":
    main()
