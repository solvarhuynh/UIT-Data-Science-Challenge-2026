"""Report-only CPU Step 1F exploratory frozen K-selection protocol."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEP1D = ROOT / "reports/task1/exp_1d_union_rank_sweep_report.json"
STEP1E = ROOT / "reports/task1/exp_1e_action_space_noise_report.json"
REPORT = ROOT / "reports/task1/exp_1f_k_selection_protocol_report.json"
EXPANDED = [23, 29, 45, 77, 135, 197]
ALL_K = [20, *EXPANDED]
EXPECTED_BENEFIT = {
    23: 0.12285675712164142, 29: 0.2875658161198867,
    45: 0.5474551100310516, 77: 0.771837451059808,
    135: 0.9189955447549616, 197: 1.0,
}
EXPECTED_COST = {
    23: 792.0975609756098, 29: 1047.5531914893618,
    45: 1494.0756756756757, 77: 2340.6125461254614,
    135: 3937.846625766871, 197: 5449.289256198347,
}
EXPECTED_SHELL = {
    23: 792.0975609756098, 29: 1245.1698113207547,
    45: 1955.3186813186812, 77: 4161.6511627906975,
    135: 11807.854545454546, 197: 18766.324324324323,
}
EXPECTED_HARM = {
    23: 0.027004557211479246, 29: 0.027003148166954403,
    45: 0.027011186524073458, 77: 0.02703427052558229,
    135: 0.02704134332706518, 197: 0.02704525370912981,
}


def minmax(values: dict[int, float]) -> dict[int, float]:
    low, high = min(values.values()), max(values.values())
    if high <= low:
        raise ValueError("minmax normalization has zero range")
    return {key: (value - low) / (high - low) for key, value in values.items()}


def choose(benefit: dict[int, float], cost: dict[int, float]) -> tuple[int, dict[int, float], dict[int, float], dict[int, float]]:
    bn, cn = minmax(benefit), minmax(cost)
    scores = {key: bn[key] - cn[key] for key in EXPANDED}
    selected = min(EXPANDED, key=lambda key: (-scores[key], key))
    return selected, bn, cn, scores


def by_k(rows: list[dict]) -> dict[int, dict]:
    return {int(row["K"]): row for row in rows}


def main() -> None:
    d = json.loads(STEP1D.read_text(encoding="utf-8"))
    e = json.loads(STEP1E.read_text(encoding="utf-8"))
    d_sweep = by_k(d.get("sweep", []))
    e_actions = by_k(e.get("action_utility_composition", []))
    e_candidates = by_k(e.get("candidate_composition", []))
    e_shell = {int(row["interval"].split("->")[1]): row for row in e.get("interval_composition", [])}
    grid_ok = d.get("status") == "PASS" and e.get("status") == "PASS" and d.get("k_selection", {}).get("final_K_values") == ALL_K and e.get("K_values") == ALL_K
    step1d_ok = grid_ok and all(abs(float(d_sweep[k]["fraction_of_action_space_gap_recovered"]) - value) <= 1e-10 for k, value in EXPECTED_BENEFIT.items())
    step1e_ok = all(abs(float(e_actions[k]["harmful_action_rate"]) - EXPECTED_HARM[k]) <= 1e-12 for k in EXPANDED)
    correction_ok = abs(float(e_actions[29]["actions_per_beneficial_action"]) - EXPECTED_COST[29]) <= 1e-10
    if not (step1d_ok and step1e_ok and correction_ok):
        raise RuntimeError("canonical Step1D/Step1E contract mismatch")

    benefit = {k: float(d_sweep[k]["fraction_of_action_space_gap_recovered"]) for k in EXPANDED}
    cost = {k: float(e_actions[k]["actions_per_beneficial_action"]) for k in EXPANDED}
    selected, bn, cn, scores = choose(benefit, cost)
    pooled = [{
        "K": k, "benefit_raw": benefit[k], "actions_per_beneficial_raw": cost[k],
        "benefit_norm": bn[k], "cost_norm": cn[k], "balance_score": scores[k],
        "harmful_action_rate": float(e_actions[k]["harmful_action_rate"]),
        "shell_actions_per_beneficial": float(e_shell[k]["actions_per_beneficial_action"]),
        "shell_harmful_action_rate": float(e_shell[k]["shell_harmful_action_rate"]),
    } for k in EXPANDED]

    lofo = []
    for held_out in range(1, 5):
        selection_folds = [fold for fold in range(1, 5) if fold != held_out]
        recovered: dict[int, float] = {}
        full = 0.0
        for k in EXPANDED:
            recovered[k] = sum(float(d_sweep[k]["per_fold"][str(f)]["sum_oracle_gain"]) - float(d_sweep[20]["per_fold"][str(f)]["sum_oracle_gain"]) for f in selection_folds)
        full = recovered[197]
        if full <= 0:
            raise ValueError("LOFO full recovery denominator is zero")
        lofo_benefit = {k: recovered[k] / full for k in EXPANDED}
        lofo_cost: dict[int, float] = {}
        for k in EXPANDED:
            totals = [e["per_fold"][str(k)][str(f)] for f in selection_folds]
            actions = sum(int(item["total_actions_added_from_K20"]) for item in totals)
            beneficial = sum(int(item["beneficial_actions_added"]) for item in totals)
            if beneficial == 0:
                raise ValueError("LOFO beneficial action denominator is zero")
            lofo_cost[k] = actions / beneficial
        chosen, lb, lc, ls = choose(lofo_benefit, lofo_cost)
        lofo.append({
            "held_out_fold": held_out, "selection_folds": selection_folds,
            "per_K": [{"K": k, "benefit_raw": lofo_benefit[k], "actions_per_beneficial_raw": lofo_cost[k], "benefit_norm": lb[k], "cost_norm": lc[k], "balance_score": ls[k]} for k in EXPANDED],
            "selected_K": chosen, "distance_from_pooled_index": abs(EXPANDED.index(chosen) - EXPANDED.index(selected)),
        })
    exact = sum(row["selected_K"] == selected for row in lofo)
    neighbor = sum(row["distance_from_pooled_index"] <= 1 for row in lofo)
    stable = exact >= 3 and neighbor == 4
    decision = "ADOPT_RESEARCH_K" if stable else "NO_ADOPTION_STABILITY_FAILED"
    adopted = selected if stable else None
    report = {
        "status": "PASS", "step": "STEP_1F", "experiment": "Action-Space Tractability / K-Selection Protocol",
        "scientific_status": "EXPLORATORY_PROTOCOL_SELECTION",
        "selection_rule_status": "EXPLORATORY_FROZEN_BEFORE_MECHANICAL_APPLICATION",
        "folds_used": [1, 2, 3, 4], "fold0_used": False, "public_labels_used": False,
        "candidate_K_values": EXPANDED, "baseline_K": 20,
        "input_artifacts": {"step1d_report": str(STEP1D.relative_to(ROOT)), "step1e_report": str(STEP1E.relative_to(ROOT))},
        "canonical_crosschecks": {"step1d_verified": step1d_ok, "step1e_verified": step1e_ok, "k29_actions_per_beneficial_corrected": correction_ok},
        "historical_candidate_noise_score": {"status": "DIAGNOSTIC_ONLY_REJECTED_FOR_K_SELECTION", "used_for_selection": False},
        "selection_rule": {"benefit_metric": "fraction_of_action_space_gap_recovered", "cost_metric": "actions_per_beneficial_action", "benefit_normalization": "minmax_over_expanded_candidate_K", "cost_normalization": "minmax_over_expanded_candidate_K", "formula": "benefit_norm - cost_norm", "tie_break": "smallest_K", "free_parameters": 0},
        "pooled": {"per_K": pooled, "selected_K": selected, "selected_K_shell_actions_per_beneficial": float(e_shell[selected]["actions_per_beneficial_action"]), "selected_K_harmful_action_rate": float(e_actions[selected]["harmful_action_rate"]), "selected_K_shell_harmful_action_rate": float(e_shell[selected]["shell_harmful_action_rate"]), "selected_K_per_fold_harmful_action_rate": {str(f): float(e["per_fold"][str(selected)][str(f)]["harmful_action_rate"]) for f in range(1,5)}},
        "lofo_stability": {"runs": lofo, "exact_match_count": exact, "neighbor_match_count": neighbor, "selection_stability_pass": stable},
        "selection_decision": decision, "adopted_research_K": adopted, "production_K_selected": False,
        "harm_guardrail_status": "DESCRIPTIVE_ONLY_NO_PRE_REGISTERED_THRESHOLD",
        "step1b_rerun_required_next": stable, "rank13_status": "REOPEN_PENDING_STEP1B_RERUN_AT_ADOPTED_K" if stable else "REJECTED_UNDER_CURRENT_K20_ONLY",
        "policy_C_minus_D_status": "UNRESOLVED", "future_step2_status": "GATED_NOT_ACTIVE", "workflow_document_updated": False,
        "notes": ["The rule was designed after Step1D/Step1E evidence existed; it is exploratory but frozen before mechanical application.", "Selection uses report-level sums for LOFO aggregation, not mean-of-ratios.", "No raw candidate/gold data, Fold0 data, public labels, training, policy replay or inference were used."],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
