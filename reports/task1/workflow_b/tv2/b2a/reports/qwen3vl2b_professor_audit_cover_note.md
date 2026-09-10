# Professor audit cover note

Previous `progress_log_2.md` and `task_results.md` stopped at `FULL_VS_400Q_SCORE_DIVERGENCE_UNEXPLAINED`.

Raw pre-fix replay now preserves the cause: PATH_A uses the Vietnamese query text, while PATH_B uses `13844`; 72 per-chunk scores are preserved and the paths match the recorded GOOD and BAD regimes respectively. The raw post-fix replay preserves PATH_A == PATH_B on all 72 chunks, with both paths exactly matching the GOOD regime at document level.

The final full run is fresh: recovered `0`, skipped `0`, fresh scored `1293198`. Its prediction SHA is `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f`; Recall is `0.799797619047619` and Precision is `0.16939285714285715`. Raw official parity preserves 72/72 rows, chunk Pearson `0.9997522649508032`, Spearman `0.999244013387365`, and zero material rank reversals.

Chain of custody is documented at byte level: the fresh remote namespace and the locally evaluated prediction both record SHA256 `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f`. The historical copy/materialization command was not recovered, so this is `BYTES_IDENTICAL_COPY_EVENT_UNRECOVERED`, not a claimed transfer-event proof.

The exact historical pre-fix Python source snapshot was not preserved. Consequently, a real source diff is `NOT_AVAILABLE` and source-code-level validation-escape proof remains `UNPROVEN`; execution-level validation/full divergence is nevertheless `PROVEN_BY_RAW_MICRO_REPLAY`. Please treat this as a documented limitation rather than silently accepting a reconstructed diff.
