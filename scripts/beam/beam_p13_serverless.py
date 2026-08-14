"""Deprecated launcher kept only to prevent accidental use.

Production P13 runs through beam_p13_run_all.py (@task_queue, RTX5090,
persistent Beam Volume, local model staging, robust logging, fold-safe resume).
"""

raise SystemExit(
    "beam_p13_serverless.py is deprecated. "
    "Use beam_p13_run_all.py for P13 production."
)
