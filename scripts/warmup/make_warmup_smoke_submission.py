try:
    from scripts._run_compat import run_relocated
except ModuleNotFoundError:  # Direct ``python scripts/<name>.py`` execution.
    from _run_compat import run_relocated  # type: ignore[import-not-found,no-redef]

if __name__ == "__main__":
    run_relocated("submission", "make_warmup_smoke_submission.py")
