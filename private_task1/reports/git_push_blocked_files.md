# Git push blocked files

The following local payload is intentionally not added to Git because it is a
large generated/data synchronization directory:

- Directory: `_tmp_phase2b_remote_sync/`
- Absolute path: `D:\udsc2026\_tmp_phase2b_remote_sync\`
- Files: 7,732
- Total size: 2,155,263,205 bytes (approximately 2.01 GiB)
- Local archive: `_tmp_phase2b_remote_sync.zip`
- Archive absolute path: `D:\udsc2026\_tmp_phase2b_remote_sync.zip`

The source directory and archive are both listed in the root `.gitignore`.
The archive is retained locally for recovery and is not pushed to GitHub.
