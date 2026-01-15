→ Switched from folder-based savings to compressed `.zip` files in `asi-backups`.

→ Reason: Including `.log` files in the folder was breaking hash integrity checks.

→ Legacy Support: For versions older than **2.9.7** (commit-7c4cabe), one must exclude `log.txt` when calculating hashes.
