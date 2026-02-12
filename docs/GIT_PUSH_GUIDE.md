# Git Push Guide (CLI)

This guide records the exact CLI steps we used to push local changes to the remote repo.
All commands run from /workspace/openvla.

## 1) Check status (ignore LIBERO)

```bash
git status
```

## 2) Stage only the main repo files (exclude LIBERO submodule)

```bash
git add -A
# If LIBERO shows up as a submodule change, do not add it:
# git reset LIBERO
```

## 3) Confirm what will be committed

```bash
git status
```

## 4) Commit with a clear message

```bash
git commit -m "update research pipeline"
```

## 5) Push to the remote (retry if needed)

```bash
git push
```

If push fails due to transient network errors, run `git push` again until it succeeds.

## Notes

- Do not push the LIBERO submodule.
- If you see files you did not change, stop and confirm before committing them.
