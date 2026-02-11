from __future__ import annotations

"""磁盘安全自检：检查路径是否在 /opt/data/private，并扫描 /workspace 大文件。"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def _is_under(path: Path, root: Path) -> bool:
    # 判断 path 是否位于 root 之下。
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _scan_large_files(root: Path, size_gb: float) -> List[Path]:
    # 扫描超过阈值的大文件。
    threshold = int(size_gb * (1024 ** 3))
    offenders = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if path.is_file() and path.stat().st_size > threshold:
                    offenders.append(path)
            except OSError:
                continue
    return offenders


def main() -> None:
    # 命令行入口。
    parser = argparse.ArgumentParser(description="Disk safety self-check")
    parser.add_argument("--workspace_root", type=Path, default=Path("/workspace"))
    parser.add_argument("--allowed_root", type=Path, default=Path("/opt/data/private"))
    parser.add_argument("--project_root", type=Path, default=Path("/opt/data/private/openvla_icms"))
    parser.add_argument("--max_workspace_file_gb", type=float, default=1.0)
    args = parser.parse_args()

    required_dirs = {
        "HF_CACHE": args.project_root / "hf_cache",
        "DATA_ROOT": args.project_root / "datasets",
        "PROBE_ROOT": args.project_root / "probe",
        "ARTIFACT_ROOT": args.project_root / "artifacts",
        "RUN_ROOT": args.project_root / "runs",
        "TMP_ROOT": args.project_root / "tmp",
    }

    print("[check] Required directories (expected under allowed root):")
    bad = []
    for name, path in required_dirs.items():
        ok = _is_under(path, args.allowed_root)
        status = "OK" if ok else "BAD"
        print(f"  - {name}: {path} => {status}")
        if not ok:
            bad.append(path)

    if bad:
        print("\n[warning] Some required paths are not under allowed root.")
        print("           Please move them under /opt/data/private and update configs.")

    offenders = _scan_large_files(args.workspace_root, args.max_workspace_file_gb)
    if offenders:
        print("\n[warning] Large files detected in /workspace:")
        for path in offenders:
            size_gb = path.stat().st_size / (1024 ** 3)
            print(f"  - {path} ({size_gb:.2f} GB)")
        print("\n[hint] Move these files to /opt/data/private/<proj>/ and remove from /workspace.")
    else:
        print("\n[check] No large files found in /workspace.")


if __name__ == "__main__":
    main()
