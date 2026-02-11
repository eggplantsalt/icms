from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from research.icms.offline_icms import OfflineICMSConfig, run_offline_icms_core


def main() -> None:
    proj_root = Path("/opt/data/private/openvla_icms")
    tmp_dir = proj_root / "tmp" / "icms_smoke"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    image_path = tmp_dir / "dummy.png"
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(image_path)

    jsonl_path = tmp_dir / "probe.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write('{"image": "dummy.png", "instruction": "pick up the cup"}\n')
        f.write('{"image": "dummy.png", "instruction": "open the drawer"}\n')

    cfg = OfflineICMSConfig(
        vla_path="openvla/openvla-7b",
        cache_dir=proj_root / "hf_cache",
        data_root_dir=proj_root / "datasets",
        probe_root_dir=proj_root / "probe",
        artifact_dir=proj_root / "artifacts" / "icms_smoke",
        tmp_dir=proj_root / "tmp",
        probe_jsonl=jsonl_path,
        probe_image_root=tmp_dir,
        batch_size=1,
        max_samples=2,
        sensitivity_samples=1,
        r=4,
        epsilon=1e-3,
    )
    run_offline_icms_core(cfg)


if __name__ == "__main__":
    main()
