from __future__ import annotations

"""Probe 数据集构建工具。

提供 RLDS 探针采样与 JSONL 探针加载，返回与 OpenVLA prompt 对齐的 batch。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.vla.datasets.rlds import make_interleaved_dataset


@dataclass
class ProbeBatchTransform:
    base_tokenizer: object
    image_transform: object
    prompt_template: str

    def __call__(self, rlds_batch: Dict[str, object]) -> Dict[str, object]:
        # 从 RLDS 字段构造 prompt 输入。
        dataset_name = rlds_batch["dataset_name"]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        instruction = f"What action should the robot take to {lang}?"

        prompt = self.prompt_template.format(instruction=instruction)
        input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids)
        pixel_values = self.image_transform(img)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "instruction": instruction,
            "dataset_name": dataset_name,
        }


class ProbeRLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: ProbeBatchTransform,
        resize_resolution: tuple[int, int],
        shuffle_buffer_size: int = 100_000,
        frame_num_parallel_calls: int = 4,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        # 复用 RLDS 配置风格，保证与训练管线一致。
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.batch_transform = batch_transform

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=0,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=frame_num_parallel_calls,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        if image_aug:
            rlds_config["frame_transform_kwargs"].update(
                {
                    "image_augment_kwargs": dict(
                        random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.05],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    )
                }
            )

        self.dataset, self.dataset_length, _ = make_interleaved_dataset(**rlds_config)

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length


class ProbeJSONLDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_root: Optional[Path] = None) -> None:
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        # JSONL 需要包含 {"image": path, "instruction": text}。
        item = self.items[idx]
        image_path = item["image"]
        instruction = item["instruction"]
        if self.image_root is not None:
            image_path = str(self.image_root / image_path)
        image = Image.open(image_path).convert("RGB")
        return {"image": image, "instruction": instruction}


@dataclass
class ProbeCollator:
    pad_token_id: int
    model_max_length: int
    padding_side: str = "right"

    def __call__(self, instances: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        # 右侧 padding，并堆叠图像张量。
        input_ids = [inst["input_ids"] for inst in instances]
        pixel_values = [inst["pixel_values"] for inst in instances]
        instructions = [inst["instruction"] for inst in instances]

        max_len = min(
            max([int(ids.shape[0]) for ids in input_ids]) if input_ids else 0,
            self.model_max_length,
        )

        padded_ids = []
        for ids in input_ids:
            if ids.shape[0] > max_len:
                ids = ids[:max_len]
            if self.padding_side == "right":
                pad_len = max_len - ids.shape[0]
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
            else:
                raise ValueError("Only right padding is supported for probe collator")
            padded_ids.append(ids)

        input_ids = torch.stack(padded_ids)
        attention_mask = input_ids.ne(self.pad_token_id)

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "instructions": instructions,
        }


def build_probe_dataloader(
    processor,
    data_root_dir: Path,
    dataset_name: str,
    prompt_template: str,
    batch_size: int,
    shuffle_buffer_size: int = 10_000,
    frame_num_parallel_calls: int = 4,
    image_aug: bool = False,
) -> DataLoader:
    # 使用 Processor 构建 RLDS probe dataloader。
    batch_transform = ProbeBatchTransform(
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_template=prompt_template,
    )
    dataset = ProbeRLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=(processor.image_processor.input_sizes[0][1], processor.image_processor.input_sizes[0][2]),
        shuffle_buffer_size=shuffle_buffer_size,
        frame_num_parallel_calls=frame_num_parallel_calls,
        image_aug=image_aug,
    )
    collator = ProbeCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        model_max_length=processor.tokenizer.model_max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=None, collate_fn=collator, num_workers=0)


def build_probe_dataloader_from_components(
    tokenizer,
    image_transform,
    data_root_dir: Path,
    dataset_name: str,
    prompt_template: str,
    batch_size: int,
    resize_resolution: tuple[int, int],
    shuffle_buffer_size: int = 10_000,
    frame_num_parallel_calls: int = 4,
    image_aug: bool = False,
) -> DataLoader:
    # 使用 tokenizer + image_transform 构建 RLDS probe dataloader。
    batch_transform = ProbeBatchTransform(
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_template=prompt_template,
    )
    dataset = ProbeRLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        frame_num_parallel_calls=frame_num_parallel_calls,
        image_aug=image_aug,
    )
    collator = ProbeCollator(
        pad_token_id=tokenizer.pad_token_id,
        model_max_length=tokenizer.model_max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=None, collate_fn=collator, num_workers=0)


def build_probe_jsonl_dataloader(
    processor,
    jsonl_path: Path,
    image_root: Optional[Path],
    prompt_template: str,
    batch_size: int,
) -> DataLoader:
    # 使用 Processor 构建 JSONL probe dataloader。
    dataset = ProbeJSONLDataset(jsonl_path, image_root=image_root)

    def collate(instances: List[Dict[str, object]]):
        input_ids = []
        pixel_values = []
        instructions = []
        for inst in instances:
            instruction = inst["instruction"]
            prompt = prompt_template.format(instruction=instruction)
            ids = processor.tokenizer(prompt, add_special_tokens=True).input_ids
            input_ids.append(torch.tensor(ids))
            pixel_values.append(processor.image_processor.apply_transform(inst["image"]))
            instructions.append(instruction)

        collator = ProbeCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            model_max_length=processor.tokenizer.model_max_length,
        )
        return collator(
            [
                {"input_ids": ids, "pixel_values": pv, "instruction": instr}
                for ids, pv, instr in zip(input_ids, pixel_values, instructions)
            ]
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)


def build_probe_jsonl_dataloader_from_components(
    tokenizer,
    image_transform,
    jsonl_path: Path,
    image_root: Optional[Path],
    prompt_template: str,
    batch_size: int,
) -> DataLoader:
    # 使用 tokenizer + image_transform 构建 JSONL probe dataloader。
    dataset = ProbeJSONLDataset(jsonl_path, image_root=image_root)

    def collate(instances: List[Dict[str, object]]):
        input_ids = []
        pixel_values = []
        instructions = []
        for inst in instances:
            instruction = inst["instruction"]
            prompt = prompt_template.format(instruction=instruction)
            ids = tokenizer(prompt, add_special_tokens=True).input_ids
            input_ids.append(torch.tensor(ids))
            pixel_values.append(image_transform(inst["image"]))
            instructions.append(instruction)

        collator = ProbeCollator(
            pad_token_id=tokenizer.pad_token_id,
            model_max_length=tokenizer.model_max_length,
        )
        return collator(
            [
                {"input_ids": ids, "pixel_values": pv, "instruction": instr}
                for ids, pv, instr in zip(input_ids, pixel_values, instructions)
            ]
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
