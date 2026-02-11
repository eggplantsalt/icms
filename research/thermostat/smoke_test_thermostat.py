from __future__ import annotations

import torch

from research.thermostat.thermostat import Thermostat, ThermostatConfig, ThermostatState


class FakeThermostat(Thermostat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._d = 1.0

    def _compute_drift(self, model, batch):
        return float(self._d)


def main() -> None:
    teacher_stats = {0: {"C_T": torch.eye(4)}}
    cfg = ThermostatConfig(update_interval=2, warmup_steps=2, min_beta=0.2, max_beta=1.0, min_gamma=0.1, max_gamma=1.0)
    thermostat = FakeThermostat(
        teacher_stats=teacher_stats,
        rep_layer_ids=[0],
        config=cfg,
        prompt_template="In: {instruction}\nOut:",
        processor_or_tokenizer=None,
    )

    state = ThermostatState(beta=1.0, gamma=1.0)
    dummy_batch = {"input_ids": torch.zeros(1, 1), "attention_mask": torch.ones(1, 1), "pixel_values": torch.zeros(1, 3, 224, 224), "instructions": ["hi"]}

    for step in range(5):
        thermostat._d = 1.0 + 0.5 * step
        out = thermostat.maybe_update(step, None, dummy_batch, state)
        print(step, out)


if __name__ == "__main__":
    main()
