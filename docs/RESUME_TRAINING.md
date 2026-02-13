# 断点续训说明（OpenVLA）

本文说明如何在本仓库进行断点续训，包含 LoRA 微调（finetune.py）与全量训练（train.py / FSDP 或 DDP）。

## 1. LoRA 微调（vla-scripts/finetune.py）

### 1.1 必备前提
- 你需要一个已保存的 LoRA adapter 目录，里面至少包含：
  - adapter_config.json
  - adapter_model.safetensors
- 该目录通常位于 run 生成的 adapter_tmp_dir 下，例如：
  /opt/data/private/openvla_icms/tmp/<exp_id>

### 1.2 配置文件里需要设置的字段
在你的 YAML（如 configs/method_icsm_hsw_thermostat.yaml）中：

```yaml
resume_adapter_dir: /opt/data/private/openvla_icms/tmp/<exp_id>
resume_global_step: 250
```

说明：
- resume_adapter_dir 用来加载 LoRA 权重，并会尝试加载该目录下的 trainer_state.pt。
- resume_global_step 用于步数偏移；如果为 0 且 trainer_state.pt 存在，会使用其中的 global_step。

### 1.3 启动续训
```bash
bash scripts/run_method_train.sh 1
```

### 1.4 如果你不想续训（从头训练）
请这样设置：
- 删除或注释 resume_adapter_dir 和 resume_full_model_dir（如果有）
- 将 resume_global_step 设为 0

示例：
```yaml
# resume_adapter_dir: /opt/data/private/openvla_icms/tmp/<exp_id>
resume_global_step: 0
```

注意：
- 不要把 resume_global_step 改成 None 或 null。当前代码会执行 int()，会报错。

### 1.5 重要限制（必须知道）
当前 finetune 续训会恢复 LoRA 权重与 optimizer/scheduler 状态（若 trainer_state.pt 存在）。
仍然不会恢复训练指标与日志状态。

trainer_state.pt 保存位置：
- 最新状态：resume_adapter_dir/trainer_state.pt
- 若 save_latest_checkpoint_only=false：每个 step 的 checkpoint 目录也会保存一份

---

## 2. 全量训练（vla-scripts/train.py，FSDP/DDP）

### 2.1 断点文件位置
训练会在 run 目录下保存：
```
<run_root_dir>/<run_id>/checkpoints/step-000250-epoch-00-loss=2.3456.pt
```

### 2.2 续训配置
在对应的 TrainConfig 里（通常不是 finetune.yaml，而是 train 相关配置）：
```yaml
pretrained_checkpoint: /opt/data/private/openvla_icms/runs/<run_id>/checkpoints/step-000250-epoch-00-loss=2.3456.pt
is_resume: true
resume_step: 250
resume_epoch: 0
```

说明：
- pretrained_checkpoint 必须指向具体的 checkpoint 文件；
- resume_step 和 resume_epoch 要与文件名里的 step/epoch 对齐，否则会触发断言。

---

## 3. 常见问题（FAQ）

### 3.1 为什么 SIGKILL (exitcode -9)？
这通常是系统 OOM 杀进程（显存或内存不足）。常见触发点：
- batch_size 太大
- grad_accumulation_steps 太大
- probe_batch_size 太大（method 模式下会额外前向）
- 同时保存/合并模型导致峰值内存

建议尝试：
1) 降低 batch_size 或 grad_accumulation_steps
2) 降低 probe_batch_size
3) 将 merge_lora_during_training 保持为 false
4) 先关闭 method_enabled 进行对照验证

### 3.2 如何确认是否真的 OOM？
建议查看系统日志（需 root）：
```
dmesg | tail -n 50
```
如果看到 OOM kill 的记录，就能确认原因。

---

## 4. 推荐操作清单
- 续训前先确认 adapter_dir / checkpoint 是否存在
- 续训时记录 resume_global_step 与 save_steps 对齐
- 需要完全可复现时建议加上手动日志记录（步数、lr、loss）

如需我帮你把“optimizer/scheduler 恢复”也做上，告诉我你的训练入口与保存策略要求。
