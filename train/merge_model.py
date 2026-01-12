import os
import json
from safetensors.torch import load_file, save_file
import torch
import shutil

# 1. 路径
# trained_ckpt_dir = "../weight/stage2"   # 你训练好的权重文件夹
# orig_ckpt_dir = "../weight/LLaMA-Omni2-7B-Bilingual"      # 原始模型文件夹
# output_dir = "../weight/stage3"                        # 输出合并后的权重

# trained_ckpt_dir = "../weight/stage1"   # 你训练好的权重文件夹
# orig_ckpt_dir = "../weight/LLaMA-Omni2-7B-Bilingual"      # 原始模型文件夹
# output_dir = "../weight/stage1/stage1-final"                        # 输出合并后的权重

trained_ckpt_dir = "../weight/stage2/checkpoint-5000"   # 你训练好的权重文件夹
orig_ckpt_dir = "../weight/LLaMA-Omni2-7B-Bilingual"      # 原始模型文件夹
output_dir = "../weight/stage1/stage2-2k-final"                        # 输出合并后的权重

os.makedirs(output_dir, exist_ok=True)

# ===== 1. 读取 index 文件 =====
with open(os.path.join(trained_ckpt_dir, "model.safetensors.index.json"), "r") as f:
    trained_index = json.load(f)
with open(os.path.join(orig_ckpt_dir, "model.safetensors.index.json"), "r") as f:
    orig_index = json.load(f)

# ===== 2. 加载训练后的权重（adaptor + llm） =====
merged_state = {}
for shard in set(trained_index["weight_map"].values()):
    shard_path = os.path.join(trained_ckpt_dir, shard)
    merged_state.update(load_file(shard_path))

# ===== 3. 覆盖/补充 speech_generator 的权重 =====
for name, shard in orig_index["weight_map"].items():
    if name.startswith("speech_generator."):
        shard_path = os.path.join(orig_ckpt_dir, shard)
        tensor = load_file(shard_path, device="cpu")[name]
        merged_state[name] = tensor

print(f"最终参数数量: {len(merged_state)}")

# ===== 4. 保存为单文件 safetensors（带 metadata） =====
metadata = {"format": "pt"}
save_file(merged_state, os.path.join(output_dir, "model.safetensors"), metadata=metadata)

# ===== 5. 复制配置和 tokenizer 文件 =====
for file in ["config.json", "generation_config.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.json", "merges.txt"]:
    src = os.path.join(trained_ckpt_dir, file)
    if os.path.exists(src):
        shutil.copy(src, output_dir)

print("✅ 合并完成，结果在:", output_dir)
