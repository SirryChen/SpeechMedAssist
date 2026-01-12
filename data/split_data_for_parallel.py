import json
import os
from math import ceil

# 配置路径
data_path = "../dataset/SpeechMedDataset/annotated_t2t.json"
split_base_path = "../dataset/SpeechMedDataset/split_s2t_part/"
output_base_path = "../dataset/SpeechMedDataset/output_s2t_part/"

"""
# 启动 4 个后台进程进行语音合成

nohup python synthesize.py \
  --data_path ../dataset/SpeechMedDataset/split_s2t_part/1.json \
  --output_path ../dataset/SpeechMedDataset/output_s2t_part/1.json \
  > ../log/syn_part1.log 2>&1 &

nohup python synthesize.py \
  --data_path ../dataset/SpeechMedDataset/split_s2t_part/2.json \
  --output_path ../dataset/SpeechMedDataset/output_s2t_part/2.json \
  > ../log/syn_part2.log 2>&1 &

nohup python synthesize.py \
  --data_path ../dataset/SpeechMedDataset/split_s2t_part/3.json \
  --output_path ../dataset/SpeechMedDataset/output_s2t_part/3.json \
  > ../log/syn_part3.log 2>&1 &

nohup python synthesize.py \
  --data_path ../dataset/SpeechMedDataset/split_s2t_part/4.json \
  --output_path ../dataset/SpeechMedDataset/output_s2t_part/4.json \
  > ../log/syn_part4.log 2>&1 &

"""


def split():
    if not os.path.exists(split_base_path):
        os.mkdir(split_base_path)
        os.mkdir(output_base_path)
    # 加载原始数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 拆分为 4 份
    total = len(data)
    print(f"数据总数: {total}")
    split_num = 8
    split_len = ceil(total / split_num)

    for i in range(split_num):
        split_data = data[i * split_len:(i + 1) * split_len]
        split_path = os.path.join(split_base_path, f"{i+1}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=4, ensure_ascii=False)
        print(f"保存: {split_path}，共 {len(split_data)} 条")


def converge():
    output_files = os.listdir(output_base_path)

    merged = []
    miss_num = 0
    for file in output_files:
        with open(os.path.join(output_base_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)

            for item in data:
                miss_flag = False
                merged_convs = []
                for i, turn in enumerate(item["conversations"]):
                    if turn["from"] in ["human", "user"] and turn.get("speech") is None:
                        miss_flag = True

                    # 如果是 assistant，并且上一个也是 assistant，就合并
                    if merged_convs and turn["from"] == "assistant" and merged_convs[-1]["from"] == "assistant":
                        merged_convs[-1]["value"] += " " + turn["value"]
                    else:
                        merged_convs.append(turn)

                item["conversations"] = merged_convs

                if miss_flag:
                    miss_num += 1
                    print(item)
                else:
                    merged.append(item)
    print(miss_num)
    # 保存合并后的完整文件
    final_path = "../dataset/SpeechMedDataset/train_s2t_normal.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f"合并完成，总样本数: {len(merged)}，保存到: {final_path}")


if __name__ == "__main__":
    split()
    converge()

