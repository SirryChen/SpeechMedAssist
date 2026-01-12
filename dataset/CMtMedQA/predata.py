import json
from tqdm import tqdm

data_path = "CMtMedQA.json"

# 读取原始 JSON 数据（假设文件名为data.json）
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 用于保存转换后的对话数据
sharegpt_data = []

# 遍历每条记录
for i, item in tqdm(enumerate(raw_data)):
    conversations = []

    # 先添加历史对话（按顺序）
    for turn in item.get("history", []):
        user_utterance, assistant_response = turn
        conversations.append({"from": "user", "value": user_utterance})
        conversations.append({"from": "assistant", "value": assistant_response})

    # 添加当前 instruction 和 output
    if item.get("instruction"):
        conversations.append({"from": "user", "value": item["instruction"]})
    if item.get("output"):
        conversations.append({"from": "assistant", "value": item["output"]})

    # 整理成 ShareGPT 格式
    sharegpt_data.append({
        "idx": f"CMtMedQA-{i}",
        "id": item["id"],
        "cate1": item["cate1"],
        "cate2": item["cate2"],
        "conversations": conversations
    })

# 保存为新的 JSON 文件
with open("CMtMedQA-sharegpt.json", "w", encoding="utf-8") as f:
    json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

