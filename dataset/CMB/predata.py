import json
from tqdm import tqdm

data_path = "./CMB-Exam/CMB-train/CMB-train-merge.json"
question_format = "医生，我有个单项选择题想考你一下。\n{question}{option}\n请你直接回答正确答案选项和选项内容。"

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

sharegpt_data = []
for i, data in tqdm(enumerate(raw_data)):
    if data["exam_type"] not in ["医师考试", "护理考试", "专业知识考试", "医学考研"]:
        continue
    if data["question_type"] != "单项选择题":
        continue
    if data["answer"] == '':
        continue
    question = question_format.format(question=data["question"], option=" ".join([f"{cc}: {text}" for cc, text in data["option"].items()]))
    answer = f"{data['answer']} {data['option'][data['answer']]}"
    conversation = [{"from": "user", "value": question}, {"from": "gpt", "value": answer}]
    sharegpt_data.append({
        "idx": f"CMB-{i}",
        "exam_type": data["exam_type"],
        "exam_class": data["exam_class"],
        "exam_subject": data["exam_subject"],
        "conversations": conversation
    })

with open("CMB-train-sharegpt.json", "w", encoding="utf-8") as f:
    json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

