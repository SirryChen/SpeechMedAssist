import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class JudgeQwen:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    def evaluate(self, prompt):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        output_ids = outputs[0][inputs.input_ids.shape[-1]:]
        reply = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return reply


def get_single_score(evaluation_result):
    pattern = r"<?整体评分>?[:：]\s*(\d+)/10"
    match = re.search(pattern, evaluation_result)

    if match:
        score = int(match.group(1))
        result = {
            "score": score,
            "reason": evaluation_result  # 因为这里只匹配分数，理由暂时置空
        }
    else:
        print(evaluation_result)
        result = None

    return result


def calculate_average_scores_from_file(file_path):
    import json

    # 读取评估结果文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    average_score = 0

    # 统计各维度得分
    for item in data:
        scores = get_single_score(item["evaluation"])
        score = scores.get("score")
        average_score += score

    average_score = average_score / len(data)

    # 打印结果
    print(f"评估文件: {file_path}")
    print(f"总对话数: {len(data)}")
    print(f"平均得分：{average_score}")

    return average_score


def evaluation_single(data_path, judge, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for i, item in tqdm(enumerate(data)):

        dialogue_text = ""
        for conv in item.get("conversations", []):
            role = "患者" if conv["from"] == "user" else "医生"
            dialogue_text += f"{role}：{conv['value']}\n"

        from prompt import judge_single_prompt
        prompt = judge_single_prompt.format(dialogue=dialogue_text.strip(), ref_response=item["ref_response"])

        evaluation_result = judge.evaluate(prompt)
        scores = get_single_score(evaluation_result)

        results.append({
            "conversation_id": item.get("conv_id", "unknown"),
            "dialogue": dialogue_text.strip(),
            "evaluation": evaluation_result,
            "score": scores
        })


    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    calculate_average_scores_from_file(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="s2t", choices=["t2t", "s2t"])
    parser.add_argument("--test_model", type=str, default="SpeechMedAssist2")
    parser.add_argument("--eval_model", type=str, default="Qwen")
    parser.add_argument("--eval_model_path", type=str, default="../../weight/Qwen2.5-72B-Instruct")

    args = parser.parse_args()

    if args.eval_model == "Qwen":
        model = JudgeQwen(args.eval_model_path)
    else:
        model = None

    data_path = f"dialog_{args.mode}_{args.test_model}.json"
    save_path = f"dialog_{args.mode}_{args.test_model}_evaluated.json"
    evaluation_single(data_path, model, save_path)

