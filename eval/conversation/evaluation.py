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
    if args.patient_profile == "MedDG":
        from prompt import judge_single_dimensions_MedDG as dimensions
    elif args.patient_profile == "AIHospital":
        from prompt import judge_single_dimensions_AIHospital as dimensions

    result = {}

    for dimension in dimensions:
        pattern = rf"<{dimension}>:\s*(\d+)/10\s*-\s*(.+)"
        match = re.search(pattern, evaluation_result)

        if match:
            score = int(match.group(1))
            reason = match.group(2)
            result[dimension] = {
                "score": score,
                "reason": reason
            }
        else:
            result[dimension] = {
                "score": None
            }
    return result

def get_compare_result(evaluation_result):
    from prompt import judge_compare_dimensions as dimensions

    result = {}

    for dimension in dimensions:
        # 构造匹配模式: <维度>: [A好/B好/打平] - 理由
        pattern = rf"<{dimension}>:\s*\[([A好B好打平]+)\]\s*-\s*(.+)"
        match = re.search(pattern, evaluation_result)

        if match:
            comparison_result = match.group(1).strip()
            result[dimension] = comparison_result
        else:
            result[dimension] = "未知"

    return result


def calculate_average_scores_from_file(file_path):
    """
    从评估结果文件中计算各个维度的平均得分

    Args:
        file_path (str): 评估结果文件路径

    Returns:
        dict: 包含各维度平均得分的字典
    """
    import json

    # 读取评估结果文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 定义评分维度
    if args.patient_profile == "MedDG":
        from prompt import judge_single_dimensions_MedDG as dimensions
    elif args.patient_profile == "AIHospital":
        from prompt import judge_single_dimensions_AIHospital as dimensions

    # 初始化统计变量
    dimension_totals = {dim: 0 for dim in dimensions}
    dimension_counts = {dim: 0 for dim in dimensions}

    # 统计各维度得分
    for item in data:
        scores = item.get("score", {})
        for dim in dimensions:
            score_info = scores.get(dim, {})
            score = score_info.get("score")
            if score is not None:
                dimension_totals[dim] += score
                dimension_counts[dim] += 1

    # 计算平均分
    average_scores = {}
    for dim in dimensions:
        if dimension_counts[dim] > 0:
            average_scores[dim] = round(dimension_totals[dim] / dimension_counts[dim], 2)
        else:
            average_scores[dim] = 0

    # 打印结果
    print(f"评估文件: {file_path}")
    print(f"总对话数: {len(data)}")
    print("各维度平均得分:")
    for dim in dimensions:
        print(f"  {dim}: {average_scores[dim]}/10")

    return average_scores


def calculate_win_rates_from_file(file_path):
    """
    从比较评估结果文件中计算两个模型的win次数和打平次数

    Args:
        file_path (str): 比较评估结果文件路径

    Returns:
        dict: 包含统计结果的字典
    """
    import json

    # 读取比较评估结果文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 定义评分维度
    from prompt import judge_compare_dimensions as dimensions

    # 初始化统计变量
    win_counts = {"A好": 0, "B好": 0, "打平": 0}
    dimension_win_counts = {dim: {"A好": 0, "B好": 0, "打平": 0} for dim in dimensions}

    # 统计各维度比较结果
    for item in data:
        comparison_results = item.get("result", {})
        for dim in dimensions:
            result = comparison_results.get(dim, "未知")
            if result in dimension_win_counts[dim]:
                dimension_win_counts[dim][result] += 1
                # 如果是综合结论，也计入总统计
                if dim == dimensions[-1]:  # 假设最后一个维度是综合结论
                    win_counts[result] += 1

    # 打印结果
    print(f"比较评估文件: {file_path}")
    print(f"总比较数: {len(data)}")
    print("\n各维度比较结果:")
    for dim in dimensions:
        counts = dimension_win_counts[dim]
        print(f"  {dim}: A好({counts['A好']}) | B好({counts['B好']}) | 打平({counts['打平']})")

    # 计算胜率（不包括打平）
    total_decided = win_counts['A好'] + win_counts['B好']
    if total_decided > 0:
        a_win_rate = round(win_counts['A好'] / total_decided * 100, 2)
        b_win_rate = round(win_counts['B好'] / total_decided * 100, 2)
        print(f"\n胜率统计 (不含打平):")
        print(f"  A模型胜率: {a_win_rate}%")
        print(f"  B模型胜率: {b_win_rate}%")


def evaluation_single(data_path, judge, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for i, item in tqdm(enumerate(data)):

        dialogue_text = ""
        repeat_flag = 0
        for conv in item.get("conversations", []):
            if "谢谢" in conv["value"]:
                repeat_flag += 1
            if "不客气" in conv["value"]:
                repeat_flag += 1
            if repeat_flag >= 2:
                break
            role = "患者" if conv["from"] == "user" else "医生"
            dialogue_text += f"{role}：{conv['value']}\n"

        if args.patient_profile == "MedDG":
            from prompt import judge_single_prompt_MedDG
            prompt = judge_single_prompt_MedDG.format(dialogue=dialogue_text.strip())
        else:
            from prompt import judge_single_prompt_AIHospital
            prompt = judge_single_prompt_AIHospital.format(dialogue=dialogue_text.strip())

        evaluation_result = judge.evaluate(prompt)

        score = get_single_score(evaluation_result)
        # print(evaluation_result)
        # print(score)

        results.append({
            "conversation_id": item.get("conv_id", "unknown"),
            "dialogue": dialogue_text.strip(),
            "evaluation": evaluation_result,
            "score": score
        })


    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    calculate_average_scores_from_file(save_path)


def evaluation_compare(data_path_a, data_path_b, judge, save_path):
    with open(data_path_a, 'r', encoding='utf-8') as f:
        data_a = json.load(f)

    with open(data_path_b, 'r', encoding='utf-8') as f:
        data_b = json.load(f)

    results = []

    for i, (item_a, item_b) in tqdm(enumerate(zip(data_a, data_b))):
        dialogue_a_text = ""
        repeat_flag = 0
        for conv in item_a.get("conversations", []):
            if "谢谢" in conv["value"]:
                repeat_flag += 1
            if "不客气" in conv["value"]:
                repeat_flag += 1
            if repeat_flag >= 2:
                break
            role = "患者" if conv["from"] == "user" else "医生"
            dialogue_a_text += f"{role}：{conv['value']}\n"

        dialogue_b_text = ""
        for conv in item_b.get("conversations", []):
            role = "患者" if conv["from"] == "user" else "医生"
            dialogue_b_text += f"{role}：{conv['value']}\n"

        from prompt import judge_compare_prompt
        prompt = judge_compare_prompt.format(
            dialogue_a=dialogue_a_text.strip(),
            dialogue_b=dialogue_b_text.strip()
        )

        comparison_result = judge.evaluate(prompt)
        score = get_compare_result(comparison_result)
        # print(comparison_result)
        # print(score)

        results.append({
            "comparison_id": i,
            "dialogue_a": dialogue_a_text.strip(),
            "dialogue_b": dialogue_b_text.strip(),
            "comparison": comparison_result,
            "result": score
        })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", type=str, default="single", choices=["single", "compare", "exp", "exp-select-ratio"])
    parser.add_argument("--patient_profile", type=str, default="AIHospital", choices=["MedDG", "AIHospital"])
    parser.add_argument("--mode", type=str, default="s2t", choices=["t2t", "s2t"])
    parser.add_argument("--model_a", type=str, default="SpeechMedAssist2-final")
    parser.add_argument("--model_b", type=str, default="Llama_omni2")
    parser.add_argument("--eval_model", type=str, default="Qwen")
    parser.add_argument("--eval_model_path", type=str, default="../../weight/Qwen2.5-72B-Instruct")

    args = parser.parse_args()

    with open(f"selected_{args.patient_profile}.json", "r", encoding="utf-8") as f:
        base_info_list = json.load(f)

    if args.eval_model == "Qwen":
        model = JudgeQwen(args.eval_model_path)
    else:
        model = None

    if args.eval_mode == "single":
        data_path = f"dialog_{args.patient_profile}_{args.mode}_{args.model_a}.json"
        save_path = f"dialog_{args.patient_profile}_{args.mode}_{args.model_a}_evaluated.json"
        evaluation_single(data_path, model, save_path)

    elif args.eval_mode == "compare":
        path_a = f"dialog_{args.patient_profile}_{args.mode}_{args.model_a}.json"
        path_b = f"dialog_{args.patient_profile}_{args.mode}_{args.model_b}.json"
        save_path = f"dialog_{args.patient_profile}_{args.mode}_{args.model_a}_{args.model_b}_evaluated.json"
        evaluation_compare(path_a, path_b, model, save_path)
