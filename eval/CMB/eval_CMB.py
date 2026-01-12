import json
from tqdm import tqdm
import random
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../")
sys.path.append(project_path)


def generate_answer(question_list, model, output_path):
    answer_list = []
    for item in tqdm(question_list, desc=f"generating responses from {TEST_MODEL}"):
        question = item["question"] if "SpeechMedAssist" in TEST_MODEL or "Zhongjing" == TEST_MODEL else item["question"] + "**不需要额外解释**\n\n"
        messages = [{"from": "user", "value": question}]
        response = model.reply(messages)["text"]

        item["response"] = response
        answer_list.append(item)

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(answer_list, f, indent=4, ensure_ascii=False)


def evaluation(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 按考试类型分类统计
    type_stats = {exam_type: {"correct": 0, "total": 0} for exam_type in exam_types}

    # 总体统计
    correct = 0
    total = len(data)

    for item in tqdm(data, desc="evaluating"):
        response = item.get('response', '')
        answer = item.get('answer', '')
        exam_type = item.get('exam_type', '')

        # 检查答案是否正确
        is_correct = answer in response

        # 更新总体统计
        if is_correct:
            correct += 1

        # 更新分类统计
        if exam_type in exam_types:
            type_stats[exam_type]["total"] += 1
            if is_correct:
                type_stats[exam_type]["correct"] += 1

    # 计算并打印总体准确率
    overall_acc = correct / total if total > 0 else 0
    print(f"总体准确率: {overall_acc:.4f} ({correct}/{total})")

    # 计算并打印各类别准确率
    print("\n各考试类型准确率:")
    print("-" * 50)
    for exam_type in exam_types:
        stats = type_stats[exam_type]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{exam_type:8s}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    return overall_acc, type_stats


def compare(model1_path, model2_path, output_path):
    """
    比较两个模型的答案，找出model1正确但model2错误的样本
    Args:
        model1_path: 第一个模型的答案文件路径
        model2_path: 第二个模型的答案文件路径
        output_path: 输出文件路径
    """
    with open(model1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(model2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    correct_in_model1_wrong_in_model2 = []

    for i, (item1, item2) in enumerate(zip(data1, data2)):
        # 检查model1是否正确
        answer = item1["answer"]
        response1 = item1.get('response', '')
        model1_correct = answer in response1

        # 检查model2是否正确
        response2 = item2.get('response', '')
        model2_correct = answer in response2

        # 如果model1正确但model2错误，保存该样本
        if model1_correct and not model2_correct:
            sample = {
                "question": item1.get('question', ''),
                "answer": answer,
                "model1_response": response1,
                "model2_response": response2,
                "model1_correct": True,
                "model2_correct": False
            }
            correct_in_model1_wrong_in_model2.append(sample)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(correct_in_model1_wrong_in_model2, f, indent=4, ensure_ascii=False)

    print(f"找到 {len(correct_in_model1_wrong_in_model2)} 个model1正确但model2错误的样本")
    print(f"结果已保存到: {output_path}")

    return correct_in_model1_wrong_in_model2


def sample_questions_by_type(questions, exam_types, sample_ratio=0.2):
    """
    对数据集按类别进行采样，每个类别按比例随机挑选样本
    Args:
        questions: 原始问题列表
        exam_types: 考试类型列表
        sample_ratio: 每个类别采样的比例，默认为0.2（20%）
    Returns:
        sampled_questions: 采样后的问题列表
    """
    # 按考试类型分类存储问题
    questions_by_type = {exam_type: [] for exam_type in exam_types}

    # 将问题按类型分类
    for item in questions:
        exam_type = item.get('exam_type')
        if exam_type in exam_types:
            questions_by_type[exam_type].append(item)

    # 对每个类型按比例进行随机采样
    sampled_questions = []
    for exam_type, type_questions in questions_by_type.items():
        # 计算采样数量（至少为1）
        sample_count = max(1, int(len(type_questions) * sample_ratio))

        # 如果采样数量大于等于该类型问题总数，则全部选取
        if sample_count >= len(type_questions):
            sampled_questions.extend(type_questions)
            print(f"{exam_type}: 采样了 {len(type_questions)} 个问题 (全部)")
        else:
            # 随机选取指定数量的问题
            sampled_questions.extend(random.sample(type_questions, sample_count))
            print(f"{exam_type}: 采样了 {sample_count} 个问题 ({sample_ratio * 100:.1f}%)")

    return sampled_questions


if __name__ == "__main__":
    TEST_DATA_PATH = "../../dataset/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"
    SELECT_DATA_PATH = "CMB_test_data.json"
    SELECT_RATIO = 1
    TEST_MODEL = "SpeechMedAssist2-2k"

    exam_types = ["医师考试", "护理考试", "专业知识考试", "医学考研"]
    question_format = "医生，我有个单项选择题想考你一下。\n{question}{option}\n请你直接回答正确答案选项和选项内容。" if TEST_MODEL != "HuatuoGPT2-7B" else "<问>：请回答下面选择题。\n[单项选择题]{question}\n{option}\n直接输出最终答案。"
    if not os.path.exists(SELECT_DATA_PATH):
        with open("../../dataset/CMB/CMB-test-choice-answer.json", encoding='utf-8') as f:
            answers = json.load(f)
        questions = []

        with open(TEST_DATA_PATH, 'r') as f:
            data = json.load(f)
            for item in data:
                if item["exam_type"] not in exam_types:
                    continue
                if item["question_type"] != "单项选择题":
                    continue
                item['question'] = question_format.format(question=item["question"],
                                                          option=" ".join(
                                                              [f"{cc}: {text}" for cc, text in item["option"].items()]))
                item["answer"] = answers[item["id"] - 1]["answer"]
                questions.append(item)

        questions = sample_questions_by_type(questions, exam_types, sample_ratio=SELECT_RATIO)
        with open(SELECT_DATA_PATH, "w", encoding='utf-8') as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)

    with open(SELECT_DATA_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    model = None
    if TEST_MODEL == "DISC-MedLLM":
        from inference.DISC_MedLLM import DISC_MedLLM
        model = DISC_MedLLM(model_path="../../weight/DISC-MedLLM")
    elif TEST_MODEL == "HuatuoGPT2-7B":
        from inference.HuatuoGPT2 import HuatuoGPT2
        model = HuatuoGPT2(model_path="../../weight/HuatuoGPT2-7B")
    elif TEST_MODEL == "Llama-Omni2-7B":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/LLaMA-Omni2-7B-Bilingual/")
    elif TEST_MODEL == "ShizhenGPT":
        from inference.ShizhenGPT import ShizhenGPT
        model = ShizhenGPT(model_path="../../weight/ShizhenGPT-7B-Omni")
    elif TEST_MODEL == "SpeechMedAssist1":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage1")
    elif TEST_MODEL == "SpeechMedAssist2":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage2-ratio-1")
    elif TEST_MODEL == "SpeechMedAssist2-2k":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage2/checkpoint-5000")
    elif TEST_MODEL == "SpeechGPT2":
        from inference.SpeechGPT2 import SpeechGPT2
        model = SpeechGPT2(model_path="../../weight/SpeechGPT-2.0-preview", max_new_token=64)
    elif TEST_MODEL == "Baichuan2":
        from inference.Baichuan2 import Baichuan2
        model = Baichuan2(model_path=f"../../weight/Baichuan2-7B")
    elif TEST_MODEL == "Qwen2-Audio":
        from inference.Qwen2_Audio import Qwen2_Audio
        model = Qwen2_Audio(model_path="../../weight/Qwen2-Audio-7B-Instruct")
    elif TEST_MODEL == "GLM4-Voice":
        from inference.GLM4_Voice import GLM4_Voice
        model = GLM4_Voice(model_path="../../weight/GLM4-Voice/", max_new_token=64)
    elif TEST_MODEL == "Zhongjing":
        from inference.Zhongjing import Zhongjing
        model = Zhongjing(model_path=f"../../weight/Zhongjing")
    elif TEST_MODEL == "Baichuan-M2":
        from inference.Baichuan_M2 import Baichuan_M2
        model = Baichuan_M2(model_path=f"../../weight/Baichuan-M2-32B")

    generate_answer(questions, model, f"{TEST_MODEL}_answer.json")
    evaluation(f"{TEST_MODEL}_answer.json")