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
        question = question_format.format(question=item["question"])
        messages = [{"from": "user", "value": question}]
        response = model.reply(messages)["text"]

        item["response"] = response
        answer_list.append(item)

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(answer_list, f, indent=4, ensure_ascii=False)


def evaluation(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    correct = 0

    for item in data:
        # 移除了 exam_type 相关的判断逻辑
        if item["answer"][0] in item["response"]:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    print(f"{TEST_MODEL}的准确率: {accuracy:.4f} ({correct}/{total})")


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

"""
python eval_CMExam.py 2>&1 | tee -a eval.log
"""

if __name__ == "__main__":
    TEST_DATA_PATH = "CMExam_test.json"
    SELECT_DATA_PATH = "CMExam_test_selected.json"
    SELECT_RATIO = 0.1
    TEST_MODEL = "SpeechMedAssist2-2k"

    question_format = "医生，我有个单项选择题想考你一下。\n{question}\n请你直接回答正确答案选项和选项内容。" if TEST_MODEL != "HuatuoGPT2-7B" else "<问>：请回答下面选择题。\n[单项选择题]{question}\n直接输出最终答案。"
    if not os.path.exists(SELECT_DATA_PATH):
        with open(TEST_DATA_PATH, 'r') as f:
            data = json.load(f)

        selected_data = random.sample(data, int(len(data) * SELECT_RATIO))
        with open(SELECT_DATA_PATH, "w", encoding='utf-8') as f:
            json.dump(selected_data, f, indent=4, ensure_ascii=False)

    with open(SELECT_DATA_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    model = None
    if TEST_MODEL == "DISC-MedLLM":
        from inference.DISC_MedLLM import DISC_MedLLM
        model = DISC_MedLLM(model_path="../../weight/DISC-MedLLM")
    elif TEST_MODEL == "HuatuoGPT2-7B":
        from inference.HuatuoGPT2 import HuatuoGPT2
        model = HuatuoGPT2(model_path="../../weight/HuatuoGPT2-7B")
    elif TEST_MODEL == "ShizhenGPT":
        from inference.ShizhenGPT import ShizhenGPT
        model = ShizhenGPT(model_path="../../weight/ShizhenGPT-7B-Omni", max_new_tokens=32)
    elif TEST_MODEL == "Llama-Omni2-7B":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/LLaMA-Omni2-7B-Bilingual/")
    elif TEST_MODEL == "SpeechMedAssist1":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage1")
    elif TEST_MODEL == "SpeechMedAssist2":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage2")
    elif TEST_MODEL == "SpeechMedAssist2-2k":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage2/checkpoint-5000")
    elif TEST_MODEL == "SpeechMedAssist2-audio-only":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage2-audio-only-with-assist/")
    elif TEST_MODEL == "SpeechGPT2":
        from inference.SpeechGPT2 import SpeechGPT2
        model = SpeechGPT2(model_path="../../../SpeechGPT-2.0-preview", max_new_token=64)
    elif TEST_MODEL == "Baichuan2":
        from inference.Baichuan2 import Baichuan2
        model = Baichuan2(model_path=f"../../weight/Baichuan2-7B")
    elif TEST_MODEL == "Qwen2-Audio":
        from inference.Qwen2_Audio import Qwen2_Audio
        model = Qwen2_Audio(model_path="../../weight/Qwen2-Audio-7B-Instruct")
    elif TEST_MODEL == "GLM4-Voice":
        from inference.GLM4_Voice import GLM4_Voice
        model = GLM4_Voice(model_path="../../weight/GLM4-Voice", max_new_token=64)
    elif TEST_MODEL == "Zhongjing":
        from inference.Baichuan2 import Baichuan2
        model = Baichuan2(model_path=f"../../weight/Zhongjing")
    elif TEST_MODEL == "Baichuan-M2":
        from inference.Baichuan_M2 import Baichuan_M2
        model = Baichuan_M2(model_path=f"../../weight/Baichuan-M2-32B")

    generate_answer(questions, model, f"{TEST_MODEL}_answer.json")
    evaluation(f"{TEST_MODEL}_answer.json")


