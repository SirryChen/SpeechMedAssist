import json
import random
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import usage_policy_med_harm, rubric, eval_prompt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../")
sys.path.append(project_path)


def generate_answer(question_list, model, output_path):
    answer_list = []
    for item in tqdm(question_list, desc=f"generating responses from {TEST_MODEL}"):
        messages = [{"from": "user", "value": item["conversations"][0]["value"],
                     "speech": os.path.join("../", item["conversations"][0]["speech"])}]
        reply = model.reply(messages)

        item["response"] = reply["text"]

        if TEST_MODEL in ["HuatuoGPT2-7B", "DISC-MedLLM", "Baichuan2", "Zhongjing"]:
            item["asr_result"] = reply["asr"]

        answer_list.append(item)

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(answer_list, f, indent=4, ensure_ascii=False)


def eval(output_path):
    """
    使用qwen_72B模型和eval_prompt对模型回答进行评估
    """
    # 加载模型回答数据
    with open(output_path, 'r', encoding='utf-8') as f:
        model_responses = json.load(f)

    # 加载qwen_72B模型
    # 注意：模型路径需要您后续补充
    qwen_72B_model_path = "../../weight/Qwen2.5-72B-Instruct"  # 请替换为实际路径

    model = AutoModelForCausalLM.from_pretrained(
        qwen_72B_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(qwen_72B_model_path)

    evaluated_results = []

    for item in tqdm(model_responses, desc="Evaluating responses"):
        question = item['question']
        model_response = item['response']

        # 构造评估提示
        evaluation_prompt = eval_prompt.format(
            question=question,
            response=model_response,
            usage_policy=usage_policy_med_harm,
            rubric=rubric
        )

        # 构造消息
        messages = [
            {"role": "user", "content": evaluation_prompt}
        ]

        # 应用聊天模板
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成评估结果
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        evaluation_result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 添加评估结果到项目中
        item["evaluation"] = evaluation_result
        evaluated_results.append(item)

    # 保存包含评估结果的数据
    eval_output_path = output_path.replace('.json', '_evaluated.json')
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluated_results, f, indent=4, ensure_ascii=False)

    print(f"评估完成，结果已保存到: {eval_output_path}")


def extract_scores_and_average(evaluated_file_path):
    """
    从评估结果文件中提取评分并计算平均分
    """
    # 重新读取评估后的文件
    with open(evaluated_file_path, 'r', encoding='utf-8') as f:
        evaluated_data = json.load(f)

    scores = []
    valid_items = 0

    print(f"从文件 {evaluated_file_path} 中提取评分...")

    for item in evaluated_data:
        evaluation_text = item.get("evaluation", "")
        if not evaluation_text:
            print(f"警告: 缺少评估文本，跳过该项")
            continue

        # 使用正则表达式提取评分
        score_match = re.search(r"score:\s*(\d+)", evaluation_text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            scores.append(score)
            # 将评分添加到项目中（如果尚未存在）
            item["score"] = score
            valid_items += 1
        else:
            print(f"警告: 无法从评估结果中提取评分，评估文本预览: {evaluation_text[:100]}...")

    # 计算并输出平均分
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"模型 {evaluated_file_path}:")
        print(f"  有效评分数量: {len(scores)}")
        print(f"  平均得分: {average_score:.2f}")
        print(f"  最低分: {min(scores)}")
        print(f"  最高分: {max(scores)}")
        print("-" * 50)

        # 更新保存的文件，包含评分信息
        with open(evaluated_file_path, 'w', encoding='utf-8') as f:
            json.dump(evaluated_data, f, indent=4, ensure_ascii=False)

        return average_score, scores
    else:
        print(f"模型 {evaluated_file_path}: 未提取到任何有效评分")
        return None, []


if __name__ == "__main__":
    TEST_DATA_PATH = "../../dataset/SpeechMedDataset/test_s2t_Med_Safety.json"
    SELECT_DATA_PATH = "test_s2t_50_Med_Safety.json"
    SELECT_RATIO = 1
    TEST_MODEL = "SpeechMedAssist2-2k"
    model_name_list = ["SpeechMedAssist1", "Llama-Omni2-7B", "SpeechMedAssist2", "HuatuoGPT2-7B", "SpeechGPT2",
                       "Qwen2-Audio", "GLM4-Voice"]

    if not os.path.exists(SELECT_DATA_PATH):
        with open(TEST_DATA_PATH, 'r') as f:
            data = json.load(f)
            if os.path.exists("test_t2t_50_Med_Safety.json"):
                with open("test_t2t_50_Med_Safety.json", 'r') as f_t2t:
                    t2t_selected_data = json.load(f_t2t)
                selected_idxs = [item['idx'] for item in t2t_selected_data]
                select_data = [item for item in data if item['idx'] in selected_idxs]
            else:
                select_data = random.sample(data, 50)

            with open(SELECT_DATA_PATH, 'w') as f:
                json.dump(select_data, f, indent=4, ensure_ascii=False)

    with open(SELECT_DATA_PATH, 'r') as f:
        data = json.load(f)
        for item in data:
            item['question'] = item["conversations"][0]["value"]

    model = None
    if TEST_MODEL == "DISC-MedLLM":
        from inference.DISC_MedLLM import DISC_MedLLM
        model = DISC_MedLLM(model_path="../../weight/DISC-MedLLM", input_speech=True)
    elif TEST_MODEL == "HuatuoGPT2-7B":
        from inference.HuatuoGPT2 import HuatuoGPT2
        model = HuatuoGPT2(model_path="../../weight/HuatuoGPT2-7B", input_speech=True)
    elif TEST_MODEL == "Llama-Omni2-7B":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/LLaMA-Omni2-7B-Bilingual/", input_speech=True)
    elif TEST_MODEL == "SpeechMedAssist1":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage1", input_speech=True)
    elif TEST_MODEL == "SpeechMedAssist2":
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage2-ratio-1/", input_speech=True)
    elif TEST_MODEL == "SpeechMedAssist2-2k":      # 0.7796
        from inference.SpeechMedAssist import SpeechMedAssist
        model = SpeechMedAssist(model_path="../../weight/stage2/checkpoint-5000")
    elif TEST_MODEL == "SpeechGPT2":
        from inference.SpeechGPT2 import SpeechGPT2
        model = SpeechGPT2(model_path="../../../SpeechGPT-2.0-preview", input_speech=True)
    elif TEST_MODEL == "Baichuan2":
        from inference.Baichuan2 import Baichuan2
        model = Baichuan2(model_path=f"../../weight/Baichuan2-7B", input_speech=True)
    elif TEST_MODEL == "Qwen2-Audio":
        from inference.Qwen2_Audio import Qwen2_Audio
        model = Qwen2_Audio(model_path="../../weight/Qwen2-Audio-7B-Instruct", input_speech=True)
    elif TEST_MODEL == "GLM4-Voice":
        from inference.GLM4_Voice import GLM4_Voice
        model = GLM4_Voice(model_path="../../weight/GLM4-Voice/", input_speech=True)
    elif TEST_MODEL == "Baichuan-M2":
        from inference.Baichuan_M2 import Baichuan_M2
        model = Baichuan_M2(model_path=f"../../weight/Baichuan-M2-32B", input_speech=True)
    elif TEST_MODEL == "Kimi-Audio":
        from inference.KimiAudio import KimiAudio
        model = KimiAudio(model_path=f"../../weight/Kimi-Audio-7B-Instruct", input_speech=True)
    elif TEST_MODEL == "ShizhenGPT":
        from inference.ShizhenGPT import ShizhenGPT
        model = ShizhenGPT(model_path=f"../../weight/ShizhenGPT-7B-Omni", input_speech=True)

    generate_answer(data, model, f"{TEST_MODEL}_s2t_answer.json")
    del model  # 删除模型引用
    model = None  # 确保没有其他变量引用模型
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理 GPU 缓存
    eval(f"{TEST_MODEL}_s2t_answer.json")
    avg_score, scores = extract_scores_and_average(f"{TEST_MODEL}_s2t_answer_evaluated.json")
    print(f"平均得分: {avg_score:.2f}")
    print(f"所有得分: {scores}")

