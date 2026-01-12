import argparse
import json
import os
import random
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


rewrite_prompt_CMtMedQA = (
    "现在需要你将上述患者与医生之间的书面医疗对话，重写为更符合“语音对话”特征的版本。\n"
    "请注意以下要求：\n"
    "1. **风格口语化、自然**：避免书面用语，比如“首先”“其次”等，改用日常对话中更自然的表达方式；\n"
    "2. **内容精简**：在尽可能保留有效信息的前提下，对话要简短，避免长段回复。每轮发言尽量控制在100词以内；\n"
    "3. **可发音友好**：去除不可发音的内容，如 Markdown 符号、括号、换行符、列表标志等；\n"
    "4. **保留有效的医学信息**：删减冗余的信息，精炼出有效的诊断逻辑和核心建议；\n"
    "5. **适当修改**：可以适当增加或减少轮次，**务必**删除对话末尾感谢、告别等无效内容，并确保对话末尾发言的为医生；\n"
    "6. 医生由医疗对话助手扮演，不能安排具体检查或治疗，只能给出建议去医院检查什么；\n"
    "7. 不要包含图像观察、表格填写等无法通过语音完成的内容。\n"
    "请你按照上述标准将对话重写为符合语音交流风格的版本，每轮一行，完成后及时停止输出，格式如下：\n"
    "患者：xxx\n医生：xxx\n患者：xxx\n医生：xxx\n……\n"
)

rewrite_prompt_HuatuoGPT2 = (
    "现在需要你将上述【单轮】患者与医生之间的医疗问答，**重写为一段符合语音交流风格的多轮对话**。\n\n"
    "请严格遵循以下要求：\n"
    "1. **模拟真实交流节奏**：患者不应一次性讲出全部病情信息，而应由医生逐步引导提问，患者逐步补充、回应，符合真实就诊场景；\n"
    "2. **医生具有主动问诊能力**：医生应该根据患者的病情描述给出分析，随后追问患者更多病情细节，当信息充足时给出最终诊治意见；\n"
    "3. **风格自然口语化**：避免书面用语（如“首先”“其次”），应使用真实对话中常见的表达方式；\n"
    "4. **内容精简、语音友好**：对话应避免长段回复，每轮控制在100词以内，并去除 Markdown、括号、列表符号等不可发音内容；\n"
    "5. **医学信息充分**：通过多轮问答逐步还原原始问答中的医学关键信息，确保医生最后能够给出清晰的建议或初步判断；\n"
    "6. **医生扮演医疗助手角色**：不能安排具体检查或治疗，只能建议就医并指出可以考虑的科室或检查项目；\n"
    "7. **删去无效内容**：删除感谢、告别等内容，确保最后一轮为医生的医学建议或总结发言；\n"
    "8. **对话轮数控制**：建议重写后的对话轮次控制在 2~5 轮之间，即4~10行，确保内容充分但不冗长；\n\n"
    "请你将原始单轮问答重写为符合上述标准的语音交流风格的版本，务必重视医生的追问，每轮一行，完成后及时停止输出，格式如下：\n"
    "患者：xxx\n医生：xxx\n患者：xxx\n医生：xxx\n……\n"
)

rewrite_prompt_HuatuoGPT2_v2 = (
    "现在需要你将上述【单轮】患者与医生之间的医疗问答，**重写为一段符合语音交流风格的多轮对话**。\n\n"
    "请严格遵循以下要求：\n"
    "1. **模拟真实交流节奏**：患者不应一次性讲出全部病情信息，而应由医生逐步引导提问，患者逐步补充、回应，符合真实就诊场景；\n"
    "2. **医生具有主动问诊能力**：医生应该根据患者的病情描述给出分析，随后追问患者更多病情细节，当信息充足时给出最终诊治意见；\n"
    "3. **风格自然口语化**：避免书面用语（如“首先”“其次”），应使用真实对话中常见的表达方式；\n"
    "4. **内容精简、语音友好**：对话应避免长段回复，每轮控制在100词以内，并去除 Markdown、括号、列表符号等不可发音内容；\n"
    "5. **医学信息充分**：通过多轮问答逐步还原原始问答中的医学关键信息，确保医生最后能够给出清晰的建议或初步判断；\n"
    "6. **医生扮演医疗助手角色**：不能安排具体检查或治疗，只能建议就医并指出可以考虑的科室或检查项目；\n"
    "7. **删去无效内容**：删除感谢、告别等内容，确保最后一轮为医生的医学建议或总结发言；\n"
    "8. **对话轮数控制**：建议重写后的对话轮次控制在 2~5 轮之间，即4~10行，确保内容充分但不冗长；\n\n"
    "请你将原始单轮问答重写为符合上述标准的语音交流风格的版本，务必重视医生的追问，**至少有3次追问**，每轮一行，完成后及时停止输出，格式如下：\n"
    "患者：xxx\n医生：xxx\n患者：xxx\n医生：xxx\n……\n"
)

rewrite_prompt_HuatuoGPT2_pretrain_Meidcal_Books = (
    "请将上述医疗问答重写为**自然简洁的患者与医生之间的单轮问答对话**。\n\n"
    "必须严格遵循以下要求：\n"
    "1. 格式为一问一答：**每个样本只能包含一轮问答**，即患者问一句，医生答一句；\n"
    "2. 患者问题应符合日常问询特征，自然流畅，不包含专业英文术语，避免书面化；\n"
    "3. 医生回答应简洁明了，准确回复患者的问题，**且必须只用一段话表达清楚关键信息**，长度控制在100字以内；\n"
    "4. 去除 Markdown、括号、列表符号等不可发音内容；\n"
    "5. 保留医学信息，删除冗余说明，避免解释性铺垫或重复；\n"
    "6. 不允许医生连续说多句，不允许多轮、不允许寒暄、感谢、告别等无关内容；\n\n"
    "输出格式如下：\n"
    "患者：xxx\n医生：xxx\n\n"
)

rewrite_prompt_MedDG = (
    "现在需要你将上述患者与医生之间的多轮医疗对话，重写为更符合“语音对话”特征的版本。\n"
    "请注意以下要求：\n"
    "1. **风格口语化、自然**：避免书面用语，比如“首先”“其次”等，改用日常对话中更自然的表达方式；\n"
    "2. **内容精简**：在尽可能保留有效信息的前提下，对话要简短，避免长段回复。每轮发言尽量控制在100词以内；\n"
    "3. **可发音友好**：去除不可发音的内容，如 Markdown 符号、括号、换行符、列表标志等；\n"
    "4. **保留有效的医学信息**：删减冗余的信息，精炼出有效的诊断逻辑和核心建议；\n"
    "5. **适当修改**：可以适当增加或减少轮次，**务必**删除对话末尾感谢、告别等无效内容，并确保对话末尾发言的为医生；\n"
    "6. 医生由医疗对话助手扮演，不能安排具体检查或治疗，只能给出建议去医院检查什么；\n"
    "7. 不要包含图像观察、表格填写等无法通过语音完成的内容；\n"
    "8. **模拟真实交流节奏**：患者首先简要描述自己的病情，由医生简要分析病情并追问患者更多的症状，患者逐步补充、回应，医生最后给出诊断结果和综合性建议；\n"
    "9. **对话轮数控制**：建议重写后的对话轮次控制在 4~8 轮之间，即8~16行，确保内容充分但不冗长；\n\n"
    "请你按照上述标准将对话重写为符合语音交流风格的版本，每轮一行，完成后及时停止输出，格式如下：\n"
    "患者：xxx\n医生：xxx\n患者：xxx\n医生：xxx\n……\n"
)

rewrite_prompt_Med_Safety = (
    "现在需要你将上述患者-医生英文对话，翻译为中文。\n"
    "请注意以下要求：\n"
    "1. **风格口语化、自然**：避免书面用语，比如“首先”“其次”等，改用日常对话中更自然的表达方式；\n"
    "2. **内容精简**：在尽可能保留有效信息的前提下，确保医生的回复在100词以内\n"
    "3. **可发音友好**：去除不可发音的内容，如 Markdown 符号、括号、换行符、列表标志等。\n\n"
    "请你按照上述标准将对话翻译为中文，患者与医生各一行，完成后及时停止输出，格式如下：\n"
    "患者：xxx\n医生：xxx\n"
)

rewrite_prompt_HuatuoGPT2_pretrain_Meidcal_Encyclopedia = (
    "请将上述医疗问答重写为**自然简洁的患者与医生之间的单轮问答对话**。\n\n"
    "必须严格遵循以下要求：\n"
    "1. 每个样本只能包含**一轮问答**，即患者问一句，医生答一句；\n"
    "2. 患者的问题可以作简单修改，使其符合日常问询特征，不包含专业英文术语；\n"
    "3. 提取医生回复中的关键信息，使其准确回复患者的问题，**且必须只用几句话表达清楚关键信息**，长度控制在50字左右；\n"
    "4. 去除 ~、Markdown、括号、列表符号等不可发音内容；\n"
    "5. 保留医学信息，避免使用“首先”、“其次”等词语，删除冗余说明，避免解释性铺垫或重复；\n"
    "输出格式如下：\n"
    "患者：xxx\n医生：xxx\n\n"
)

def filter_name(data):
    for conversation in data["conversations"]:
        conversation["value"] = conversation["value"].replace("仲景", "")
    return data


def rewrite_t2t(args):
    """
    使用指定模型将医疗问答对话重写为更符合语音对话风格的数据。
    支持从中断处恢复处理。
    """

    model_path = args.model_path
    data_path = args.data_path
    selected_ratio = args.selected_ratio
    selected_data_path = args.selected_data_path
    output_path = args.output_path
    save_steps = args.save_steps

    if "CMtMedQA" in data_path:
        rewrite_prompt = rewrite_prompt_CMtMedQA
    elif "pretrain_Meidcal_Books" in data_path:
        rewrite_prompt = rewrite_prompt_HuatuoGPT2_pretrain_Meidcal_Books
    elif "MedDG" in data_path:
        rewrite_prompt = rewrite_prompt_MedDG
    elif "med_safety" in data_path:
        rewrite_prompt = rewrite_prompt_Med_Safety
    elif "HuatuoGPT2_Pretrain_Meidcal_Encyclopedia" in data_path:
        rewrite_prompt = rewrite_prompt_HuatuoGPT2_pretrain_Meidcal_Encyclopedia
    else:
        rewrite_prompt = rewrite_prompt_HuatuoGPT2

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if selected_ratio == 1:
        selected_data_path = data_path

    if os.path.exists(selected_data_path):
        with open(selected_data_path, "r", encoding="utf-8") as f:
            selected_data = json.load(f)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        selected_data = random.sample(data_list, int(len(data_list) * selected_ratio))

        if args.exist_data_path is not None:
            with open(args.exist_data_path, "r", encoding="utf-8") as f:
                exist_data = json.load(f)
            exist_idx = [item["id"] for item in exist_data]
            selected_data = [item for item in selected_data if item["id"] not in exist_idx]

        with open(selected_data_path, "w", encoding="utf-8") as f:
            json.dump(selected_data, f, indent=4, ensure_ascii=False)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            rewrite_data = json.load(f)
    else:
        rewrite_data = []

    processed_count = len(rewrite_data)

    with tqdm(total=len(selected_data), initial=processed_count, desc="rewrite data") as pbar:
        for idx in range(processed_count, len(selected_data)):
            data = filter_name(selected_data[idx])

            conversation = ""
            for content in data["conversations"]:
                role = "患者：" if content["from"] in ["human", "user"] else "医生："
                conversation += f"{role}{content['value']}\n"

            prompt = conversation + rewrite_prompt
            messages = [{"role": "user", "content": prompt}]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(**model_inputs, max_new_tokens=256)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            lines = response.strip().split("\n")
            dialog = []
            for line in lines:
                match = re.match(r'(患者|医生)：(.*)', line)
                if match:
                    role = match.group(1)
                    content = match.group(2).strip()
                    dialog.append({
                        "from": "user" if role == "患者" else "assistant",
                        "value": content
                    })

            if len(dialog) % 2 == 1:
                if dialog[-1]["from"] == "user":
                    # 最后一轮是患者，删除
                    dialog = dialog[:-1]
                elif len(dialog) >= 2 and dialog[-1]["from"] == "assistant" and dialog[-2]["from"] == "assistant":
                    # 医生连续发言两轮，删除最后一轮
                    dialog = dialog[:-1]

            rewrite_data.append({
                "idx": data.get("idx") if data.get("idx") is not None else data["id"],
                "id": data.get("id"),
                "cate1": data.get("cate1"),
                "cate2": data.get("cate2"),
                "conversations": dialog
            })

            pbar.update(1)

            if (idx + 1) % save_steps == 0 or (idx + 1) == len(selected_data):
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(rewrite_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../weight/Qwen2.5-32B-Instruct/")
    parser.add_argument("--data_path", type=str, default="../dataset/SpeechMedDataset/filtered_huatuo_pretrain_t2t.json")
    parser.add_argument("--selected_ratio", type=float, default=0.5)
    parser.add_argument("--selected_data_path", type=str, default="../dataset/SpeechMedDataset/selected_filtered_huatuo_pretrain_t2t.json")
    parser.add_argument("--output_path", type=str, default="../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2_pretrain.json")
    parser.add_argument("--save_steps", type=int, default=2000)

    Args = parser.parse_args()
    rewrite_t2t(Args)
