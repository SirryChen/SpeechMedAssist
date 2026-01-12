import argparse
import json
import os
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

filter_prompt_template = (
    "{conversation}\n\n"
    "你是一位专业严谨的医学数据审核专家，请你阅读以上医生与患者之间的医疗对话，判断该对话是否适合用于构建高质量的**医疗语音对话训练数据**。\n\n"
    "请严格按照以下标准逐项进行审核，**只有在全部满足的情况下才应保留**：\n"
    "1. 医学内容准确，符合临床常识，不包含错误信息或误导性建议；\n"
    "2. 患者的陈述清晰、具体、充分、完整，不能过于简短，也不能是一个片段，表达了明确的健康问题或困扰；\n"
    "3. 医生的回复具有针对性，围绕问题展开，并提供了合理建议或判断；\n"
    "4. 对话结构完整，具备良好的提问-应答逻辑，信息衔接自然，不存在明显跳跃、中断或关键信息缺失；\n"
    "5. 内容健康、安全、合规，**不得**包含任何违法违规、歧视、性暗示、暴力、侮辱性语言或其他不当表达；\n"
    "6. 对话内容适合重写为多轮对话，即患者描述病情并回答医生问题，医生分析病情并追问相关问题；\n"
    "7. 对话中**不应包含语音对话中无法完成的操作**，例如上传图片、查看图像、填写表格、点击链接、发送定位等。\n\n"
    "请你严格基于上述6项标准，重点关注患者的问题，综合判断该对话是否适合保留用于构建多轮医疗对话数据集？\n"
    "直接输出判断结果，输出格式为【保留：是/否】。\n"
)

filter_prompt_template2 = (
    "{conversation}\n\n"
    "你是一个医疗对话数据审查助手。现在请你判断以下一组问答是否可以用于构建高质量的医疗对话数据集。请仅根据这一条内容进行判断。\n\n筛选标准如下：\n\n"
    "1. 语言合适性：不包含大量过长的英文专有名词或英文缩写；中文表达清晰，符合中文使用习惯。\n"
    "2. 问题日常化：用户提问应贴近普通人的日常生活或医学常识，不能过于专业或明显超出普通人的知识范围。\n"
    "3. 内容合规性：内容必须健康、安全、合法合规；不得包含任何违法违规、歧视、性暗示、暴力、侮辱性语言或其他不当表达。\n\n"
    "请你严格基于上述3项标准，重点关注患者的问题，综合判断该对话是否适合保留用于构建医疗问答数据集？\n"
    "直接输出判断结果，输出格式为【保留：是/否】。\n"
)


def filter_base(data, question_len=40, image_remove=True):
    image_flag = False
    short_flag = False
    for conversation in data["conversations"]:
        if image_remove and "图像" in conversation["value"]:
            image_flag = True
        if len(conversation["value"]) <= question_len and conversation["value"].count("?") < 3 and conversation["value"].count("？") < 3:
            short_flag = True
    return image_flag or short_flag


def filter_data_by_llm(args):
    """
    使用大模型筛选出内容合理、完整、符合语音风格的对话数据。
    """

    model_path = args.model_path
    data_path = args.data_path
    selected_ratio = args.selected_ratio
    question_len = args.question_len
    image_remove = args.image_remove
    selected_data_path = args.selected_data_path
    output_path = args.output_path
    save_steps = args.save_steps
    removed_data_path = args.removed_data_path

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
        with open(selected_data_path, "w", encoding="utf-8") as f:
            json.dump(selected_data, f, indent=4, ensure_ascii=False)

    def load_or_empty(path):
        return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []

    filtered_data = load_or_empty(output_path)
    removed_data = load_or_empty(removed_data_path)

    processed_count = len(filtered_data) + len(removed_data)

    with tqdm(total=len(selected_data), initial=processed_count) as pbar:
        for idx in range(processed_count, len(selected_data)):
            data = selected_data[idx]

            if filter_base(data, question_len, image_remove):
                removed_data.append(data)
            else:
                conversation = ""
                for content in data["conversations"]:
                    role = "患者：" if content["from"] in ["human", "user"] else "医生："
                    conversation += f"{role}{content['value']}\n"

                prompt = filter_prompt_template2.format(conversation=conversation.strip())
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(**model_inputs, max_new_tokens=20)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                keep_flag = re.search(r"保留：是", generated_text)

                if keep_flag:
                    filtered_data.append(data)
                else:
                    removed_data.append(data)

            if (idx + 1) % save_steps == 0 or (idx + 1) == len(selected_data):
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_data, f, indent=4, ensure_ascii=False)
                with open(removed_data_path, "w", encoding="utf-8") as f:
                    json.dump(removed_data, f, indent=4, ensure_ascii=False)

            pbar.update(1)
            pbar.set_description(f"保留:{len(filtered_data)} 删除:{len(removed_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../weight/Qwen2.5-14B-Instruct/")
    parser.add_argument("--data_path", type=str, default="../dataset/HuatuoGPT2-SFT/HuatuoGPT2-GPT4-SFT-140K.json")
    # parser.add_argument("--data_path", type=str, default="../dataset/HuatuoGPT2-Pretraining-Instruction/data/HuatuoGPT2_Pretrain_Meidcal_Books_cn.json")
    parser.add_argument("--selected_ratio", type=float, default=0.1)
    parser.add_argument("--question_len", type=int, default=5)
    parser.add_argument("--image-remove", type=bool, default=False)
    parser.add_argument("--selected_data_path", type=str, default="../dataset/SpeechMedDataset/selected_huatuo_pretrain_t2t.json")
    parser.add_argument("--output_path", type=str, default="../dataset/SpeechMedDataset/filtered_huatuo_pretrain_t2t.json")
    parser.add_argument("--removed_data_path", type=str, default="../dataset/SpeechMedDataset/removed_huatuo_pretrain_t2t.json")
    parser.add_argument("--save_steps", type=int, default=5000)
    Args = parser.parse_args()

    filter_data_by_llm(Args)
