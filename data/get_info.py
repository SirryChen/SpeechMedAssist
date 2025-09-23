import argparse
import json
import re
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer


def identify_age_gender(args):
    """
    使用大模型识别已筛选对话中的患者性别和年龄段。
    """

    model_path = args.model_path
    input_path = args.data_path
    output_path = args.output_path
    save_steps = args.save_steps

    age_gender_prompt_template = (
        "{conversation}\n\n"
        "你是一位医学对话分析专家，请根据上述医患对话内容，结合患者提到的症状、用词及描述，推理出患者的性别和年龄段。\n\n"
        "请遵循以下判断逻辑进行推断：\n"
        "- 如果提到与女性特有疾病（如月经、怀孕、妇科等）相关的信息，性别应为“女”；\n"
        "- 如果提到如前列腺、睾丸等男性特有问题，性别应为“男”；\n"
        "- 如果症状暗示与年龄有关（如青春期、老年斑、骨质疏松等），请结合上下文判断年龄段；\n"
        "- 若信息不足，请谨慎选择“未知”。\n\n"
        "性别选项：[男，女，未知]；年龄段选项：[少年，青年，成年，老年，未知]。\n\n"
        "请严格按照以下格式输出：\n性别：<男/女/未知>\n年龄段：<少年/青年/成年/老年/未知>\n"
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(input_path, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)

    annotated_data = []

    with tqdm(total=len(filtered_data), desc="annotating age/gender") as pbar:
        for idx, data in enumerate(filtered_data):
            conversation = ""
            for content in data["conversations"]:
                role = "患者：" if content["from"] in ["human", "user"] else "医生："
                conversation += f"{role}{content['value']}\n"

            prompt = age_gender_prompt_template.format(conversation=conversation.strip())
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(**model_inputs, max_new_tokens=20)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            gender = re.search(r"性别[:：]?(男|女|未知)", generated_text)
            age_group = re.search(r"年龄段[:：]?(少年|青年|成年|老年|未知)", generated_text)

            data["gender"] = gender.group(1) if gender else "未知"
            data["age"] = age_group.group(1) if age_group else "未知"
            annotated_data.append(data)

            if (idx + 1) % save_steps == 0 or (idx + 1) == len(filtered_data):
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(annotated_data, f, indent=4, ensure_ascii=False)

            pbar.update(1)

    # ======= 最终统计性别与年龄段分布 =======
    gender_counter = Counter([d["gender"] for d in annotated_data])
    age_counter = Counter([d["age"] for d in annotated_data])
    total = len(annotated_data)

    print("\n=== 性别分布 ===")
    for gender in ["男", "女", "未知"]:
        count = gender_counter[gender]
        print(f"{gender}：{count} ({count / total:.2%})")

    print("\n=== 年龄段分布 ===")
    for age in ["少年", "青年", "成年", "老年", "未知"]:
        count = age_counter[age]
        print(f"{age}：{count} ({count / total:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../weight/Qwen2.5-14B-Instruct/")
    # parser.add_argument("--data_path", type=str, default="../dataset/SpeechMedDataset/train_t2t_CMtMedQA.json")
    parser.add_argument("--data_path", type=str, default="../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2.json")
    parser.add_argument("--selected_ratio", type=float, default=1)
    parser.add_argument("--selected_data_path", type=str, default="../dataset/SpeechMedDataset/selected_t2t.json")
    # parser.add_argument("--output_path", type=str, default="../dataset/SpeechMedDataset/annotated_CMtMedQA_t2t.json")
    parser.add_argument("--output_path", type=str, default="../dataset/SpeechMedDataset/annotated_HuatuoGPT2_t2t.json")
    parser.add_argument("--save_steps", type=int, default=2000)
    args = parser.parse_args()

    identify_age_gender(args)
