import json
import argparse
import random
import logging
import re
from patient_handler import PatientModel
from prompt import patient_reply_prompt_MedDG, patient_desc_prompt_MedDG, patient_desc_prompt_AIHospital, patient_reply_prompt_AIHospital
from copy import deepcopy
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../")
sys.path.append(project_path)


def setup_logger(level):
    """设置logger"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_base_info(file_path, select_path, select_ratio):
    if not os.path.exists(select_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data =  json.load(f)
        select_data = random.sample(data, int(len(data) * select_ratio) if int(len(data) * select_ratio) > 0 else 1)
        with open(select_path, "w", encoding="utf-8") as f:
            json.dump(select_data, f, indent=4, ensure_ascii=False)

    else:
        with open(select_path, 'r', encoding='utf-8') as f:
            select_data = json.load(f)
        for i, item in enumerate(select_data):
            item["idx"] = i

    return select_data


def sharegpt2text(sharegpt_data):
    """
    process sharegpt data to "患者：...\n医生：...\n"
    """
    text = ""
    conv = sharegpt_data['conversations']
    for i, item in enumerate(conv):
        if i % 2 == 0:
            text += f"患者：{item['value']}\n"
        else:
            text += f"医生：{item['value']}\n"
    return text

def save_as_sharegpt(conversations, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

def age_group_map(age):
    if age is None:
        return None
    if age <= 17:
        age_group = "少年"
    elif age <= 30:
        age_group = "青年"
    elif age <= 59:
        age_group = "成年"
    else:
        age_group = "老年"
    return age_group

def run_dialogue(
    base_info_list,
    doctor_model,
    patient_model: PatientModel,
    max_turns: int,
    output_path: str,
    logger
):
    all_conversations = []

    for conv_id, base_info in enumerate(base_info_list):
        if args.patient_profile == "MedDG":
            info = sharegpt2text(base_info)
            patient_model.patient_gender = base_info.get("gender")
            patient_model.patient_age = age_group_map(base_info.get("age"))
        elif args.patient_profile in ["AIHospital", "demo"]:
            info = "<基本信息> {}\n".format(base_info["profile"])
            medical_record = base_info["medical_record"]
            if "现病史" in medical_record:
                info += "<现病史> {}\n".format(medical_record["现病史"].strip())
            if "既往史" in medical_record:
                info += "<既往史> {}\n".format(medical_record["既往史"].strip())
            if "个人史" in medical_record:
                info += "<个人史> {}\n".format(medical_record["个人史"].strip())

            pattern = r"性别[:：]\s*(男|女).*?年龄[:：]\s*(\d+)"
            match = re.search(pattern, base_info["medical_record"]["一般资料"], re.S)
            if match:
                patient_model.patient_gender = match.group(1)
                patient_model.patient_age = age_group_map(int(match.group(2)))

        logger.info(f"\n========== 开始第 {conv_id+1} 组对话 ==========\n")
        doctor_model.conv_id = conv_id
        patient_model.conv_id = conv_id
        logger.info(f"\n{'=' * 50}\n原始信息：\n {info}{'=' * 50}\n")

        patient_question = patient_model.desc(info)
        dialog = {
            "conv_id": conv_id,
            "base_info": info,
            "conversations": [
                {
                    "from": "user",
                    "value": patient_question["text"],
                    "speech": patient_question["speech"]
                }
            ]
        }
        logger.info(f"[Patient] {patient_question['text']}")

        for round_idx in range(max_turns):
            # 医生回复
            conversations = deepcopy(dialog["conversations"])
            doctor_reply = doctor_model.reply(conversations, round_idx)
            logger.info(f"[Doctor ️] {doctor_reply['text']}")
            dialog["conversations"].append({"from": "assistant", "value": doctor_reply["text"], "speech": doctor_reply["speech"]})

            if round_idx + 1 == max_turns:
                break
            # 患者继续提问
            patient_reply = patient_model.reply(info, sharegpt2text(dialog), round_idx+1)
            logger.info(f"[Patient] {patient_reply['text']}")
            dialog["conversations"].append({"from": "user", "value": patient_reply['text'], "speech": patient_reply["speech"]})

            if "结束对话" in patient_reply['text']:
                break

        all_conversations.append(dialog)
        patient_model.patient_gender = None
        patient_model.patient_age = None

    save_as_sharegpt(all_conversations, output_path)
    logger.info(f"\n✅ 所有对话保存至 {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model", type=str, default="GLM4-Voice")
    parser.add_argument("--patient_model_path", type=str, default="../../weight/Qwen2.5-72B-Instruct")
    parser.add_argument("--base_info_path", type=str, default="demo_AIHospital.json", choices=["../../dataset/MedDG/MedDG-sharegpt-test.json", "../../dataset/AIHospital/patients.json", "demo_AIHospital.json"])
    parser.add_argument("--ref_wav_path", type=str, default="../../dataset/ref_audio/Aishell-2018A-EVAL/spk_info.json")
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--input_speech", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--output_speech", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--patient_profile", type=str, default="demo", choices=["MedDG", "AIHospital", "demo"])
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="设置日志级别")
    args = parser.parse_args()

    # 设置logger
    logger = setup_logger(args.log_level)

    TEST_MODEL = args.test_model
    args.save_path = f"dialog_{args.patient_profile}_{'s2t' if args.input_speech else 't2t'}_{TEST_MODEL}.json"

    if TEST_MODEL == "SpeechMedAssist2":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage3/"
    elif TEST_MODEL == "SpeechMedAssist2-audio-only":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2-audio-only-with-assist/"
    elif TEST_MODEL == "SpeechMedAssist2-audio-only-wo-assistant":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2-audio-only/"
    elif TEST_MODEL == "SpeechMedAssist1":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage1-final"
    elif TEST_MODEL == "SpeechMedAssist2-2k":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2-2k-final"
    elif TEST_MODEL == "DISC_MedLLM":
        from inference.DISC_MedLLM import DISC_MedLLM as DoctorModel
        args.doctor_model_path = "../../weight/DISC-MedLLM"
    elif TEST_MODEL == "HuatuoGPT2":
        from inference.HuatuoGPT2 import HuatuoGPT2 as DoctorModel
        args.doctor_model_path = "../../weight/HuatuoGPT2-7B"
    elif TEST_MODEL == "Baichuan2":
        from inference.Baichuan2 import Baichuan2 as DoctorModel
        args.doctor_model_path = "../../weight/Baichuan2-7B"
    elif TEST_MODEL == "Qwen2-Audio":
        from inference.Qwen2_Audio import Qwen2_Audio as DoctorModel
        args.doctor_model_path = "../../weight/Qwen2-Audio-7B-Instruct"
    elif TEST_MODEL == "GLM4-Voice":
        from inference.GLM4_Voice import GLM4_Voice as DoctorModel
        args.doctor_model_path = "../../weight/GLM4-Voice"
    elif TEST_MODEL == "SpeechGPT2":
        from inference.SpeechGPT2 import SpeechGPT2 as DoctorModel
        args.doctor_model_path = "../../../SpeechGPT-2.0-preview"
    elif TEST_MODEL == "Zhongjing":
        from inference.Zhongjing import Zhongjing as DoctorModel
        args.doctor_model_path = "../../weight/Zhongjing"
    elif TEST_MODEL == "Llama-Omni2-7B":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/LLaMA-Omni2-7B-Bilingual/"
    elif TEST_MODEL == "KimiAudio":
        from inference.KimiAudio import KimiAudio as DoctorModel
        args.doctor_model_path = "../../weight/Kimi-Audio-7B-Instruct/"
    elif TEST_MODEL == "ShizhenGPT":
        from inference.ShizhenGPT import ShizhenGPT as DoctorModel
        args.doctor_model_path = "../../weight/ShizhenGPT-7B-Omni"


    base_info = load_base_info(args.base_info_path, f"selected_{args.patient_profile}.json", 0.1 if args.patient_profile == "MedDG" else 0.25)

    doctor = DoctorModel(
        model_path=args.doctor_model_path,
        input_speech=args.input_speech,
        output_speech=args.output_speech,
        speech_output_dir=f"wav_{args.patient_profile}_{TEST_MODEL}"
    )

    patient = PatientModel(
        model_path=args.patient_model_path,
        patient_speech=args.input_speech,
        ref_wav_path=args.ref_wav_path,
        speech_output_dir=f"wav_{args.patient_profile}_{TEST_MODEL}",
        desc_prompt=patient_desc_prompt_MedDG if args.patient_profile == "MedDG" else patient_desc_prompt_AIHospital,
        reply_prompt=patient_reply_prompt_MedDG if args.patient_profile == "MedDG" else patient_reply_prompt_AIHospital
    )

    run_dialogue(base_info, doctor, patient, args.max_turns, args.save_path, logger)
