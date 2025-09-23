import json
import argparse
import logging
from copy import deepcopy
from patient_handler import PatientModel
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../")
sys.path.append(project_path)

from inference.utils import ASRModel


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

def run_dialogue(
    doctor_model,
    patient_model: PatientModel,
    output_path: str,
    logger
):
    all_conversations = []

    for conv_id in range(patient_model.get_num()):

        logger.info(f"\n========== 开始第 {conv_id} 组对话 ==========\n")
        doctor_model.conv_id = conv_id
        patient_model.conv_id = conv_id
        patient_question = patient_model.get_question()

        logger.info(f"[Patient] {patient_question}")

        dialog = {
            "conv_id": conv_id,
            "ref_response": patient_model.get_ref_response(),
            "conversations": [
                {
                    "from": "user",
                    "value": patient_question["text"],
                    "speech": patient_question["speech"]
                }
            ]
        }
        conversations = deepcopy(dialog["conversations"])
        doctor_reply = doctor_model.reply(conversations)

        dialog["conversations"].append({"from": "assistant", "value": doctor_reply["text"], "speech": doctor_reply["speech"],
                                        "ASR": asr_model.speech2text(doctor_reply["speech"]) if args.output_ASR and doctor_reply["speech"] is not None else None})

        logger.info(f"[Doctor ️] {doctor_reply['text']}")

        all_conversations.append(dialog)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=4)
    logger.info(f"\n✅ 所有对话保存至 {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model", type=str, default="experiment-align")
    parser.add_argument("--input_speech", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--output_speech", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--output_ASR", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="设置日志级别")
    args = parser.parse_args()

    # 设置logger
    logger = setup_logger(args.log_level)
    kwargs = {}

    TEST_MODEL = args.test_model
    args.data_path = f"../../dataset/SpeechMedDataset/test_{'s2t' if parser.parse_args().input_speech else 't2t'}_Medical_Encyclopedia.json"
    args.save_path = f"dialog_{'s2t' if args.input_speech else 't2t'}_{TEST_MODEL}.json"

    if args.output_ASR:
        asr_model = ASRModel()

    if TEST_MODEL == "SpeechMedAssist2":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage3"
    elif TEST_MODEL == "SpeechMedAssist1":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage1-final"
    elif TEST_MODEL == "SpeechMedAssist2-2k":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2-2k-final"
    elif TEST_MODEL == "SpeechMedAssist2-audio-only":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2-audio-only-with-assist/"
    elif TEST_MODEL == "SpeechMedAssist2-10000":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2/checkpoint-10000"
    elif TEST_MODEL == "SpeechMedAssist2-20000":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2/checkpoint-20000"
    elif TEST_MODEL == "SpeechMedAssist2-30000":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2/checkpoint-30000"
    elif TEST_MODEL == "SpeechMedAssist2-40000":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2/checkpoint-40000"
    elif TEST_MODEL == "SpeechMedAssist2-50000":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage2/checkpoint-50000"
    elif TEST_MODEL == "SpeechMedAssist2-final":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/stage3/"
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
        kwargs["max_new_token"] = 100
    elif TEST_MODEL == "Llama-Omni2-7B":
        from inference.SpeechMedAssist import SpeechMedAssist as DoctorModel
        args.doctor_model_path = "../../weight/LLaMA-Omni2-7B-Bilingual/"
    elif TEST_MODEL == "KimiAudio":
        from inference.KimiAudio import KimiAudio as DoctorModel
        args.doctor_model_path = "../../weight/Kimi-Audio-7B-Instruct/"
    elif TEST_MODEL == "ShizhenGPT":
        from inference.ShizhenGPT import ShizhenGPT as DoctorModel
        args.doctor_model_path = "../../weight/ShizhenGPT-7B-Omni"

    doctor = DoctorModel(
        model_path=args.doctor_model_path,
        input_speech=args.input_speech,
        output_speech=args.output_speech,
        speech_output_dir=f"wav_{TEST_MODEL}",
        **kwargs
    )

    patient = PatientModel(
        data_path=args.data_path,
        patient_speech=args.input_speech,
        speech_output_dir=f"wav_{TEST_MODEL}"
    )

    run_dialogue(doctor, patient, args.save_path, logger)
