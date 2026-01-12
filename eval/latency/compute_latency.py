import json
import argparse
import random
import logging
import os
import sys
import time
import statistics

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../"))


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


def compute_time_to_first_token(model, message):
    """
    计算从模型输入到第一个输出token的时间
    """
    if args.stream_support:
        response = next(model.stream_reply(message))
        start_time = response["start_time"]
    else:
        start_time = time.time()
        _ = model.reply(message)
    end_time = time.time()

    return end_time - start_time


"""
python compute_latency.py --test_model KimiAudio
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model", type=str, default="Qwen2-Audio")
    parser.add_argument("--input_speech", type=bool, default=True)
    parser.add_argument("--output_speech", type=bool, default=True)
    parser.add_argument("--repeat", type=int, default=5, help="重复测试次数")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="设置日志级别")
    args = parser.parse_args()

    logger = setup_logger(args.log_level)

    TEST_MODEL = args.test_model

    kwargs = {}

    if TEST_MODEL == "SpeechMedAssist":
        from inference.SpeechMedAssist import SpeechMedAssist as Model
        args.doctor_model_path = "../../weight/stage3"
        args.stream_support = True
    elif TEST_MODEL == "GLM4-Voice":
        from inference.GLM4_Voice import GLM4_Voice as Model
        args.doctor_model_path = "../../weight/GLM4-Voice"
        args.stream_support = True
    elif TEST_MODEL == "KimiAudio":
        from inference.KimiAudio import KimiAudio as Model
        args.doctor_model_path = "../../weight/Kimi-Audio-7B-Instruct"
        args.stream_support = False
    elif TEST_MODEL == "SpeechGPT2":
        from inference.SpeechGPT2 import SpeechGPT2 as Model
        args.doctor_model_path = "../../weight/SpeechGPT-2.0-preview"
        args.stream_support = False
    elif TEST_MODEL == "Zhongjing":
        from inference.Zhongjing import Zhongjing as Model
        args.doctor_model_path = "../../weight/Zhongjing"
        args.stream_support = False
    elif TEST_MODEL == "Llama-Omni2-7B":
        from inference.SpeechMedAssist import SpeechMedAssist as Model
        args.doctor_model_path = "../../weight/LLaMA-Omni2-7B-Bilingual/"
        args.stream_support = True
    elif TEST_MODEL == "Qwen2-Audio":
        from inference.Qwen2_Audio import Qwen2_Audio as Model
        args.doctor_model_path = "../../weight/Qwen2-Audio-7B-Instruct"
        args.stream_support = False

    model = Model(
        model_path=args.doctor_model_path,
        input_speech=args.input_speech,
        output_speech=args.output_speech,
        speech_output_dir=f"wav",
        **kwargs
    )

    message = [{
        "from": "user",
        "value": "",
        "speech": "./test2.mp3"
    }]

    # 多次计算并取平均
    times = []
    for i in range(args.repeat):
        t = compute_time_to_first_token(model, message)
        logger.info(f"[Run {i+1}] Time to first token: {t:.4f} seconds")
        times.append(t)

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    logger.info(f"Average TTFT over {args.repeat} runs: {avg_time:.4f} ± {std_time:.4f} seconds")
