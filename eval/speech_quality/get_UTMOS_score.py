import torch
import json
import torchaudio
import os
from tqdm import tqdm

# 设置权重缓存目录
os.environ["TORCH_HOME"] = "../../weight/SpeechMOS/"
os.makedirs("../../weight/SpeechMOS/", exist_ok=True)


def get_mos(audio_path):
    with open(audio_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device)

    total_score = 0.0
    count = 0

    # 逐条音频单独评测
    for item in tqdm(data, desc="Processing"):
        wave_path = item["conversations"][1]["speech"]
        full_wave_path = os.path.join(args.data_base_path, wave_path)

        wave, sample_rate = torchaudio.load(full_wave_path)
        wave = wave.to(device)

        score = predictor(wave, sample_rate)
        total_score += score.item()
        count += 1

    average_score = total_score / count if count > 0 else 0.0
    print(f"{args.test_model} 平均UTMOS：{average_score}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model", type=str, default="GLM4-Voice")
    parser.add_argument("--data_base_path", type=str, default="../single_round")

    args = parser.parse_args()
    data_path = f"{args.data_base_path}/dialog_s2t_{args.test_model}.json"

    get_mos(data_path)
