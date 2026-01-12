import sys
import os
import re
import json
import random
import logging
import argparse
import warnings
import copy

warnings.filterwarnings("ignore")
from loguru import logger
logger.remove()

import torch
import torchaudio
from tqdm import tqdm

sys.path.append('../CosyVoice')
sys.path.append('../CosyVoice/third_party/Matcha-TTS')
sys.path.append('../FishSpeech')  # 添加 Fish-speech 路径

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

from fish_speech.models.text2semantic.inference import init_model as fish_init_model, generate_long
from fish_speech.models.dac.inference import load_model as load_codec_model

logging.disable(logging.CRITICAL)

torch.manual_seed(2025)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2025)

import gc
import torch
def synthesize_with_fish_speech(text, model, decode_one_token, codec_model, device, prompt_text=None, prompt_tokens=None):

    # 设置随机种子
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)

    # 清理显存
    torch.cuda.empty_cache()

    # === 生成语义编码 ===
    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,
        top_p=0.8,
        repetition_penalty=1.1,
        temperature=0.8,
        compile=False,
        iterative_prompt=True,
        chunk_length=300,
        prompt_text=[] if prompt_text is None else prompt_text,
        prompt_tokens=[] if prompt_tokens is None else prompt_tokens,
    )

    codes = []
    for response in generator:
        if response.action == "sample":
            codes.append(response.codes.cpu())  # 移到 CPU，节省显存
        elif response.action == "next":
            break

    if not codes:
        raise RuntimeError("未生成任何语义编码！")

    # 将所有 codes 拼接，并放回 device
    codes_tensor = torch.cat(codes, dim=1).to(device).long()
    indices_lens = torch.tensor([codes_tensor.shape[1]], device=device, dtype=torch.long)

    if prompt_tokens is None:
        prompt_tokens = copy.deepcopy(codes_tensor.cpu())
        prompt_text = text

    # === 解码音频 ===
    with torch.no_grad():  # 避免存储中间梯度
        fake_audios, _ = codec_model.decode(codes_tensor, indices_lens)
        waveform = fake_audios[0, 0].detach().cpu()  # 放回 CPU，释放 GPU 显存

    # 清理变量显存
    del codes_tensor, indices_lens, fake_audios
    torch.cuda.empty_cache()
    gc.collect()

    # === 重采样为 16k ===
    waveform_resampled = torchaudio.functional.resample(
        waveform.unsqueeze(0), orig_freq=codec_model.sample_rate, new_freq=16000
    )

    return waveform_resampled, prompt_text, prompt_tokens


def main(args):
    wav_base_path = "../"
    # 加载说话人信息
    with open(args.spk_info_path, 'r', encoding='utf-8') as f:
        spk_info = json.load(f)

    if not os.path.exists(args.wav_save_path):
        os.mkdir(args.wav_save_path)
    # CosyVoice
    cosyvoice = CosyVoice2(args.cosyvoice_path, load_jit=False, load_trt=False)
    # FishSpeech
    fish_device = "cuda" if torch.cuda.is_available() else "cpu"
    fish_precision = torch.bfloat16
    fish_model, fish_decode_one_token = fish_init_model(
        checkpoint_path=args.fish_ckpt_path,
        device=fish_device,
        precision=fish_precision,
        compile=True
    )
    with torch.device(fish_device):
        fish_model.setup_caches(
            max_batch_size=1,
            max_seq_len=fish_model.config.max_seq_len,
            dtype=next(fish_model.parameters()).dtype
        )
    fish_codec_model = load_codec_model(
        config_name="modded_dac_vq",
        checkpoint_path=args.fish_codec_ckpt,
        device=fish_device
    )
    fish_codec_model.eval()

    data_path = args.data_path
    selected_data_path = args.selected_data_path
    selected_ratio = args.selected_ratio
    output_path = args.output_path
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

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            syn_data = json.load(f)
    else:
        syn_data = []

    processed_num = len(syn_data)
    with tqdm(total=len(selected_data), initial=processed_num, desc="Generating speech") as pbar:
        for idx, item in enumerate(selected_data):
            if idx < processed_num:
                continue
            speech_id = item["idx"]
            gender = item.get("gender")
            age_group = item.get("age")

            if (gender not in ['男', '女'] and age_group not in ["少年", "青年", "成年", "老年"]) or spk_info[gender].get(age_group) is None:
                # NOTE fish-speech
                ref_flag = False
                new_conversations = []
                prompt_text = None
                prompt_tokens = None

                for round_id, turn in enumerate(item["conversations"]):
                    if turn["from"] != "human" and turn["from"] != "user":
                        new_conversations.append(turn)
                        continue
                    if turn.get("value") is not None:
                        question = turn["value"]
                    else:
                        question = turn["text"]
                    if not re.search(r'[\u4e00-\u9fff]', str(question)):
                        new_conversations.append(turn)
                        if question is None:
                            print(item)
                        continue

                    # 使用 fish-speech 合成语音
                    waveform, prompt_text, prompt_tokens = synthesize_with_fish_speech(
                        text=question,
                        model=fish_model,
                        decode_one_token=fish_decode_one_token,
                        codec_model=fish_codec_model,
                        device=fish_device,
                        prompt_text=prompt_text,
                        prompt_tokens=prompt_tokens,
                    )

                    # 保存语音
                    speech_file = os.path.join(args.wav_save_path, f"{speech_id}-{round_id}.wav")
                    torchaudio.save(speech_file, waveform.cpu(), sample_rate=16000)

                    # 构建输出 turn
                    turn = {
                        "from": "human",
                        "value": question,
                        "speech": speech_file
                    }
                    new_conversations.append(turn)

            else:
                # NOTE cosyvoice
                ref_flag = True
                if age_group not in ["少年", "青年", "成年", "老年"]:
                    age_group = random.choice(["少年", "青年", "成年", "老年"])

                new_conversations = []

                # 选择 prompt
                while True:
                    speaker_id = random.choice(list(spk_info[gender][age_group].keys()))
                    prompts = spk_info[gender][age_group][speaker_id]
                    prompt_wav_path, prompt_text = random.choice(prompts)
                    if not os.path.exists(os.path.join(wav_base_path, prompt_wav_path)):
                        continue
                    prompt_speech_16k = load_wav(os.path.join(wav_base_path, prompt_wav_path), 16000)
                    duration = prompt_speech_16k.shape[1] / 16000
                    if duration < 30:
                        break

                for round_id, turn in enumerate(item["conversations"]):
                    if turn["from"] != "human" and turn["from"] != "user":
                        new_conversations.append(turn)
                        continue

                    if turn.get("value") is not None:
                        question = turn["value"]
                    else:
                        question = turn["text"]

                    if not re.search(r'[\u4e00-\u9fff]', str(question)):
                        new_conversations.append(turn)
                        if question is None:
                            print(item)
                        continue

                    # 合成语音
                    speech = next(cosyvoice.inference_zero_shot(
                        question,
                        prompt_text=prompt_text,
                        prompt_speech_16k=prompt_speech_16k,
                        stream=False
                    ))['tts_speech']

                    resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)
                    speech = resampler(speech)

                    # 保存语音
                    speech_file = os.path.join(args.wav_save_path, f"{speech_id}-{round_id}.wav")
                    torchaudio.save(speech_file, speech, 16000)

                    # 构建输出 turn
                    turn = {
                        "from": "human",
                        "value": question,
                        "speech": speech_file
                    }
                    new_conversations.append(turn)

            item["conversations"] = new_conversations
            item["syn_model"] = "CosyVoice" if ref_flag else "FishSpeech"
            syn_data.append(item)

            # 定期保存
            if (len(syn_data) % args.save_interval == 0) or (idx + 1 == len(selected_data)):
                with open(args.output_path, 'w', encoding="utf-8") as f:
                    json.dump(syn_data, f, indent=4, ensure_ascii=False)

            pbar.update(1)

    # 最终保存
    with open(args.output_path, 'w', encoding="utf-8") as f:
        json.dump(syn_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS data generation using CosyVoice and Fish-speech")

    parser.add_argument("--cosyvoice_path", type=str, default="../weight/CosyVoice2-0.5B")
    parser.add_argument("--fish_ckpt_path", type=str, default="../weight/openaudio-s1-mini")
    parser.add_argument("--fish_codec_ckpt", type=str, default="../weight/openaudio-s1-mini/codec.pth")
    parser.add_argument("--data_path", type=str, default="../dataset/SpeechMedDataset/annotated_t2t.json")
    parser.add_argument("--selected_data_path", type=str, default="../dataset/SpeechMedDataset/selected_s2t.json")
    parser.add_argument("--output_path", type=str, default="../dataset/SpeechMedDataset/train_s2t_normal.json")
    parser.add_argument("--wav_save_path", type=str, default="../dataset/SpeechMedDataset/wav/")
    parser.add_argument("--spk_info_path", type=str, default="../dataset/ref_audio/spk_info.json")
    parser.add_argument("--selected_ratio", type=float, default=1)
    parser.add_argument("--save_interval", type=int, default=2000)

    args = parser.parse_args()
    main(args)
