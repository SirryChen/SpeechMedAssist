import os
import torch
import whisper
import torchaudio
import argparse
import pyttsx3
from transformers import AutoTokenizer, AutoConfig
from torch.nn.utils.rnn import pad_sequence
from utils import RealTimeVADRecorder, SpeechDecoder

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../")
sys.path.append(project_path)
from model import *
from model.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN


def load_pretrained_model(model_path, output_speech=False):
    model_cls = Speech2SQwen2ForCausalLM if output_speech else SpeechQwen2ForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    # config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = model_cls.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16).cuda()
    return tokenizer, model


def load_and_process_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).permute(1, 0)  # (T, n_mels)
    return mel.cuda().bfloat16(), mel.shape[0]


def synthesize_speech(text, output_path="output.wav"):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    print(f"[🔊] 音频保存在: {output_path}")
    try:
        torchaudio.backend.sox_io_backend.save(output_path, torch.tensor([]), 16000)
    except:
        pass
    os.system(f"start {output_path}" if os.name == "nt" else f"aplay {output_path}")


def main(args):
    recorder = RealTimeVADRecorder(sample_rate=16000)
    tokenizer, model = load_pretrained_model(args.model_path, output_speech=args.output_speech)
    speech_decoder = SpeechDecoder(
        model_dir=args.speech_decoder_path,
        hop_len=args.hop_len,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        load_onnx=args.load_onnx,
        prompt_speech_path=args.prompt_speech_path,
    )

    print("=== 🗣️ 交互式多轮对话开始 ===")
    print("输入 'exit' 退出\n")

    conversation = []
    speech_rounds = []  # 存储所有轮次的 mel 特征

    round_idx = 0
    while True:
        round_idx += 1
        if args.input_speech:
            print(f"[🎙️] 第 {round_idx} 轮开始录音，检测到语音后自动开始...")
            mel = recorder.record_audio(save=True, save_path=f"input_{round_idx}.wav")
            speech_rounds.append(mel)
            content = DEFAULT_SPEECH_TOKEN
        else:
            print("[⌨️] 请输入文本内容：")
            text = input().strip()
            if text.lower() == 'exit':
                break
            content = text

        # 构建多轮对话历史
        conversation.append({"role": "user", "content": content})

        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")[0]
        input_ids[input_ids == tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        input_ids = input_ids.unsqueeze(0).to("cuda")

        if args.input_speech and len(speech_rounds) > 0:
            speech_tensor = pad_sequence(speech_rounds, batch_first=True, padding_value=0.0).to("cuda").bfloat16()
            speech_lengths = torch.tensor([mel.shape[0] for mel in speech_rounds]).to("cuda")
        else:
            speech_tensor = None
            speech_lengths = None

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_lengths,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            if args.output_speech:
                output_ids, output_units = outputs
                audio = speech_decoder.generate(output_units, stream=True)
            else:
                output_ids = outputs

        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"[🤖] 模型回答：\n{response_text}")

        conversation.append({"role": "assistant", "content": response_text})

        if args.output_speech:
            import sounddevice as sd
            audio_np = audio.squeeze().cpu().numpy()  # 转为 numpy 格式（1D）
            sd.play(audio_np, samplerate=24000)
            sd.wait()
            print("[🔊] 语音播放完成")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=r"../weight/stage3")
    parser.add_argument("--speech_decoder_path", type=str, default=r'../weight/cosy2_decoder')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--s2s", action="store_true", default=True, help="whether to generate speech")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--hop-len", type=int, default=None)
    parser.add_argument("--load-jit", action="store_true")
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--load-onnx", action="store_true")
    parser.add_argument("--prompt-speech-path", type=str, default=r"prompt_zh.wav")

    # 新增交互开关
    parser.add_argument("--input_speech", action="store_true", default=False, help="whether to input speech")
    parser.add_argument("--output_speech", action="store_true", default=True, help="whether to play speech")

    args = parser.parse_args()

    main(args)
