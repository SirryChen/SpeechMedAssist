import os
import torch
import torchaudio

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../SpeechGPT-2.0-preview")
sys.path.append(project_path)

import json
from transformers import HfArgumentParser
from mimo_qwen2_grouped import MIMOModelArguments
from demo_gradio import Inference


class SpeechGPT2:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="SpeechGPT2_output", max_new_token=None):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)

        codec_ckpt_path = os.path.join(model_path, "SpeechGPT-2.0-preview-Codec/sg2_codec_ckpt.pkl")
        codec_config_path = os.path.join(model_path, "Codec/config/sg2_codec_config.yaml")
        greeting_jsonl_path = os.path.join(model_path, "extra/greetings.jsonl")
        greeting_line_idx = 0
        model_path = os.path.join(model_path, "SpeechGPT-2.0-preview-7B")
        args = type("Args", (), {
            "model_path": model_path,
            "codec_ckpt_path": codec_ckpt_path,
            "codec_config_path": codec_config_path,
        })()

        parser = HfArgumentParser((MIMOModelArguments,))
        model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        model_args.model_name_or_path = model_path

        self.inference = Inference(
            model_path,
            args,
            model_args,
            codec_ckpt_path,
            codec_config_path,
        )
        if max_new_token is not None:
            self.inference.generate_kwargs["max_new_tokens"] = max_new_token

        with open(greeting_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == greeting_line_idx:
                    greeting = json.loads(line)
                    greeting_text = greeting["text"]
                    greeting_audio = greeting["audio"]
                    break
        self.greeting_text = greeting_text
        self.greeting_audio = greeting_audio
        self.clear_history()

        if not self.input_speech and not self.output_speech:
            self.mod = "t2t"
        elif self.input_speech and self.output_speech:
            self.mod = "s2s"
        elif self.input_speech and not self.output_speech:
            self.mod = "s2t"

        self.round_idx = 0

    def clear_history(self):
        self.inference.clear_history()
        self.inference.set_greeting(self.greeting_text, self.greeting_audio)


    def reply(self, messages, round_idx=None):
        """
        inference有记忆机制，messages 只能一轮一轮地输入
        """
        if round_idx is None:
            self.clear_history()
        elif round_idx <= self.round_idx:
            self.clear_history()
            self.round_idx = round_idx
        else:
            self.round_idx = round_idx

        if self.input_speech:
            speech = messages[-1]["speech"]
            reply, wav = self.inference.forward(
                task="thought",
                input=speech,
                text=None if messages[-1].get("added_value") is None else messages[-1]["added_value"].replace("[cough]{cough}", ""),
                mode=self.mod,
            )
        else:
            text = messages[-1]["value"]
            reply, wav = self.inference.forward(
                task="thought",
                input=text,
                text=text,
                mode=self.mod,
            )

        if self.output_speech:
            speech = self.save_audio(wav[1], round_idx)
        else:
            speech = None

        return {"text": reply, "speech": speech}

    def save_audio(self, audio_tensor, round_idx):
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
        wav_tensor = torch.from_numpy(audio_tensor).float()

        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)  # 添加通道维度

        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        torchaudio.save(output_path, wav_tensor, 24000)

        return output_path


if __name__ == "__main__":
    model = SpeechGPT2(model_path="../../SpeechGPT-2.0-preview", input_speech=True, output_speech=False)
    print(model.reply([
        {
            "from": "user",
            "value": "",
            "speech": "./test2.mp3"     # line 283 in the code of SpeechGPT2 demo_gradio.py -> add  'or input.endswith(".mp3")'
        }
    ]))