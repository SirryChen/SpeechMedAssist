import os
import torch
import librosa
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer
)
from copy import deepcopy


class ShizhenGPT:
    def __init__(self, model_path, input_speech=True, output_speech=False, speech_output_dir="ShizhenGPT_output", device="cuda:0", dtype="bfloat16", max_new_tokens=1024):
        """
        语音-文本交互模型
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.processor.chat_template = self.processor.tokenizer.chat_template
        self.max_new_tokens = max_new_tokens
        self.device = self.model.device

        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir

    def process_audio(self, audio_path):
        """
        加载并重采样音频文件
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
            if y.ndim > 1:
                y = y[:, 0]  # 取单声道
            return y
        except Exception as e:
            print(f"[Error] processing audio {audio_path}: {e}")
            return None

    def build_messages(self, messages):
        """
        构造对话历史，带上音频占位符
        """
        history = []
        for msg in messages:
            role = msg["from"]
            value = msg.get("value", "")
            speech_path = msg.get("speech")

            if role in ["human", "user"]:
                if speech_path is not None and self.input_speech:
                    history.append({"role": "user", "content": "<|audio_bos|><|AUDIO|><|audio_eos|>"})
                else:
                    history.append({"role": "user", "content": value})
            elif role == "assistant":
                history.append({"role": "assistant", "content": value})
        return history

    def prepare_inputs(self, messages):
        """
        遍历所有消息，收集 audios，并生成模型输入
        """
        history = self.build_messages(messages)

        # 收集所有语音数据
        audio_data = []
        for msg in messages:
            if msg.get("speech") is not None and self.input_speech:
                audio_data.append(self.process_audio(msg["speech"]))

        text = self.processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

        input_data = self.processor(
            text=[text],
            audios=audio_data if audio_data else None,
            return_tensors="pt",
            padding=True
        )
        for k, v in input_data.items():
            if hasattr(v, "to"):
                input_data[k] = v.to(self.device)
        return input_data

    def reply(self, messages, round_idx=1, temperature=0.7, top_p=0.9):
        """
        非流式推理
        """
        input_data = self.prepare_inputs(messages)

        outputs = self.model.generate(
            **input_data,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        output_ids = outputs[0][input_data["input_ids"].shape[-1]:]

        reply_text = self.processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        return {"text": reply_text, "speech": None}

    def stream_reply(self, messages, temperature=0.7, top_p=0.9):
        """
        流式推理
        """
        input_data = self.prepare_inputs(messages)

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
        generation_kwargs = dict(
            **input_data,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        response = ""
        for new_text in streamer:
            response += new_text
            yield {"text": response, "speech": None}


if __name__ == "__main__":
    model = ShizhenGPT(model_path="../weight/ShizhenGPT-7B-Omni")

    # print(model.reply([
    #     {
    #         "from": "user",
    #         "value": "",
    #         "speech": "./20250920_115710.wav"
    #     }
    # ]))

    # 文本测试
    # print(model.reply([{"from": "user", "value": "你好，请介绍一下中医的望闻问切。"}]))

    # 单轮语音测试
    print(model.reply([
        {
            "from": "user",
            "value": "",
            "speech": "./test2.mp3"
        }
    ]))

    # 多轮语音测试
    # messages = [
    #     {
    #         "from": "human",
    #         "value": "医生，我上个月10号做了流产手术，25号去医院复查说挺好的，可是下午开始流褐色的血，还有点小腹胀痛，这是怎么回事？",
    #         "speech": "../dataset/SpeechMedDataset/wav/huatuogpt2_sft_3006-0.wav"
    #     },
    #     {
    #         "from": "assistant",
    #         "value": "手术后有些出血和腹痛是正常的，不过你这个情况我也要了解清楚。你这出血是断断续续的吗？有没有发热或者觉得特别不舒服？"
    #     },
    #     {
    #         "from": "human",
    #         "value": "出血就是褐色的那种，不是很多，有时候肚子有点胀胀的疼，没有发烧，就是感觉不太对劲。",
    #         "speech": "../dataset/SpeechMedDataset/wav/huatuogpt2_sft_3006-2.wav"
    #     }
    # ]
    # print(model.reply(messages))
