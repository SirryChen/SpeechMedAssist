# use env of SpeechGPT2
import torch
import librosa
import os
import sys
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
sys.path.append(os.path.dirname(__file__))
from utils import TTSModel
import torchaudio


def sharegpt2Qwen2Audio(messages):
    new_messages = []

    for item in messages:
        role = "user" if item["from"] in ["human", "user"] else "assistant"
        content = []

        # 音频（仅 human 可能有 speech）
        if role in ["user", "human"] and item.get("speech") is not None and item["speech"]:
            content.append({
                "type": "audio",
                "audio_url": item["speech"]
            })
            if item.get("added_value") is not None:
                content.append({"type": "text", "text": item["added_value"].replace("[cough]{cough}", "")})

        # 文本
        if "value" in item and item["value"]:
            content.append({
                "type": "text",
                "text": item["value"] if item.get("speech") is None else "你是一个医生，请回复这段患者的语音。"
            })

        # 加入新格式对话
        new_messages.append({
            "role": role,
            "content": content
        })

    return new_messages


class Qwen2_Audio:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="Qwen2Audio-outputs", max_new_token=256):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.max_new_token = max_new_token
        self.conv_id = None
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")

        self.max_new_token = max_new_token

        if output_speech:
            self.TTSModel = TTSModel()


    def reply(self, messages, round_idx=0):
        new_messages = sharegpt2Qwen2Audio(messages)

        text = self.processor.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False)
        if self.input_speech:
            audios = []
            for message in new_messages:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(librosa.load(
                                ele['audio_url'],
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                                          )
        else:
            audios = None

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.model.device)

        generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_token)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        reply = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if self.output_speech:
            audio = self.TTSModel.synthesize_speech(reply)
            speech = self.save_audio(audio, round_idx)
            return {"text": reply, "speech": speech}
        else:
            return {"text": reply, "speech": None}


    def save_audio(self, audio_tensor, round_idx):
        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
        torchaudio.save(output_path, audio_tensor.cpu(), 24000)

        return output_path


if __name__ == "__main__":
    model = Qwen2_Audio("../weight/Qwen2-Audio-7B-Instruct", input_speech=True, output_speech=False)

    print(model.reply([
        {
            "from": "user",
            "value": "",
            "speech": "./test2.mp3"
        }
    ]))
