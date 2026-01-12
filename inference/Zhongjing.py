import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torchaudio
from .utils import ASRModel, TTSModel, sharegpt_old2new
try:
    from jiwer import cer as compute_cer
except ImportError:
    # 如果没有jiwer，使用简单的编辑距离计算
    def compute_cer(hypothesis, reference):
        """计算字符错误率 (CER)"""
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        if not reference:
            return 1.0 if hypothesis else 0.0
        distance = levenshtein_distance(hypothesis, reference)
        return distance / len(reference)


class Zhongjing:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="Zhongjing_output", max_new_token=512):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.max_new_token = max_new_token
        self.conv_id = None
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)

        if input_speech:
            self.asr_model = ASRModel()

        if output_speech:
            self.TTSModel = TTSModel()

    def reply(self, messages, round_idx=0):
        if self.input_speech:
            for i, turn in enumerate(messages):
                if i % 2 == 0:
                    turn["value"] = self.asr_model.speech2text(turn["speech"]) if turn.get("added_value") is None else turn["added_value"].format(cough=self.asr_model.speech2text(turn["speech"]))

        new_messages = sharegpt_old2new(messages)

        prompt = self.tokenizer.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_token)
        output_ids = outputs[0][inputs.input_ids.shape[-1]:]
        reply = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        if self.input_speech:
            if self.output_speech:
                audio = self.TTSModel.synthesize_speech(reply)
                speech = self.save_audio(audio, round_idx)
                return {"text": reply, "asr": messages[0]["value"], "speech": speech}
            else:
                return {"text": reply, "asr": messages[0]["value"], "speech": None}
        else:
            return {"text": reply, "speech": None}

    def save_audio(self, audio_tensor, round_idx):
        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
        torchaudio.save(output_path, audio_tensor.cpu(), 24000)

        return output_path

    def calculate_cer(self, reference_text, asr_text):
        """
        计算字符错误率 (CER)
        :param reference_text: 参考文本
        :param asr_text: ASR识别结果
        :return: CER值 (0-1之间，越小越好)
        """
        if not reference_text:
            return 1.0 if asr_text else 0.0
        if not asr_text:
            return 1.0
        
        return compute_cer(asr_text, reference_text)


