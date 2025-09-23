import os
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .utils import ASRModel, TTSModel, sharegpt_old2new


class Baichuan_M2:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="Baichuan_M2_output"):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
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
                    turn["value"] = self.asr_model.speech2text(turn["speech"])

        new_messages = sharegpt_old2new(messages)
        text = self.tokenizer.apply_chat_template(
            new_messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode='off'  # on/off/auto
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256
        )
        output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ][0]

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        reply = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        if self.input_speech:
            if self.output_speech:
                audio = self.TTSModel.synthesize_speech(reply)
                speech = self.save_audio(audio, round_idx)
                return {"text": reply, "thinking_content": thinking_content, "asr": messages[0]["value"], "speech": speech}
            else:
                return {"text": reply, "thinking_content": thinking_content, "asr": messages[0]["value"]}
        else:
            return {"text": reply, "thinking_content": thinking_content,}

    def save_audio(self, audio_tensor, round_idx):
        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
        torchaudio.save(output_path, audio_tensor.cpu(), 24000)

        return output_path