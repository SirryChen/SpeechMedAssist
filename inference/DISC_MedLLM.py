import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torchaudio
from .utils import ASRModel, TTSModel, sharegpt_old2new


class DISC_MedLLM:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="DISC_MedLLM_output"):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None

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
        reply = self.model.chat(self.tokenizer, new_messages)

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

