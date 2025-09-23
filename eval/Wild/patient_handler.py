import torch
import json
import os
import sys
import torchaudio
sys.path.append('../../FishSpeech')  # 添加 Fish-speech 路径
from fish_speech.models.text2semantic.inference import init_model as fish_init_model, generate_long
from fish_speech.models.dac.inference import load_model as load_codec_model

current_dir = os.path.dirname(os.path.abspath(__file__))


class PatientModel:
    def __init__(self, data_path, patient_speech=False, speech_output_dir="patient_outputs"):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.patient_speech = patient_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = 0

    def get_num(self):
        return len(self.data)

    def get_question(self):
        text = self.data[self.conv_id]["conversations"][0]["value"]
        if not self.patient_speech:
            return {"text": text, "speech": None}
        else:
            speech = self.data[self.conv_id]["conversations"][0]["speech"]
            speech = speech.replace("m4a", "wav")
            return {"text": text, "speech": os.path.join(current_dir, "../../dataset/SMA_publicTest/", speech)}

    def get_ref_response(self):
        text = self.data[self.conv_id]["conversations"][1]["value"]
        return text

