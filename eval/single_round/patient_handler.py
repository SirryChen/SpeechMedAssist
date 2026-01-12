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

        if patient_speech:

            # 初始化 fish-speech
            self.fish_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.fish_precision = torch.bfloat16
            self.fish_model, self.fish_decode_one_token = fish_init_model(
                checkpoint_path="../../weight/openaudio-s1-mini",
                device=self.fish_device,
                precision=self.fish_precision,
                compile=True
            )
            with torch.device(self.fish_device):
                self.fish_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.fish_model.config.max_seq_len,
                    dtype=next(self.fish_model.parameters()).dtype
                )
            self.fish_codec_model = load_codec_model(
                config_name="modded_dac_vq",
                checkpoint_path="../../weight/openaudio-s1-mini/codec.pth",
                device=self.fish_device
            )
            self.fish_codec_model.eval()

    def get_num(self):
        return len(self.data)

    def get_question(self):
        text = self.data[self.conv_id]["conversations"][0]["value"]
        if not self.patient_speech:
            return {"text": text, "speech": None}
        else:
            speech = self._synthesize_with_fish_speech(text)
            return {"text": text, "speech": self.save_audio(speech)}

            # speech = self.data[self.conv_id]["conversations"][0]["speech"]
            # return {"text": text, "speech": os.path.join("../", speech)}

    def get_ref_response(self):
        text = self.data[self.conv_id]["conversations"][1]["value"]
        return text

    def _synthesize_with_fish_speech(self, text):
        torch.manual_seed(2025)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(2025)
        torch.cuda.empty_cache()

        generator = generate_long(
            model=self.fish_model,
            device=self.fish_device,
            decode_one_token=self.fish_decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=0,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            compile=False,
            iterative_prompt=True,
            chunk_length=300,
            prompt_text=[],
            prompt_tokens=[],
        )

        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes.cpu())
            elif response.action == "next":
                break
        if not codes:
            raise RuntimeError("未生成任何语义编码！")

        codes_tensor = torch.cat(codes, dim=1).to(self.fish_device).long()
        indices_lens = torch.tensor([codes_tensor.shape[1]], device=self.fish_device, dtype=torch.long)

        with torch.no_grad():
            fake_audios, _ = self.fish_codec_model.decode(codes_tensor, indices_lens)
            waveform = fake_audios[0, 0].detach().cpu()

        speech = torchaudio.functional.resample(
            waveform.unsqueeze(0), orig_freq=self.fish_codec_model.sample_rate, new_freq=16000
        )

        return speech

    def save_audio(self, audio_tensor):

        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"patient.wav")

        torchaudio.save(output_path, audio_tensor.cpu(), 16000)

        return output_path
