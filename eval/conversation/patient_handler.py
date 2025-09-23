import torch
import json
import os
import sys
import random
import copy
import torchaudio
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
sys.path.append('../../FishSpeech')  # 添加 Fish-speech 路径
from fish_speech.models.text2semantic.inference import init_model as fish_init_model, generate_long
from fish_speech.models.dac.inference import load_model as load_codec_model
import contextlib


current_dir = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def add_sys_paths(paths):
    """临时加入多个 sys.path"""
    old_path = list(sys.path)
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path[:] = old_path


class PatientModel:
    def __init__(self, model_path, patient_speech=False, ref_wav_path=None, speech_output_dir="patient_outputs", desc_prompt=None, reply_prompt=None):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.model = Qwen2ForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
        self.output_speech = patient_speech
        self.speech_output_dir = speech_output_dir
        os.makedirs(speech_output_dir, exist_ok=True)

        self.desc_prompt = desc_prompt
        self.reply_prompt = reply_prompt

        self.conv_id = None
        self.round_idx = None
        self.patient_gender = None
        self.patient_age = None
        self.prompt_speech = None
        self.prompt_text = None

        if patient_speech:
            for key in list(sys.modules.keys()):
                if "cosyvoice" in key:
                    del sys.modules[key]

            with add_sys_paths([os.path.join(current_dir, "../../CosyVoice"),
                                os.path.join(current_dir, "../../CosyVoice/third_party/Matcha-TTS")]):
                from cosyvoice.utils.file_utils import load_wav
                from cosyvoice.cli.cosyvoice import CosyVoice2

            with open(ref_wav_path, encoding="utf-8") as f:
                self.ref_wav = json.load(f)
            self.wav_base_path = "../../"
            # 初始化 cosyvoice
            self.cosyvoice = CosyVoice2(os.path.join(current_dir, "../../weight/CosyVoice2-0.5B"), load_jit=False, load_trt=False)

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


    def desc(self, base_info):
        self.round_idx = 0
        self.prompt_speech = None
        self.prompt_text = None
        prompt = self.desc_prompt.format(base_info=base_info)
        return self.generate(prompt)

    def reply(self, base_info, history_conv_text, round_idx):
        self.round_idx = round_idx
        prompt = self.reply_prompt.format(base_info=base_info, history_conv_text=history_conv_text)
        return self.generate(prompt)

    def generate(self, prompt):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        output_ids = outputs[0][inputs.input_ids.shape[-1]:]
        reply = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        speech = self.text_to_speech(reply) if self.output_speech else None
        return {"text": reply, "speech": speech}

    def text_to_speech(self, text):
        gender = self.patient_gender
        age_group = self.patient_age

        print(gender)
        print(age_group)
        if (
                gender not in self.ref_wav or
                age_group not in self.ref_wav[gender] or
                not self.ref_wav[gender][age_group] or
                random.random() < 1
        ):
            # 使用 Fish-speech 合成
            print("使用 Fish-speech 合成")
            speech = self._synthesize_with_fish_speech(text, self.prompt_text, self.prompt_speech)
            return self.save_audio(speech, original_sr=16000)

        # 使用 CosyVoice 合成
        print("使用 CosyVoice 合成")
        if self.prompt_speech is None:
            while True:
                speaker_id = random.choice(list(self.ref_wav[gender][age_group].keys()))
                prompts = self.ref_wav[gender][age_group][speaker_id]
                prompt_wav_path, self.prompt_text = random.choice(prompts)
                full_prompt_wav_path = os.path.join(self.wav_base_path, prompt_wav_path)
                if not os.path.exists(full_prompt_wav_path):
                    continue
                self.prompt_speech = load_wav(full_prompt_wav_path, 16000)
                duration = self.prompt_speech.shape[1] / 16000
                if duration < 30:
                    break
            print("prompt_wav:", prompt_wav_path)

        speech = next(self.cosyvoice.inference_zero_shot(
            text,
            prompt_text=self.prompt_text,
            prompt_speech_16k=self.prompt_speech,
            stream=False
        ))['tts_speech']
        return self.save_audio(speech, original_sr=self.cosyvoice.sample_rate)

    def _synthesize_with_fish_speech(self, text, prompt_text=None, prompt_tokens=None):
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
            # prompt_text=[] if prompt_text is None else prompt_text,
            # prompt_tokens=[] if prompt_tokens is None else prompt_tokens,
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

        # if self.prompt_speech is None:
        #     self.prompt_speech = copy.deepcopy(codes_tensor.cpu())
        #     self.prompt_text = text

        with torch.no_grad():
            fake_audios, _ = self.fish_codec_model.decode(codes_tensor, indices_lens)
            waveform = fake_audios[0, 0].detach().cpu()

        speech = torchaudio.functional.resample(
            waveform.unsqueeze(0), orig_freq=self.fish_codec_model.sample_rate, new_freq=16000
        )
        return speech

    def save_audio(self, audio_tensor, original_sr, target_sr=16000):
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
            audio_tensor = resampler(audio_tensor)

        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"patient_{self.round_idx}.wav")

        torchaudio.save(output_path, audio_tensor.cpu(), target_sr)

        return output_path
