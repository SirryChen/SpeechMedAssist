import os
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import TextIteratorStreamer
from threading import Thread
import time

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../")
sys.path.append(project_path)

from model import SpeechQwen2ForCausalLM, Speech2SQwen2ForCausalLM
from model.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from inference.utils import SpeechDecoder
from torch.nn.utils.rnn import pad_sequence
import whisper
import torchaudio
import tempfile
from typing import Dict, Any
import numpy as np


def sharegpt_old2new(messages):
    new_messages = []
    for message in messages:
        new_messages.append({"role": message["from"], "content": message["value"]})
    return new_messages

def build_unit_tokenizer(vocab_size):
    import os
    from transformers import BertTokenizer
    with open("unit_vocab.txt", "w") as f:
        for i in range(vocab_size + 1):
            f.write(str(i) + "\n")
    tokenizer = BertTokenizer(vocab_file="unit_vocab.txt")
    os.remove("unit_vocab.txt")
    return tokenizer


class SpeechMedAssist:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="SMA_output", max_new_token=256, input_max_length=4096):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.max_new_token = max_new_token
        self.conv_id = None
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)

        model_cls = Speech2SQwen2ForCausalLM if output_speech else SpeechQwen2ForCausalLM
        config = AutoConfig.from_pretrained(model_path)
        config.speech_encoder = os.path.join(current_dir, "../weight/whisper/large-v3.pt")
        config.tts_tokenizer = os.path.join(current_dir, config.tts_tokenizer)
        config.tokenizer_model_max_length = input_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = model_cls.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16).cuda()

        if output_speech:
            self.decoder = SpeechDecoder(
                model_dir=os.path.join(project_path, "weight/cosy2_decoder")
            )
            self.model.eval()
            self.model = torch.compile(self.model)
            self.unit_tokenizer = build_unit_tokenizer(self.model.config.unit_vocab_size)

    def load_speech(self, path):
        speech = whisper.load_audio(path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        return speech


    def get_text_embedding(self, text):
        inputs = self.tokenizer(text, add_generation_prompt=True, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        input_embedding = self.model.get_model().embed_tokens(input_ids.to(self.model.device))

        return input_embedding

    def get_speech_embedding(self, speech):
        speech_list = [self.load_speech(speech)]
        speech_tensors = pad_sequence(
            speech_list,
            batch_first=True,
            padding_value=0  # Whisper 训练时通常也对无声部分 pad 为 0
        ).to(self.model.device).bfloat16()
        speech_lengths = torch.LongTensor([len(speech) for speech in speech_list]).to(self.model.device)

        speech_feature = self.model.encode_speech(speech_tensors, speech_lengths)[0]

        return speech_feature


    def reply(self, messages, round_idx=0):
        if self.input_speech:
            speech_list = []
            for i, turn in enumerate(messages):
                if i % 2 == 0:
                    turn["value"] = DEFAULT_SPEECH_TOKEN if turn.get("added_value") is None else turn["added_value"].format(cough=DEFAULT_SPEECH_TOKEN)
                    speech_list.append(self.load_speech(turn["speech"]))
            speech_tensors = pad_sequence(
                speech_list,
                batch_first=True,
                padding_value=0  # Whisper 训练时通常也对无声部分 pad 为 0
            ).to(self.model.device).bfloat16()
            speech_lengths = torch.LongTensor([len(speech) for speech in speech_list]).to(self.model.device)
        else:
            speech_tensors = None
            speech_lengths = None

        input_ids = self.tokenizer.apply_chat_template(sharegpt_old2new(messages), add_generation_prompt=True, return_tensors="pt")[0]
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        input_ids = input_ids.unsqueeze(0).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                speech=speech_tensors if self.input_speech else None,
                speech_lengths=speech_lengths if self.input_speech else None,
                do_sample=False,
                max_new_tokens=self.max_new_token,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.output_speech:
                output_ids, output_units = outputs
                audio = self.decoder.generate(output_units, stream=False)
            else:
                output_ids = outputs
                audio = None

        reply = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if self.output_speech and audio is not None:
            speech = self.save_audio(audio, round_idx)
        else:
            speech = None

        return {"text": reply, "speech": speech}

    def stream_reply(self, messages):
        if self.input_speech:
            speech_list = []
            for i, turn in enumerate(messages):
                if i % 2 == 0:  # user语音回合
                    turn["value"] = DEFAULT_SPEECH_TOKEN
                    speech_list.append(self.load_speech(turn["speech"]))
            speech_tensors = pad_sequence(
                speech_list,
                batch_first=True,
                padding_value=0
            ).to(self.model.device).bfloat16()
            speech_lengths = torch.LongTensor([len(speech) for speech in speech_list]).to(self.model.device)
        else:
            speech_tensors = None
            speech_lengths = None

        start_time = time.time()
        # 构造输入
        input_ids = self.tokenizer.apply_chat_template(
            sharegpt_old2new(messages),
            add_generation_prompt=True,
            return_tensors="pt"
        )[0]
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        input_ids = input_ids.unsqueeze(0).to(self.model.device)

        # streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False, timeout=15)
        # 单位 streamer
        from transformers import TextIteratorStreamer as UnitStreamer
        streamer_unit = UnitStreamer(self.unit_tokenizer, skip_prompt=False, skip_special_tokens=False, timeout=15) \
            if self.output_speech else None

        # 后台生成线程
        thread = Thread(target=self.model.generate, kwargs=dict(
            inputs=input_ids,
            speech=speech_tensors,
            speech_lengths=speech_lengths,
            do_sample=False,
            max_new_tokens=self.max_new_token,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
            streamer_unit=streamer_unit if self.output_speech else None,
            use_cache=True,
        ))
        thread.start()

        generated_text = ""
        stop_str = "<|im_end|>"
        tts_speechs = []
        num_generated_units = 0
        session = None

        if self.output_speech:
            # 初始化 vocoder 状态
            session = self.decoder.init_prompt()

        for new_text in streamer:
            generated_text += new_text
            finalize = generated_text.endswith(stop_str)

            audio_out = None
            if self.output_speech and streamer_unit is not None:
                output_unit = list(map(int, streamer_unit.token_cache))
                if len(output_unit) > num_generated_units:
                    new_units = output_unit[num_generated_units:]
                    num_generated_units = len(output_unit)
                    new_units_tensor = torch.LongTensor(new_units).to(self.model.device)
                    tts_chunk, session = self.decoder.process_unit_chunk(
                        new_units_tensor, session, finalize=finalize
                    )
                    if tts_chunk is not None:
                        tts_speechs.append(tts_chunk)
                        audio_out = torch.cat(tts_speechs, dim=-1).cpu()

            if finalize:
                generated_text = generated_text[:-len(stop_str)]
                if self.output_speech and streamer_unit is not None:
                    streamer_unit.end()

            yield {
                "text": generated_text,
                "speech": audio_out,  # 直接返回拼接的语音tensor
                "sample_rate": 24000,
                "finalize": finalize,
                "error_code": 0,
                "start_time": start_time
            }

            if finalize:
                break

    def save_audio(self, audio_tensor, round_idx):
        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
        torchaudio.save(output_path, audio_tensor.cpu(), 24000)

        return output_path



    def generate_from_array(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """
        直接从内存中的语音数组进行推理，适配 VoiceBench 的音频输入格式。
        Args:
            audio_array: 1D numpy array (float32/float64), 单通道语音
            sample_rate: 采样率
        Returns:
            文本回复
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            tmp_path = tmpf.name
        try:
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            if audio_array.ndim == 1:
                tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
            elif audio_array.ndim == 2:
                # 取第一通道或做均值
                if audio_array.shape[0] == 1:
                    tensor = torch.from_numpy(audio_array).float()
                else:
                    tensor = torch.from_numpy(audio_array.mean(axis=0)).float().unsqueeze(0)
            else:
                raise ValueError("audio_array must be 1D or 2D numpy array")
            torchaudio.save(tmp_path, tensor, sample_rate)
            messages = [{"from": "user", "value": "", "speech": tmp_path}]
            result = self.reply(messages, round_idx=0)
            return result.get("text", "")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


    def generate_from_voicebench_item(self, audio_item: Dict[str, Any]) -> str:
        """
        适配 VoiceBench 数据集中 datasets.Audio 列的样本：
        通常包含 keys: ['array', 'sampling_rate', ...]
        """
        array = audio_item.get("array")
        sr = audio_item.get("sampling_rate")
        if array is None or sr is None:
            # 兼容可能的变体字段
            array = audio_item.get("audio", {}).get("array", array)
            sr = audio_item.get("audio", {}).get("sampling_rate", sr)
        if array is None or sr is None:
            raise ValueError("VoiceBench audio item missing 'array' or 'sampling_rate'")
        return self.generate_from_array(array, int(sr))


if __name__ == "__main__":
    model = SpeechMedAssist(model_path="../weight/stage2-cough2", input_speech=True, output_speech=False)
    print(model.reply([
        {
            "from": "user",
            "value": "",
            "speech": "./20250920_115710.m4a"
        }
    ]))

    for item in model.stream_reply([
        {
            "from": "user",
            "value": "我最近左侧肋骨下一直有隐痛，是怎么回事呢？",
            "speech": "./test2.mp3"
        }
    ]):
        print(item["text"])




