# use env SpeechMedAssist
import os
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../GLM-4-Voice")
sys.path.append(project_path)
sys.path.append(os.path.join(current_dir, r"../../GLM-4-Voice/third_party/Matcha-TTS"))

import torch
import torchaudio
import uuid
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
from audio_process import AudioStreamProcessor

from transformers.generation.streamers import BaseStreamer
from threading import Thread
from queue import Queue


class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class GLM4_Voice:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="GLM4_Voice_output", max_new_token=512, dtype="bfloat16", block_size_list=[25, 50, 100, 150, 200]):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.max_new_token = max_new_token
        self.block_size_list = block_size_list
        self.conv_id = None
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)

        self.model_path = os.path.join(model_path, "glm-4-voice-9b")
        self.flow_path = os.path.join(model_path, "glm-4-voice-decoder")
        self.tokenizer_path = os.path.join(model_path, "glm-4-voice-tokenizer")

        # 量化配置
        if dtype == "int4":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            self.bnb_config = None

        # 加载tokenizer和本地GLM4模型（AutoModel，trust_remote_code，device_map）
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            quantization_config=self.bnb_config if self.bnb_config else None,
            device_map="auto"
        ).eval()

        self.device = self.model.device

        # 音频解码器
        if output_speech:
            flow_config = os.path.join(self.flow_path, "config.yaml")
            flow_checkpoint = os.path.join(self.flow_path, 'flow.pt')
            hift_checkpoint = os.path.join(self.flow_path, 'hift.pt')
            self.audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint, hift_ckpt_path=hift_checkpoint, device=self.device)
            self.audio_processor = AudioStreamProcessor()
            self.model = torch.compile(self.model)
        else:
            self.audio_decoder = None
            self.audio_processor = None
        # Whisper特征提取器和VQEncoder（音频token提取）
        if self.tokenizer_path:
            self.whisper_model = WhisperVQEncoder.from_pretrained(self.tokenizer_path).eval().to(self.device)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.tokenizer_path)
        else:
            self.whisper_model = None
            self.feature_extractor = None

    def extract_audio_tokens(self, audio_path):
        if not self.whisper_model or not self.feature_extractor:
            raise RuntimeError("Whisper模型或特征提取器未初始化")
        audio_tokens = extract_speech_token(self.whisper_model, self.feature_extractor, [audio_path])[0]
        if len(audio_tokens) == 0:
            raise ValueError("未提取到音频token")
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        return f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"

    def save_audio(self, audio_tensor, round_idx):
        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
        torchaudio.save(output_path, audio_tensor.cpu(), 22050, format="wav")
        return output_path

    def reply(self, messages, round_idx=0, temperature=1.0, top_p=1.0):
        """
        本地推理，输入为messages，输出为dict（含text和可选speech）。
        :param messages: [{from, value, speech?}]
        :return: {"text": ..., "speech": ...}
        """
        # 构造输入历史
        history = []
        for msg in messages:
            role = msg.get("from")
            value = msg.get("value")
            speech_path = msg.get("speech")
            if self.input_speech and speech_path and role == "human":
                user_input = self.extract_audio_tokens(speech_path)
                history.append({"role": "user", "content": {"path": speech_path}})
            elif role == "human":
                user_input = value
                history.append({"role": "user", "content": value})
            elif role == "assistant":
                history.append({"role": "assistant", "content": value})
        # 当前输入
        if self.input_speech and messages[-1].get("speech"):
            user_input = self.extract_audio_tokens(messages[-1]["speech"])
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
        else:
            user_input = messages[-1]["value"]
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        # 拼接历史token串
        inputs = ""
        for h in history[:-1]:
            if h["role"] == "user":
                if isinstance(h["content"], dict) and "path" in h["content"]:
                    inputs += f"<|user|>\n<|audio|>{h['content']['path']}<|assistant|>streaming_transcription\n"
                else:
                    inputs += f"<|user|>\n{h['content']}<|assistant|>streaming_transcription\n"
            elif h["role"] == "assistant":
                if isinstance(h["content"], dict) and "path" in h["content"]:
                    inputs += f"<|assistant|>\n<|audio|>{h['content']['path']}\n"
                else:
                    inputs += f"<|assistant|>\n{h['content']}\n"
        if "<|system|>" not in inputs:
            inputs = f"<|system|>\n{system_prompt}\n" + inputs
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        # 编码输入
        input_ids = self.tokenizer([inputs], return_tensors="pt").input_ids.to(self.device)
        # 本地推理
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=self.max_new_token,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        output_ids = outputs[0]
        # 分离text_tokens和audio_tokens
        text_tokens, audio_tokens = [], []
        audio_offset = self.tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for token_id in output_ids[input_ids.shape[1]:]:
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            else:
                text_tokens.append(token_id)
        # 用text_tokens解码纯文本
        reply_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        # 音频合成
        if self.output_speech and self.audio_decoder and audio_tokens:
            tts_token = torch.tensor(audio_tokens, device=self.audio_decoder.device).unsqueeze(0)
            this_uuid = str(uuid.uuid1())
            audio, _ = self.audio_decoder.token2wav(tts_token, this_uuid)
            speech_path = self.save_audio(audio, round_idx)
        else:
            speech_path = None
        return {"text": reply_text, "speech": speech_path}

    def stream_reply(self, messages, temperature=0.2, top_p=0.8, max_new_tokens=256):
        """
        流式推理，输入为messages（user含speech，assistant仅文本），流式返回文本和语音。
        yield {"text": ..., "audio": ...}  # text为增量文本，audio为增量音频（如有）
        """

        # 构造输入历史
        history = []
        for msg in messages:
            role = msg.get("from")
            value = msg.get("value")
            speech_path = msg.get("speech")
            if self.input_speech and speech_path and role == "human":
                user_input = self.extract_audio_tokens(speech_path)
                history.append({"role": "user", "content": {"path": speech_path}})
            elif role == "human":
                user_input = value
                history.append({"role": "user", "content": value})
            elif role == "assistant":
                history.append({"role": "assistant", "content": value})
        # 当前输入
        if self.input_speech and messages[-1].get("speech"):
            user_input = self.extract_audio_tokens(messages[-1]["speech"])
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
        else:
            user_input = messages[-1]["value"]
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        # 拼接历史token串

        inputs = ""
        for h in history[:-1]:
            if h["role"] == "user":
                if isinstance(h["content"], dict) and "path" in h["content"]:
                    inputs += f"<|user|>\n<|audio|>{h['content']['path']}<|assistant|>streaming_transcription\n"
                else:
                    inputs += f"<|user|>\n{h['content']}<|assistant|>streaming_transcription\n"
            elif h["role"] == "assistant":
                if isinstance(h["content"], dict) and "path" in h["content"]:
                    inputs += f"<|assistant|>\n<|audio|>{h['content']['path']}\n"
                else:
                    inputs += f"<|assistant|>\n{h['content']}\n"
        if "<|system|>" not in inputs:
            inputs = f"<|system|>\n{system_prompt}\n" + inputs
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        start_time = time.time()

        # 编码输入
        inputs = self.tokenizer([inputs], return_tensors="pt").to(self.device)
        # 流式推理
        streamer = TokenStreamer(skip_prompt=True)
        thread = Thread(
            target=self.model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                streamer=streamer
            )
        )
        thread.start()
        text_tokens, audio_tokens = [], []
        audio_offset = self.tokenizer.convert_tokens_to_ids('<|audio_0|>')
        complete_tokens = []
        tts_speechs = []
        tts_mels = []
        prev_mel = None
        is_finalize = False
        block_size_idx = 0
        block_size = self.block_size_list[block_size_idx]
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        this_uuid = str(uuid.uuid4())
        # 流式合成
        for token_id in streamer:
            if token_id == self.tokenizer.convert_tokens_to_ids('<|user|>'):
                is_finalize = True

            # 只有未结束时才累积文本 token
            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)

            # 触发合成
            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                if block_size_idx < len(self.block_size_list) - 1:
                    block_size_idx += 1
                    block_size = self.block_size_list[block_size_idx]

                tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)

                if prev_mel is not None:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                if self.audio_decoder:
                    tts_speech, tts_mel = self.audio_decoder.token2wav(
                        tts_token,
                        uuid=this_uuid,
                        prompt_token=flow_prompt_speech_token.to(self.device),
                        prompt_feat=prompt_speech_feat.to(self.device),
                        finalize=is_finalize
                    )
                    prev_mel = tts_mel

                    audio_bytes = self.audio_processor.process(
                        tts_speech.clone().cpu().numpy()[0],
                        last=is_finalize
                    )
                else:
                    tts_speech, tts_mel, audio_bytes = None, None, None

                if tts_speech is not None:
                    tts_speechs.append(tts_speech.squeeze())
                if tts_mel is not None:
                    tts_mels.append(tts_mel)

                if audio_bytes:
                    print("流式输出：不返回完整文本，只返回音频")
                    yield {"text": "", "audio": audio_bytes, "start_time": start_time}

                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_tokens = []

        # ===== 最终输出 =====
        complete_text = self.tokenizer.decode(
            complete_tokens, spaces_between_special_tokens=False
        )

        audio_np = None
        if self.audio_decoder and tts_speechs:
            tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
            audio_np = tts_speech.numpy()

        yield {
            "text": complete_text,   # 完整文本
            "speech": audio_np,       # numpy 音频数据
            "sample_rate": 22050,
            "start_time": start_time
        }


if __name__ == "__main__":
    from pydub import AudioSegment

    audio = AudioSegment.from_file("./20250920_115710.m4a", format="m4a")

    # 导出为 wav
    audio.export("./20250920_115710.wav", format="wav")

    model = GLM4_Voice(model_path="../weight/GLM4-Voice", input_speech=True, output_speech=True)

    print(model.reply([
        {
            "from": "user",
            "value": "",
            "speech": "./20250920_115710.wav"
        }
    ]))
    # for item in model.stream_reply([
    #     {
    #         "from": "user",
    #         "value": "",
    #         "speech": "./test2.mp3"
    #     }
    # ]):
    #     print(item["text"])
