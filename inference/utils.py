import os
import whisper
import torchaudio
import torch
import numpy as np
import time
import torch.nn.functional as F
import threading
import uuid
from collections import deque
from hyperpyyaml import load_hyperpyyaml
import sys
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


class RealTimeVADRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=0.2, vad_threshold=0.001, silence_duration=1.5, max_record_duration=20.0, pre_buffer_seconds=0.5):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.max_record_duration = max_record_duration
        self.pre_buffer_size = int(pre_buffer_seconds / chunk_duration)

        self.frames = []
        self.pre_speech_buffer = deque(maxlen=self.pre_buffer_size)
        self.recording = False
        self.vad_started = False
        self.last_audio_time = None


    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        is_speech = volume_norm > self.vad_threshold

        now = time.time()
        tensor_data = torch.from_numpy(indata.copy()).float()

        # 始终保留预缓冲帧
        self.pre_speech_buffer.append(tensor_data)

        if is_speech:
            self.last_audio_time = now
            if not self.vad_started:
                print("[🎤] 检测到语音开始")
                self.vad_started = True
                self.frames.extend(list(self.pre_speech_buffer))  # 补上说话前的音频
            self.frames.append(tensor_data)
        elif self.vad_started and (now - self.last_audio_time > self.silence_duration):
            print("[🛑] 语音结束")
            self.recording = False

    def record(self) -> torch.Tensor:
        import sounddevice as sd
        print("[⏺️] 开始监听麦克风...说话后自动录音")
        self.recording = True
        self.frames = []
        self.pre_speech_buffer.clear()
        self.vad_started = False
        self.last_audio_time = time.time()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
                            callback=self._callback, blocksize=self.chunk_size):
            while self.recording:
                time.sleep(0.05)
                if self.vad_started:
                    duration = len(self.frames) * self.chunk_size / self.sample_rate
                    if duration > self.max_record_duration:
                        print("[⚠️] 达到最大录音时间，强制停止")
                        self.recording = False

        print("[✅] 录音完成")
        audio_tensor = torch.cat(self.frames, dim=0).squeeze()  # shape: (samples,)
        return audio_tensor

    def record_audio(self, save=False, save_path="recorded.wav") -> torch.Tensor:
        """
        录音并返回 Whisper 所需的 log-mel 频谱 (T, n_mels)
        """
        audio = self.record()  # torch.FloatTensor, [-1.0, 1.0]

        if save:
            torchaudio.save(save_path, audio.unsqueeze(0), sample_rate=self.sample_rate)
            print(f"[💾] 录音已保存: {save_path}")

        # Whisper 需要固定长度的 pad_or_trim
        audio_padded = whisper.pad_or_trim(audio, length=whisper.audio.N_SAMPLES)
        mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128).permute(1, 0)  # (T, n_mels)

        return mel



def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)

def process_units(input_str):
    import re
    numbers = re.findall(r'\d+', input_str)
    units = list(map(int, numbers))
    return units

class SpeechDecoder:
    def __init__(self,
                 model_dir,
                 device="cuda",
                 hop_len=None,
                 load_jit=False,
                 load_trt=False,
                 load_onnx=False,
                 prompt_speech_path=None,
                 ):
        with add_sys_paths([os.path.join(current_dir, "../third_party/CosyVoice"), os.path.join(current_dir, "../third_party")]):
            from cosyvoice.cli.frontend import CosyVoiceFrontEnd
            from cosyvoice.utils.file_utils import load_wav
            from hyperpyyaml import load_hyperpyyaml

        self.device = device

        # Config
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        # Frontend
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            '{}/campplus.onnx'.format(model_dir),
            '{}/speech_tokenizer_v2.onnx'.format(model_dir),
            '{}/spk2info.pt'.format(model_dir),
            False,
            configs['allowed_special']
        )
        self.sample_rate = configs['sample_rate']

        # Load models
        self.flow = configs['flow']
        self.flow.load_state_dict(torch.load('{}/flow.pt'.format(model_dir), map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        self.flow.decoder.fp16 = False
        self.hift = configs['hift']
        hift_state_dict = {k.replace('generator.', ''): v for k, v in
                           torch.load('{}/hift.pt'.format(model_dir), map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        if load_jit:
            self.load_jit('{}/flow.encoder.fp32.zip'.format(model_dir))
        if load_trt is True and load_onnx is True:
            load_onnx = False
            # logging.warning('can not set both load_trt and load_onnx to True, force set load_onnx to False')
        if load_onnx:
            self.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.format(model_dir))
        if load_trt:
            self.load_trt('{}/flow.decoder.estimator.fp16.Volta.plan'.format(model_dir))

        self.token_hop_len = hop_len if hop_len is not None else 2 * self.flow.input_frame_rate
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # dict used to store session related variable
        self.lock = threading.Lock()
        self.hift_cache_dict = {}
        self.prompt_speech_16k = load_wav(prompt_speech_path if prompt_speech_path else os.path.join(current_dir, "prompt_zh.wav"), 16000)

    def load_jit(self, flow_encoder_model):
        print("Loading JIT model")
        self.flow.encoder = torch.jit.load(flow_encoder_model, map_location=self.device)

    def load_onnx(self, flow_decoder_estimator_model):
        print("Loading ONNX model")
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option,
                                                                   providers=providers)

    def load_trt(self, flow_decoder_estimator_model):
        print("Loading TRT model")
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(
                f.read())
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()
        self.flow.decoder.fp16 = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                             self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                             self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def entry(self, generated_tokens, prompt_speech_16k, stream=False, speed=1.0):
        prompt_speech_feat = torch.zeros(1, 0, 80)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        embedding = self.frontend._extract_spk_embedding(prompt_speech_16k)
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.hift_cache_dict[this_uuid] = None
        if stream:
            token_offset = 0
            this_tts_speech_output = []
            while True:
                if len(generated_tokens) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = generated_tokens[
                                            :token_offset + self.token_hop_len + self.flow.pre_lookahead_len].unsqueeze(
                        0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=embedding,
                        uuid=this_uuid,
                        token_offset=token_offset,
                        finalize=False
                    )
                    token_offset += self.token_hop_len
                    this_tts_speech_output.append(this_tts_speech.cpu())
                else:
                    break
            this_tts_speech_token = generated_tokens.unsqueeze(0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=embedding,
                uuid=this_uuid,
                token_offset=token_offset,
                finalize=True
            )
            this_tts_speech_output.append(this_tts_speech.cpu())
            return torch.cat(this_tts_speech_output, dim=1)
        else:
            this_tts_speech_token = generated_tokens.unsqueeze(0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=embedding,
                uuid=this_uuid,
                token_offset=0,
                finalize=True,
                speed=speed
            )
            return this_tts_speech.cpu()

    def generate(self, units, stream=False):
        units = torch.LongTensor(process_units(units)).cuda()
        tts_speech = self.entry(
            units,
            self.prompt_speech_16k,
            stream=stream,
        )
        return tts_speech

    def init_prompt(self):
        prompt_speech_feat = torch.zeros(1, 0, 80)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        embedding = self.frontend._extract_spk_embedding(self.prompt_speech_16k)
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.hift_cache_dict[this_uuid] = None

        session = {
            'uuid': this_uuid,
            'prompt_feat': prompt_speech_feat,
            'prompt_token': prompt_speech_token,
            'embedding': embedding,
            'token_offset': 0,
            'generated_tokens': None
        }
        return session

    def process_unit_chunk(self, new_chunk, session, finalize=False):
        if session["generated_tokens"] is None:
            session["generated_tokens"] = new_chunk
        else:
            session["generated_tokens"] = torch.cat([session["generated_tokens"], new_chunk], dim=-1)

        token_offset = session['token_offset']
        this_uuid = session['uuid']
        prompt_feat = session['prompt_feat']
        prompt_token = session['prompt_token']
        embedding = session['embedding']
        generated_tokens = session["generated_tokens"]

        tts_speech = self.token2wav(
            token=generated_tokens.unsqueeze(0),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            uuid=this_uuid,
            token_offset=token_offset,
            finalize=finalize,
        )
        if not finalize:
            session["token_offset"] = len(generated_tokens) - self.flow.pre_lookahead_len
        else:
            session["token_offset"] = len(generated_tokens)
        return tts_speech.cpu(), session


from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class ASRModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.abspath(os.path.join(current_dir, "../weight"))
        self.asr_model = AutoModel(
            model=os.path.join(model_path, "SenseVoiceSmall"),
            vad_model=os.path.join(model_path, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            disable_update=True,
            disable_pbar=True
        )

    def speech2text(self, speech_path):
        res = self.asr_model.generate(
            input=speech_path,
            cache={},
            language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])

        return text


import sys
import os
import torch
import torchaudio

class TTSModel:
    def __init__(self, cosyvoice_path=os.path.join(current_dir, "../weight/CosyVoice2-0.5B")):
        # 永久添加路径，因为 CosyVoice 内部模块初始化时也需要导入 matcha
        cosy_path = os.path.join(current_dir, "../CosyVoice")
        matcha_path = os.path.join(current_dir, "../CosyVoice/third_party/Matcha-TTS")
        
        if cosy_path not in sys.path:
            sys.path.insert(0, cosy_path)
        if matcha_path not in sys.path:
            sys.path.insert(0, matcha_path)
        
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        self.cosyvoice = CosyVoice2(cosyvoice_path, load_jit=False, load_trt=False)
        self.prompt_text = "智能手机最重要的功能是通信和互联网"
        self.prompt_speech = load_wav(os.path.join(current_dir, "./prompt_zh.wav"), 16000)

    def synthesize_speech(self, text, output_path=None):

        speech = self.cosyvoice.inference_zero_shot(
            text,
            prompt_text=self.prompt_text,
            prompt_speech_16k=self.prompt_speech,
            stream=False
        )

        speech = next(speech)['tts_speech']

        return speech


def sharegpt_old2new(messages):
    new_messages = []
    for message in messages:
        new_messages.append({"role": message["from"], "content": message["value"]})
    return new_messages


if __name__ == "__main__":
    asr = ASRModel()
    print(asr.speech2text("./prompt_zh.wav"))
