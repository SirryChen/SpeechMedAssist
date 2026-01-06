# use env OpenS2S
import os
import sys
import tempfile
import torch
import numpy as np

# 添加 OpenS2S 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
opens2s_root = os.path.join(current_dir, "../../OpenS2S")
opens2s_root = os.path.abspath(opens2s_root)
if opens2s_root not in sys.path:
    sys.path.insert(0, opens2s_root)

from transformers import AutoTokenizer, GenerationConfig
from src.modeling_omnispeech import OmniSpeechModel
from src.feature_extraction_audio import WhisperFeatureExtractor
from src.constants import (
    DEFAULT_AUDIO_START_TOKEN, 
    DEFAULT_AUDIO_END_TOKEN, 
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_TTS_START_TOKEN,
    AUDIO_TOKEN_INDEX
)

try:
    import soundfile as sf
except ImportError:
    raise ImportError("Please install soundfile: pip install soundfile")

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    try:
        import librosa
        HAS_LIBROSA = True
    except ImportError:
        HAS_LIBROSA = False
        print("Warning: Neither scipy nor librosa is available. Audio resampling may not work correctly.")


def load_audio_waveform(audio_path, target_sr=16000, mono=True):
    """
    加载音频文件并转换为目标采样率和单声道
    不依赖 sox，使用 soundfile + scipy/librosa
    """
    # 使用 soundfile 加载音频
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = waveform.T  # 转换为 (channels, length)
    
    # 转换为单声道
    if mono and waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0, keepdims=True)
    
    # 重采样
    if target_sr is not None and target_sr != sample_rate:
        if HAS_SCIPY:
            # 使用 scipy 重采样
            num_samples = int(waveform.shape[1] * target_sr / sample_rate)
            waveform = signal.resample(waveform, num_samples, axis=1)
        elif HAS_LIBROSA:
            # 使用 librosa 重采样
            if waveform.shape[0] == 1:
                waveform_resampled = librosa.resample(
                    waveform[0], orig_sr=sample_rate, target_sr=target_sr
                )
                waveform = waveform_resampled.reshape(1, -1)
            else:
                waveform_resampled = []
                for ch in range(waveform.shape[0]):
                    ch_resampled = librosa.resample(
                        waveform[ch], orig_sr=sample_rate, target_sr=target_sr
                    )
                    waveform_resampled.append(ch_resampled)
                waveform = np.array(waveform_resampled)
        else:
            # 如果没有重采样库，只做简单的警告
            print(f"Warning: Cannot resample from {sample_rate}Hz to {target_sr}Hz. Using original sample rate.")
        sample_rate = target_sr
    
    # WhisperFeatureExtractor 期望 (samples,) 形状
    if mono and waveform.shape[0] == 1:
        waveform = waveform[0]

    return waveform, sample_rate


def sharegpt2OpenS2S(messages, system_prompt=None):
    """将 sharegpt 格式的消息转换为 OpenS2S 格式"""
    new_messages = []
    
    # 添加 system prompt
    if system_prompt:
        new_messages.append({"role": "system", "content": system_prompt})
    
    for item in messages:
        if item.get("from") == "system":
            if "value" in item and item["value"]:
                new_messages.append({"role": "system", "content": item["value"]})
            continue
        
        if item["from"] in ["human", "user"]:
            # 优先使用语音输入
            if item.get("speech") is not None and item["speech"]:
                audio_path = item["speech"]
                if os.path.exists(audio_path):
                    # 使用音频文件路径，稍后处理
                    new_messages.append({
                        "role": "user",
                        "content": {"audio_path": audio_path}
                    })
                else:
                    # 如果音频文件不存在，尝试使用文本
                    if "value" in item and item["value"]:
                        new_messages.append({"role": "user", "content": item["value"]})
            elif "value" in item and item["value"]:
                new_messages.append({"role": "user", "content": item["value"]})
        else:
            # assistant 消息
            if "value" in item and item["value"]:
                new_messages.append({"role": "assistant", "content": item["value"]})
    
    return new_messages


class OpenS2S:
    def __init__(self, model_path, 
                 input_speech=True, 
                 output_speech=False, 
                 speech_output_dir="OpenS2S_output"):
        """
        初始化 OpenS2S 模型
        
        Args:
            model_path: 模型路径
            input_speech: 是否支持语音输入
            output_speech: 是否支持语音输出（暂时不支持）
            speech_output_dir: 语音输出目录（暂时不使用）
            system_prompt: 系统提示词
        """
        if "stage" in model_path:
            system_prompt = "You are SpeechMedAssist, a medical dialogue assistant capable of processing both speech and text questions from patients, and generating speech and text. You can communicate with patients, provide analysis of their condition, ask about more information multiple times if the condition is not clear, and offer final medical consultation advice when information is sufficient."
        else:
            system_prompt = "You are a helpful assistant."

            self.model_path = model_path
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.system_prompt = system_prompt
        self.conv_id = None
        
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)
        
        # 加载模型和 tokenizer
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model = OmniSpeechModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.model.cuda()
        self.model.eval()
        
        # 加载音频特征提取器
        self.audio_extractor = WhisperFeatureExtractor.from_pretrained(
            os.path.join(model_path, "audio")
        )
        
        self.sampling_params = {
            "temperature": 0.2,
            "top_p": 0.8,
            "max_new_tokens": 512,
        }
        
        self.round_idx = 0

    def get_input_params(self, messages):
        """处理消息，转换为模型输入格式"""
        new_messages = []
        audios = []
        
        # 添加 system prompt
        if self.system_prompt:
            new_messages.append({"role": "system", "content": self.system_prompt})
        
        for turn in messages:
            role = turn["role"]
            content = turn["content"]
            
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, dict):
                # 处理音频输入
                if content.get("audio_path"):
                    audio_path = content["audio_path"]
                    if os.path.exists(audio_path):
                        # 使用不依赖 sox 的方法加载音频
                        waveform, _ = load_audio_waveform(
                            audio_path, 
                            target_sr=self.audio_extractor.sampling_rate,
                            mono=True
                        )
                        audios.append(waveform)
                        new_content = f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                    else:
                        new_content = ""
                else:
                    new_content = content.get("text", "")
            else:
                new_content = ""
            
            if new_content:
                new_messages.append({"role": role, "content": new_content})
        
        # 构建 prompt
        prompt = self.tokenizer.apply_chat_template(
            new_messages, 
            add_generation_prompt=True, 
            tokenize=False, 
            enable_thinking=False
        )
        prompt += DEFAULT_TTS_START_TOKEN
        
        # 处理音频 token
        segments = prompt.split(f"{DEFAULT_AUDIO_TOKEN}")
        input_ids = []
        for idx, segment in enumerate(segments):
            if idx != 0:
                input_ids += [AUDIO_TOKEN_INDEX]
            input_ids += self.tokenizer.encode(segment)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        
        # 处理音频特征
        if audios:
            speech_inputs = self.audio_extractor(
                audios,
                sampling_rate=self.audio_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_values = speech_inputs.input_features
            speech_mask = speech_inputs.attention_mask
        else:
            speech_values, speech_mask = None, None
        
        return input_ids, speech_values, speech_mask

    def reply(self, messages, round_idx=0):
        """
        生成回复
        
        Args:
            messages: sharegpt 格式的消息列表
            round_idx: 当前轮次索引
            
        Returns:
            {"text": reply_text, "speech": None}
        """
        # 转换消息格式
        new_messages = sharegpt2OpenS2S(messages, system_prompt=self.system_prompt)
        
        # 获取输入参数
        input_ids, speech_values, speech_mask = self.get_input_params(new_messages)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        if speech_values is not None:
            speech_values = speech_values.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
            speech_mask = speech_mask.to(device='cuda', non_blocking=True)
        
        # 准备生成参数
        generation_config = GenerationConfig.from_dict(self.generation_config.to_dict())
        generation_config.update(
            temperature=self.sampling_params["temperature"],
            top_p=self.sampling_params["top_p"],
            max_new_tokens=self.sampling_params["max_new_tokens"],
            do_sample=True if self.sampling_params["temperature"] > 0.001 else False,
        )
        
        # 生成回复（只生成文本，不生成语音）
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=None,
                speech_values=speech_values,
                speech_mask=speech_mask,
                spk_emb=None,
                units_gen=False,  # 不生成语音单元
                generation_config=generation_config,
                use_cache=True,
            )
        
        # 解码输出
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        return {"text": output_text, "speech": None}


if __name__ == "__main__":
    # 测试代码
    model = OpenS2S(
        model_path="../../OpenS2S/weight/OpenS2S",
        input_speech=True, 
        output_speech=False
    )
    model.conv_id = 0
    
    # 测试语音输入
    print("测试语音输入:")
    print(model.reply([
        {
            "from": "user",
            "value": "",
            "speech": "./test2.mp3"
        }
    ]))
    
    # 测试文本输入
    print("\n测试文本输入:")
    print(model.reply([
        {
            "from": "user",
            "value": "你好，请介绍一下你自己。"
        }
    ]))
