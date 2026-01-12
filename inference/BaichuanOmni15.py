# use env BaichuanOmni15
import os
import sys
import re
import json

try:
    import ujson
except ImportError:
    ujson = json

import numpy as np
import torch
import torchaudio

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../Baichuan-Omni-1.5")
# 将项目路径添加到 sys.path 的最前面，确保优先导入
sys.path.insert(0, project_path)
sys.path.append(os.path.join(project_path, "./third_party/cosy24k_vocoder"))

from transformers import AutoTokenizer, AutoModelForCausalLM

# 在导入 generation 模块之前，先修补 transformers 兼容性问题
# 某些版本的 transformers 可能没有 is_torchdynamo_compiling 或 is_quanto_available 函数
try:
    import transformers.generation.utils as gen_utils
    # 修补 is_torchdynamo_compiling
    if not hasattr(gen_utils, 'is_torchdynamo_compiling'):
        def is_torchdynamo_compiling():
            """兼容性函数：检查是否在 torchdynamo 编译模式下"""
            try:
                import torch._dynamo
                return torch._dynamo.is_compiling()
            except (ImportError, AttributeError):
                return False
        gen_utils.is_torchdynamo_compiling = is_torchdynamo_compiling
    
    # 修补 is_quanto_available（某些版本可能没有）
    if not hasattr(gen_utils, 'is_quanto_available'):
        def is_quanto_available():
            """兼容性函数：检查 quanto 是否可用"""
            try:
                import quanto
                return True
            except ImportError:
                return False
        gen_utils.is_quanto_available = is_quanto_available
except (ImportError, AttributeError):
    pass  # 如果无法导入 transformers.generation.utils，继续

# 延迟导入 generation 模块，只在需要时导入
# 这些变量将在类初始化时设置
decode_wave_vocoder = None
GenerationAudioTokens = None

from cosy24k_vocoder import Cosy24kVocoder

# 从官方代码中需要的常量，如果没有定义则使用默认值
try:
    from constants import (
        sampling_rate,
        COSY_VOCODER,
        g_cache_dir,
        role_prefix,
        wave_concat_overlap,
        MODEL_PATH,
    )
except ImportError:
    # 如果 constants 不存在，使用默认值
    sampling_rate = 24000
    COSY_VOCODER = os.path.join(project_path, "./third_party/cosy24k_vocoder")
    g_cache_dir = "./BaichuanOmni15_cache"
    role_prefix = {
        "user": "<|user|>\n",
        "assistant": "<|assistant|>\n",
        "system": "<|system|>\n",
        "audiogen": "<|audiogen|>",
    }
    wave_concat_overlap = 400
    MODEL_PATH = None  # 需要用户提供


def sharegpt2BaichuanOmni15(messages):
    """
    将 sharegpt 格式的消息转换为 Baichuan-Omni-1.5 格式
    """
    new_messages = []
    
    for item in messages:
        role = "user" if item["from"] in ["human", "user"] else "assistant"
        content = item.get("value", "") or ""
        added_value = item.get("added_value", "") or ""
        speech = item.get("speech")
        
        if role == "user" and speech:
            # 如果有语音，需要在内容中包含音频 token
            # 优先使用 added_value，如果没有则使用 value
            # 这里我们保存原始信息，在构建输入时再处理
            text_to_use = added_value if added_value else content
            new_messages.append({
                "role": role,
                "content": text_to_use,
                "speech": speech,
                "added_value": added_value,  # 保存 added_value 标志
            })
        else:
            # 优先使用 added_value，如果没有则使用 value
            text_to_use = added_value if added_value else content
            new_messages.append({
                "role": role,
                "content": text_to_use,
            })
    
    return new_messages


class BaichuanOmni15:
    def __init__(
        self,
        model_path,
        cosy_vocoder_dir=None,
        input_speech=True,
        output_speech=False,
        speech_output_dir="BaichuanOmni15_output",
        dtype="bfloat16",
    ):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None
        self.round_idx = 0
        
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)
        
        # 检查必需的依赖
        self._check_dependencies()
        
        # 设置模型路径
        if MODEL_PATH is None:
            self.model_path = model_path
        else:
            self.model_path = MODEL_PATH
        
        # 设置 vocoder 路径
        if cosy_vocoder_dir is None:
            cosy_vocoder_dir = COSY_VOCODER
        
        # 加载模型和 tokenizer
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        
        # 加载模型和 tokenizer
        # 注意：模型加载可能会检查额外依赖，如果失败会在 check_imports 阶段报错
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=torch_dtype
            ).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.training = False
            self.model.bind_processor(self.tokenizer, training=False, relative_path="/")
            
            # 修补模型兼容性问题：某些版本的 transformers 可能没有 _validate_model_class 方法
            # generation.py 中的 GenerationAudioTokens.generate() 会调用这个方法
            if not hasattr(self.model, '_validate_model_class'):
                # 创建兼容性方法
                def _validate_model_class(model_instance):
                    """兼容性方法：验证模型类"""
                    # 如果模型有 _validate_model_kwargs，尝试调用它；否则直接返回
                    try:
                        if hasattr(model_instance, '_validate_model_kwargs'):
                            return model_instance._validate_model_kwargs()
                    except:
                        pass
                    return None
                
                # 将方法绑定到模型实例
                import types
                self.model._validate_model_class = types.MethodType(_validate_model_class, self.model)
            
            # 修补 _validate_assistant 方法：某些版本的 transformers 需要 tokenizer 参数
            # 检查 _validate_assistant 方法的签名
            if hasattr(self.model, '_validate_assistant'):
                import inspect
                try:
                    sig = inspect.signature(self.model._validate_assistant)
                    params = list(sig.parameters.keys())
                    
                    # 如果方法需要 tokenizer 参数，创建一个包装方法
                    if len(params) > 1 and ('tokenizer' in params or 'assistant_tokenizer' in params):
                        original_validate_assistant = self.model._validate_assistant
                        
                        def _validate_assistant_wrapper(model_instance, assistant_model=None, tokenizer=None, assistant_tokenizer=None):
                            """包装方法：为 _validate_assistant 提供默认参数"""
                            try:
                                # 尝试使用原始方法，传递 tokenizer 参数
                                if tokenizer is None:
                                    tokenizer = self.tokenizer
                                if assistant_tokenizer is None:
                                    assistant_tokenizer = self.tokenizer
                                
                                # 根据方法签名决定如何调用
                                if 'assistant_tokenizer' in params:
                                    return original_validate_assistant(assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
                                elif 'tokenizer' in params:
                                    return original_validate_assistant(assistant_model, tokenizer=tokenizer)
                                else:
                                    return original_validate_assistant(assistant_model)
                            except (TypeError, AttributeError) as e:
                                # 如果原始方法不接受这些参数，尝试不带参数调用
                                try:
                                    return original_validate_assistant(assistant_model)
                                except:
                                    # 如果都失败，直接返回 None（静默失败）
                                    return None
                        
                        self.model._validate_assistant = types.MethodType(_validate_assistant_wrapper, self.model)
                except (ValueError, TypeError):
                    # 如果无法获取签名，尝试创建一个通用的包装方法
                    original_validate_assistant = self.model._validate_assistant
                    
                    def _validate_assistant_wrapper(model_instance, assistant_model=None, tokenizer=None, assistant_tokenizer=None):
                        """通用包装方法：为 _validate_assistant 提供默认参数"""
                        try:
                            # 尝试传递 tokenizer 参数
                            if tokenizer is None:
                                tokenizer = self.tokenizer
                            if assistant_tokenizer is None:
                                assistant_tokenizer = self.tokenizer
                            return original_validate_assistant(assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
                        except TypeError:
                            try:
                                return original_validate_assistant(assistant_model, tokenizer=tokenizer)
                            except TypeError:
                                try:
                                    return original_validate_assistant(assistant_model)
                                except:
                                    return None
                    
                    self.model._validate_assistant = types.MethodType(_validate_assistant_wrapper, self.model)
            
            # 修补 _has_unfinished_sequences 方法：某些版本的 transformers 不接受 cur_len 和 max_length 参数
            # generation.py 中的代码可能传递了这些参数，但新版本不接受
            if hasattr(self.model, '_has_unfinished_sequences'):
                import inspect
                try:
                    sig = inspect.signature(self.model._has_unfinished_sequences)
                    params = list(sig.parameters.keys())
                    
                    # 检查方法是否接受 cur_len 和 max_length 参数
                    accepts_cur_len = 'cur_len' in params
                    accepts_max_length = 'max_length' in params
                    
                    # 如果方法不接受这些参数，创建一个包装方法
                    if not accepts_cur_len or not accepts_max_length:
                        original_has_unfinished = self.model._has_unfinished_sequences
                        
                        def _has_unfinished_sequences_wrapper(model_instance, *args, cur_len=None, max_length=None, **kwargs):
                            """包装方法：过滤掉新版本不接受的参数"""
                            # 构建新的 kwargs，移除不被接受的参数
                            filtered_kwargs = kwargs.copy()
                            if not accepts_cur_len and 'cur_len' in filtered_kwargs:
                                filtered_kwargs.pop('cur_len')
                            if not accepts_max_length and 'max_length' in filtered_kwargs:
                                filtered_kwargs.pop('max_length')
                            
                            # 调用原始方法，只传递被接受的参数
                            return original_has_unfinished(*args, **filtered_kwargs)
                        
                        self.model._has_unfinished_sequences = types.MethodType(_has_unfinished_sequences_wrapper, self.model)
                except (ValueError, TypeError):
                    # 如果无法获取签名，创建一个通用的包装方法，总是过滤掉 cur_len 和 max_length
                    original_has_unfinished = self.model._has_unfinished_sequences
                    
                    def _has_unfinished_sequences_wrapper(model_instance, *args, cur_len=None, max_length=None, **kwargs):
                        """通用包装方法：总是忽略 cur_len 和 max_length 参数"""
                        # 移除 cur_len 和 max_length 参数
                        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['cur_len', 'max_length']}
                        return original_has_unfinished(*args, **filtered_kwargs)
                    
                    self.model._has_unfinished_sequences = types.MethodType(_has_unfinished_sequences_wrapper, self.model)
        except ImportError as e:
            # 检查是否是缺少依赖
            error_msg = str(e).lower()
            missing_packages = []
            
            # 检查常见的缺失依赖
            if "easydict" in error_msg:
                missing_packages.append("easydict")
            if "speechbrain" in error_msg:
                missing_packages.append("speechbrain")
            # 可以添加更多依赖检查
            
            if missing_packages:
                raise ImportError(
                    f"模型加载失败：缺少必需的依赖包: {', '.join(missing_packages)}\n"
                    f"请安装: pip install {' '.join(missing_packages)}\n"
                    f"\n详细错误信息:\n{e}"
                )
            else:
                # 如果无法识别缺失的依赖，直接抛出原始错误
                raise ImportError(
                    f"模型加载失败，可能是缺少依赖包。\n"
                    f"请检查错误信息并安装相应的依赖。\n"
                    f"\n详细错误信息:\n{e}"
                ) from e
        
        # 加载 vocoder
        vocoder_path = os.path.join(cosy_vocoder_dir, "hift.pt")
        self.vocoder = Cosy24kVocoder.from_pretrained(vocoder_path)
        self.vocoder = self.vocoder.cuda()
        
        # 延迟导入 generation 模块（只在需要时导入）
        if self.output_speech:
            self._import_generation_module()
        
        # 获取特殊 token
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_start_token_id
        )
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_end_token_id
        )
        self.audiogen_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audiogen_start_token_id
        )
        self.audiogen_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audiogen_end_token_id
        )
        
        self.special_token_pattern = re.compile(
            r'<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>'
        )
        
        # 对话历史
        self.history = []
        
        # 创建缓存目录
        self.cache_dir = os.path.join(speech_output_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 音频拼接重叠参数
        self.wave_concat_overlap = wave_concat_overlap
    
    def _check_dependencies(self):
        """检查必需的依赖包"""
        missing_packages = []
        
        # 检查 easydict（模型加载需要）
        try:
            import easydict
        except ImportError:
            missing_packages.append("easydict")
        
        # 如果 output_speech=True，检查 speechbrain（音频生成需要）
        if self.output_speech:
            try:
                import speechbrain
            except ImportError:
                missing_packages.append("speechbrain")
        
        if missing_packages:
            raise ImportError(
                f"缺少必需的依赖包: {', '.join(missing_packages)}\n"
                f"请安装: pip install {' '.join(missing_packages)}\n"
                f"\n安装命令:\n"
                f"  pip install {' '.join(missing_packages)}"
            )
    
    def _import_generation_module(self):
        """延迟导入 generation 模块"""
        global decode_wave_vocoder, GenerationAudioTokens
        
        if decode_wave_vocoder is not None and GenerationAudioTokens is not None:
            return  # 已经导入
        
        # 在导入 generation 模块之前，先修补 transformers 兼容性问题
        # 某些版本的 transformers 可能没有 is_torchdynamo_compiling 或 is_quanto_available 函数
        try:
            import transformers.generation.utils as gen_utils
            # 修补 is_torchdynamo_compiling
            if not hasattr(gen_utils, 'is_torchdynamo_compiling'):
                def is_torchdynamo_compiling():
                    """兼容性函数：检查是否在 torchdynamo 编译模式下"""
                    try:
                        import torch._dynamo
                        return torch._dynamo.is_compiling()
                    except (ImportError, AttributeError):
                        return False
                gen_utils.is_torchdynamo_compiling = is_torchdynamo_compiling
            
            # 修补 is_quanto_available（某些版本可能没有）
            if not hasattr(gen_utils, 'is_quanto_available'):
                def is_quanto_available():
                    """兼容性函数：检查 quanto 是否可用"""
                    try:
                        import quanto
                        return True
                    except ImportError:
                        return False
                gen_utils.is_quanto_available = is_quanto_available
            
            # 验证修补是否成功
            if not hasattr(gen_utils, 'is_quanto_available'):
                raise RuntimeError("Failed to patch is_quanto_available in transformers.generation.utils")
        except (ImportError, AttributeError, RuntimeError) as patch_error:
            # 如果修补失败，记录警告但继续尝试导入
            import warnings
            warnings.warn(
                f"Failed to patch transformers.generation.utils: {patch_error}. "
                f"Generation module import may fail if it requires is_quanto_available."
            )
        
        first_error = None
        # 尝试多种导入路径
        try:
            # 首先尝试 web_demo.generation（这是最常见的情况，类似 Baichuan-Audio）
            from web_demo.generation import decode_wave_vocoder, GenerationAudioTokens
            return  # 成功导入
        except ImportError as e:
            first_error = e
            try:
                # 如果失败，尝试直接从项目根目录导入（官方代码可能使用这种方式）
                from generation import decode_wave_vocoder, GenerationAudioTokens
                return  # 成功导入
            except ImportError as e2:
                # 检查是否是依赖问题
                error_msg = str(first_error) + " | " + str(e2)
                if "speechbrain" in error_msg.lower():
                    raise ImportError(
                        f"Failed to import generation module due to missing dependency 'speechbrain'. "
                        f"Please install speechbrain: pip install speechbrain\n"
                        f"Or if speechbrain is not needed, check and modify generation.py to remove this dependency.\n"
                        f"First error: {first_error}\n"
                        f"Second error: {e2}"
                    )
                else:
                    # 最后检查文件是否存在，给出更明确的错误信息
                    web_demo_path = os.path.join(project_path, "web_demo", "generation.py")
                    root_generation_path = os.path.join(project_path, "generation.py")
                    
                    # 检查是否是 transformers 兼容性问题
                    error_str = str(first_error) + " | " + str(e2)
                    if "is_quanto_available" in error_str or "quanto" in error_str.lower():
                        raise ImportError(
                            f"Failed to import generation module due to transformers compatibility issue. "
                            f"The generation.py file requires 'is_quanto_available' from transformers.generation.utils, "
                            f"but it's not available in your transformers version.\n"
                            f"Possible solutions:\n"
                            f"1. Update transformers: pip install --upgrade transformers\n"
                            f"2. Or modify web_demo/generation.py to handle missing is_quanto_available\n"
                            f"3. Or set output_speech=False to skip audio generation\n"
                            f"\nFirst error: {first_error}\n"
                            f"Second error: {e2}"
                        )
                    elif os.path.exists(web_demo_path):
                        raise ImportError(
                            f"Found generation.py at {web_demo_path} but failed to import. "
                            f"First error: {first_error}\n"
                            f"Second error: {e2}\n"
                            f"Please check if web_demo is a proper Python package with __init__.py, "
                            f"or install missing dependencies (e.g., pip install speechbrain)."
                        )
                    elif os.path.exists(root_generation_path):
                        raise ImportError(
                            f"Found generation.py at {root_generation_path} but failed to import. "
                            f"First error: {first_error}\n"
                            f"Second error: {e2}\n"
                            f"Please check the project structure or install missing dependencies (e.g., pip install speechbrain)."
                        )
                    else:
                        raise ImportError(
                            f"Cannot find generation module. "
                            f"Expected one of: {web_demo_path} or {root_generation_path}. "
                            f"Please check if Baichuan-Omni-1.5 project exists at {project_path}"
                        )
    
    def _wave_concat(self, wave_list, start, overlap=400):
        """拼接波形，带重叠处理"""
        new_wave_list = []
        cur = start
        for wave in wave_list[start:]:
            if (
                cur - 1 >= 0
                and wave_list[cur - 1].shape[1] > overlap
                and wave.shape[1] > overlap
            ):
                new_wave_list.append(
                    (
                        wave_list[cur - 1][:, -overlap:]
                        * torch.linspace(
                            1.0, 0.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                        + wave[:, :overlap]
                        * torch.linspace(
                            0.0, 1.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                    )
                )
            new_wave_list.append(wave)
            cur += 1
        return torch.cat(new_wave_list, dim=1)
    
    def _save_local(self, wave, local_path):
        """保存音频文件并返回 token 格式"""
        # wave 应该是一个列表，需要拼接
        if isinstance(wave, list):
            wave_tensor = torch.cat(wave, dim=0)
        else:
            wave_tensor = wave
        torchaudio.save(local_path, wave_tensor.cpu(), sampling_rate)
        return (
            self.audiogen_start_token
            + ujson.dumps({"path": local_path}, ensure_ascii=False)
            + self.audiogen_end_token
        )
    
    def _generate_text_step(self, pret, plen, kv_cache_flag, audiogen_flag=True):
        """生成文本步骤"""
        if not kv_cache_flag:
            textret = self.model.generate(
                pret.input_ids.cuda(),
                attention_mask=pret.attention_mask.cuda(),
                audios=pret.audios.cuda() if pret.audios is not None else None,
                encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
                bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
                tokenizer=self.tokenizer,
                max_new_tokens=50 if audiogen_flag else 1024,
                stop_strings=[self.audiogen_start_token, "<|endoftext|>"] if audiogen_flag else ["<|endoftext|>"],
                do_sample=True,
                temperature=0.8,
                top_k=20,
                top_p=0.85,
                repetition_penalty=1.1,
                return_dict_in_generate=True,
            )
        else:
            textret = self.model.generate(
                pret.sequences,
                attention_mask=torch.ones_like(pret.sequences),
                tokenizer=self.tokenizer,
                past_key_values=pret.past_key_values,
                stop_strings=[self.audiogen_start_token, ",", "!", "?", "，", "。", "！", "？", ". "],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.3,
                top_k=20,
                top_p=0.85,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
            )
        newtext = self.tokenizer.decode(textret.sequences[0, plen:])
        return textret, newtext
    
    def _generate_audio_step(self, pret):
        """生成音频步骤"""
        # 确保 generation 模块已导入
        if GenerationAudioTokens is None or decode_wave_vocoder is None:
            self._import_generation_module()
        
        # 调用 GenerationAudioTokens.generate，传递 tokenizer 参数
        # 某些版本的 transformers 需要 tokenizer 参数用于 _validate_assistant
        audioret = GenerationAudioTokens.generate(
            self.model,
            pret.sequences,
            attention_mask=torch.ones_like(pret.sequences),
            past_key_values=pret.past_key_values if pret.past_key_values is not None else None,
            tokenizer=self.tokenizer,  # 传递 tokenizer 参数
            max_new_tokens=500,
            do_sample=True,
            temperature=0.5,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.3,
            return_dict_in_generate=True,
        )
        wave_segment = decode_wave_vocoder(audioret.audios_sequences.clone(), self.vocoder, self.model)
        return audioret, wave_segment
    
    def _preprocess_messages(self, messages, audiogen_flag=True):
        """预处理消息格式"""
        text = ""
        for i, msg in enumerate(messages):
            if audiogen_flag and msg["role"] == "assistant":
                text += role_prefix["audiogen"]
            text += role_prefix[msg["role"]]
            text += msg["content"]
        if audiogen_flag:
            text += role_prefix["audiogen"]
        text += role_prefix["assistant"]
        return text
    
    def _load_audio(self, audio_path):
        """加载音频文件"""
        wave, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            wave = torchaudio.functional.resample(wave, sr, sampling_rate)
        wave_pkg = (
            sampling_rate,
            (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(np.int16),
        )
        return wave_pkg
    
    def reply(self, messages, round_idx=0):
        """
        生成回复（非流式）
        
        Args:
            messages: sharegpt 格式的消息列表
            round_idx: 当前轮次索引
            
        Returns:
            dict: {"text": str, "speech": str or None}
        """
        # 转换消息格式
        new_messages = sharegpt2BaichuanOmni15(messages)
        
        # 处理音频输入
        processed_messages = []
        for msg in new_messages:
            if msg.get("speech"):
                # 如果有语音，需要包装成音频 token 格式
                audio_path = os.path.abspath(msg["speech"])
                audio_token = (
                    self.audio_start_token
                    + ujson.dumps({"path": audio_path}, ensure_ascii=False)
                    + self.audio_end_token
                )
                # 逻辑：如果有 added_value 且有 speech，使用 added_value + audio_token
                # 如果没有 added_value 但有 speech，只使用 audio_token（忽略 content）
                has_added_value = msg.get("added_value", "") and msg.get("added_value", "").strip()
                if has_added_value:
                    # 有 added_value 和 speech：使用 added_value + audio_token
                    combined_content = msg["content"] + "\n" + audio_token
                    processed_messages.append({
                        "role": msg["role"],
                        "content": combined_content,
                    })
                else:
                    # 只有语音（如果没有 added_value，即使有 content 也只用 audio_token）
                    processed_messages.append({
                        "role": msg["role"],
                        "content": audio_token,
                    })
            else:
                processed_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        # 构建输入
        content = self._preprocess_messages(processed_messages, audiogen_flag=self.output_speech)
        pret = self.model.processor([content])
        plen = pret.input_ids.shape[1]
        
        # 生成文本
        ret, text_segment = self._generate_text_step(pret, plen, False, audiogen_flag=self.output_speech)
        
        full_text = re.sub(self.special_token_pattern, "", text_segment)
        wave_list = []
        
        if self.output_speech:
            # 流式生成音频和文本（按照官方代码的逻辑）
            start = 0
            output_dir = os.path.join(
                self.speech_output_dir,
                str(self.conv_id) if self.conv_id else "default",
            )
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(100):
                m = ret.sequences[0, -1].item()
                if m == self.tokenizer.eos_token_id:
                    # 如果是 eos token，检查是否还有内容需要生成音频
                    if ret.sequences.shape[1] - plen > 1:
                        ret.sequences[0, -1] = self.model.config.audio_config.audiogen_start_token_id
                        ret, wave_segment = self._generate_audio_step(ret)
                        wave_list.extend(wave_segment)
                        # 保存音频片段
                        audio_path = os.path.join(output_dir, f"assistant_turn{round_idx}_round{i}.wav")
                        full_text += self._save_local(wave_segment, audio_path)
                    break
                
                # 替换为 audiogen_start_token 并生成音频
                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_start_token_id
                ret, wave_segment = self._generate_audio_step(ret)
                wave_list.extend(wave_segment)
                
                # 保存音频片段
                audio_path = os.path.join(output_dir, f"assistant_turn{round_idx}_round{i}.wav")
                full_text += self._save_local(wave_segment, audio_path)
                
                # 替换为 audiogen_end_token 并生成下一段文本
                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_end_token_id
                plen = ret.sequences.shape[1]
                ret, text_segment = self._generate_text_step(ret, plen, True, True)
                full_text += re.sub(self.special_token_pattern, "", text_segment)
                
                # 检查是否应该结束
                if self.tokenizer.eos_token_id in ret.sequences[0]:
                    break
            
            # 拼接所有音频片段
            speech_path = None
            if wave_list:
                if len(wave_list) > start:
                    wave = self._wave_concat(wave_list, start, overlap=self.wave_concat_overlap)
                    # 保存最终拼接的音频
                    speech_path = os.path.join(output_dir, f"doctor_{round_idx}.wav")
                    # 转换为正确的格式并保存
                    wave_clamped = torch.clamp(wave.squeeze(), -0.99, 0.99)
                    torchaudio.save(speech_path, wave_clamped.unsqueeze(0).cpu(), sampling_rate)
        else:
            speech_path = None
        
        # 清理特殊 token
        final_text = re.sub(self.special_token_pattern, "", full_text).strip()
        
        return {"text": final_text, "speech": speech_path}


if __name__ == "__main__":
    # 使用示例
    # 注意：如果 output_speech=True，需要安装 speechbrain: pip install speechbrain
    # model = BaichuanOmni15(
    #     model_path="../weight/Baichuan-Omni-1d5",
    #     input_speech=True,
    #     output_speech=False,  # 如果需要音频输出，确保已安装 speechbrain
    # )
    # model.conv_id = 0
    
    # result = model.reply([
    #     {
    #         "from": "user",
    #         "value": "",
    #         "speech": "./test2.mp3"
    #     }
    # ], round_idx=0)
    
    # print("Text:", result["text"])
    # print("Speech:", result["speech"])

    model = BaichuanOmni15(
        model_path="../weight/Baichuan-Omni-1d5",
        input_speech=False,
        output_speech=False,  # 如果需要音频输出，确保已安装 speechbrain
    )
    model.conv_id = 0
    
    result = model.reply([
        {
            "from": "user",
            "value": "下列有关鳃弓的描述，错误的是A: 由间充质增生形成 B: 人胚第4周出现 C: 相邻鳃弓之间为鳃沟 D: 共5对鳃弓 E: 位于头部两侧\n请你直接回答正确答案选项和选项内容。"
        }
    ], round_idx=0)
    
    print("Text:", result["text"])
    print("Speech:", result["speech"])
