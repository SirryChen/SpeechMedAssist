# use env MiniCPMo26
import os
import sys
import torch
import librosa
from transformers import AutoModel, AutoTokenizer


def sharegpt2MiniCPMo26(messages):
    """
    将 sharegpt 格式的消息转换为 MiniCPM-o-2.6 格式
    """
    new_messages = []
    
    for item in messages:
        role = "user" if item["from"] in ["human", "user"] else "assistant"
        
        if role == "user":
            # 用户消息：包含音频
            if item.get("speech") is not None and item["speech"]:
                # 加载音频文件
                audio, _ = librosa.load(item["speech"], sr=16000, mono=True)
                new_messages.append({
                    "role": "user",
                    "content": [audio]
                })
            else:
                # 如果没有音频，使用文本（如果模型支持）
                if "value" in item and item["value"]:
                    new_messages.append({
                        "role": "user",
                        "content": item["value"]
                    })
        else:
            # 助手消息：文本回复
            if "value" in item and item["value"]:
                new_messages.append({
                    "role": "assistant",
                    "content": item["value"]
                })
    
    return new_messages


class MiniCPMo26:
    def __init__(self, model_path='openbmb/MiniCPM-o-2_6', input_speech=True, output_speech=False, 
                 speech_output_dir="MiniCPMo26_output", ref_audio_path=None, language='en'):
        """
        初始化 MiniCPM-o-2.6 模型
        
        Args:
            model_path: 模型路径或 HuggingFace 模型名称
            input_speech: 是否支持语音输入
            output_speech: 是否支持语音输出
            speech_output_dir: 语音输出目录
            ref_audio_path: 参考音频路径（用于TTS）
            language: 语言设置（'en' 或 'zh'）
        """
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None
        self.language = language
        
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)
        
        # 处理模型路径
        # 判断是否是 HuggingFace 模型名称（如 'openbmb/MiniCPM-o-2_6'）
        # HuggingFace 模型名称特征：包含 '/' 但不以 './' 或 '../' 开头，且不是绝对路径
        is_hf_model_name = (
            not os.path.isabs(model_path) and 
            '/' in model_path and 
            not model_path.startswith('./') and 
            not model_path.startswith('../')
        )
        
        # 对于本地路径，转换为绝对路径（相对于当前工作目录）
        if not is_hf_model_name:
            if not os.path.isabs(model_path):
                # 相对于当前工作目录解析
                model_path = os.path.abspath(model_path)
            
            # 确保模型路径存在（HuggingFace 模型名称会跳过此检查）
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # 尝试修复可能损坏的缓存（如果存在）
        # transformers 会将自定义模块缓存到 ~/.cache/huggingface/modules/transformers_modules/
        import shutil
        import glob
        cache_module_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules/MiniCPM-o-2_6')
        required_files = ['image_processing_minicpmv.py', 'modeling_minicpmo.py', 'configuration_minicpm.py']
        
        # 检查并修复缓存
        need_fix = False
        if os.path.exists(cache_module_dir):
            # 检查缓存目录中的关键文件是否存在
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(cache_module_dir, f))]
            if missing_files:
                need_fix = True
        else:
            need_fix = True
        
        if need_fix:
            # 删除缓存目录（如果存在）
            if os.path.exists(cache_module_dir):
                print(f"Cache is incomplete, removing cache directory...")
                shutil.rmtree(cache_module_dir, ignore_errors=True)
            
            # 创建缓存目录并复制所有 Python 自定义模块文件
            print("Copying custom module files from model directory to cache...")
            os.makedirs(cache_module_dir, exist_ok=True)
            
            # 查找模型目录中所有 Python 文件（自定义模块）
            python_files = glob.glob(os.path.join(model_path, '*.py'))
            for py_file in python_files:
                filename = os.path.basename(py_file)
                # 只复制自定义模块文件（排除可能的其他文件）
                if filename in required_files or filename.startswith(('modeling_', 'configuration_', 'image_processing_', 'processing_')):
                    dest_file = os.path.join(cache_module_dir, filename)
                    shutil.copy2(py_file, dest_file)
                    print(f"  Copied {filename}")
                    
                    # 如果是 modeling 文件，尝试在文件末尾添加补丁代码
                    if filename.startswith('modeling_'):
                        try:
                            with open(dest_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # 检查是否有 Resampler 类且没有 _initialize_weights 方法
                            if 'Resampler' in content and '_initialize_weights' not in content:
                                # 在文件末尾添加补丁代码
                                patch_code = '''

# Patch for transformers compatibility: add _initialize_weights to Resampler classes
import inspect
for name, obj in list(globals().items()):
    if (inspect.isclass(obj) and 'Resampler' in name and 
        not hasattr(obj, '_initialize_weights')):
        def _initialize_weights(self):
            """Empty method to avoid transformers initialization error"""
            pass
        obj._initialize_weights = _initialize_weights
'''
                                with open(dest_file, 'a', encoding='utf-8') as f:
                                    f.write(patch_code)
                                print(f"  Added patch code to {filename}")
                        except Exception as e:
                            print(f"  Warning: Could not patch {filename}: {e}")
            
            print("Cache fixed. Ready to load model.")
        
        # 修补 torch.nn.Module 的 __getattr__ 方法，为 _initialize_weights 返回空方法
        import torch.nn as nn
        original_getattr = nn.Module.__getattr__
        
        def patched_getattr(self, name):
            """修补的 __getattr__，为 _initialize_weights 返回空方法"""
            if name == '_initialize_weights':
                # 如果模块没有 _initialize_weights 方法，添加一个
                if not hasattr(self.__class__, '_initialize_weights'):
                    # 定义一个接受任意参数的函数（smart_apply 会传递 self）
                    def _empty_init_weights(*args, **kwargs):
                        """Empty method to avoid transformers initialization error"""
                        pass
                    self.__class__._initialize_weights = _empty_init_weights
                # 返回未绑定的方法，让 smart_apply 可以正确调用
                return self.__class__._initialize_weights
            return original_getattr(self, name)
        
        # 临时替换方法
        nn.Module.__getattr__ = patched_getattr
        
        try:
            # 加载模型
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                attn_implementation='sdpa',  # sdpa or flash_attention_2, no eager
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False
            )
        finally:
            # 恢复原始方法
            nn.Module.__getattr__ = original_getattr
        
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 初始化TTS（如果需要语音输出）
        if output_speech:
            self.model.init_tts()
            self.model.tts.float()
        
        # 加载参考音频并获取系统提示
        if ref_audio_path is None:
            # 默认使用女性助手声音
            ref_audio_path = './assets/input_examples/assistant_female_voice.wav'
        
        if os.path.exists(ref_audio_path):
            self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
            self.sys_prompt = self.model.get_sys_prompt(
                ref_audio=self.ref_audio, 
                mode='audio_assistant', 
                language=language
            )
        else:
            print(f"Warning: Reference audio file not found: {ref_audio_path}")
            self.ref_audio = None
            self.sys_prompt = None
        
        self.round_idx = 0
        self.history = []  # 保存对话历史
    
    def reset_history(self):
        """重置对话历史，开始新的对话"""
        self.history = []
        self.round_idx = 0
    
    def reply(self, messages, round_idx=0):
        """
        处理消息并返回回复
        
        Args:
            messages: sharegpt 格式的消息列表
            round_idx: 当前轮次索引
        
        Returns:
            dict: 包含 'text' 和 'speech' 的字典
        """
        # 转换消息格式
        new_messages = sharegpt2MiniCPMo26(messages)
        
        # 检查是否有音频输入（用于 TTS 生成 speaker embedding）
        # 如果 output_speech=True 但没有音频输入，可能会导致 spk_bounds 为空
        try:
            import numpy as np
            has_audio_input = any(
                isinstance(msg.get("content"), list) and 
                any(isinstance(item, (list, np.ndarray)) for item in msg.get("content", []))
                for msg in new_messages
            )
        except ImportError:
            # 如果没有 numpy，使用更简单的检查
            has_audio_input = any(
                isinstance(msg.get("content"), list) and len(msg.get("content", [])) > 0
                for msg in new_messages
            )
        
        # 如果 output_speech=True 但没有音频输入，警告并考虑禁用 TTS
        if self.output_speech and not has_audio_input:
            import warnings
            warnings.warn(
                "output_speech=True but no audio input found. "
                "TTS may fail due to missing speaker embeddings. "
                "Consider setting output_speech=False or providing audio input."
            )
        
        # 检查传入的消息是否包含助手回复（说明是完整历史）
        has_assistant = any(item.get("from") not in ["human", "user"] for item in messages)
        
        if has_assistant:
            # 如果传入的是完整历史，使用转换后的完整消息
            # 检查是否需要添加系统提示
            if self.sys_prompt is not None and (len(new_messages) == 0 or new_messages[0].get("role") != "system"):
                msgs = [self.sys_prompt] + new_messages
            else:
                msgs = new_messages
            # 重置内部历史，因为使用了完整历史
            self.history = []
        else:
            # 如果只传入新的用户消息，使用内部历史记录
            if len(self.history) == 0:
                # 第一轮对话，添加系统提示
                if self.sys_prompt is not None:
                    msgs = [self.sys_prompt] + new_messages
                else:
                    msgs = new_messages
            else:
                # 后续轮次，使用历史记录
                msgs = self.history + new_messages
        
        # 生成参数
        sampling_params = {
            'sampling': True,
            'max_new_tokens': 128,
            'use_tts_template': True if self.output_speech else False,
            'generate_audio': True if self.output_speech else False,
            'temperature': 0.3,
        }
        
        # 如果需要输出语音，设置输出路径
        if self.output_speech:
            os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
            output_audio_path = os.path.join(
                self.speech_output_dir, 
                str(self.conv_id), 
                f"doctor_{round_idx}.wav"
            )
            sampling_params['output_audio_path'] = output_audio_path
        else:
            output_audio_path = None
        
        # 调用模型进行推理
        try:
            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                **sampling_params
            )
        except (ValueError, IndexError) as e:
            # 捕获 transformers 版本兼容性问题和 spk_bounds 索引错误
            error_str = str(e)
            is_compatibility_error = (
                'past_key_values' in error_str or 
                'Cache' in error_str or 
                'spk_bounds' in error_str or
                'index -1 is out of bounds' in error_str
            )
            
            if is_compatibility_error:
                # transformers 版本兼容性问题或 spk_bounds 为空
                error_msg = (
                    f"Model compatibility error: {e}\n"
                    "This error may be caused by:\n"
                    "1. Version mismatch between transformers and the model's TTS module\n"
                    "2. Missing speaker embeddings (spk_bounds is empty)\n"
                    "3. Incompatible input format\n"
                    "Possible solutions:\n"
                    "1. Try updating transformers: pip install --upgrade transformers\n"
                    "2. Or try downgrading transformers to a compatible version\n"
                    "3. Or disable TTS output by setting output_speech=False"
                )
                if self.output_speech:
                    print(f"Warning: {error_msg}")
                    print("Attempting to retry without TTS...")
                    # 尝试不使用 TTS（完全禁用 TTS 相关功能）
                    sampling_params_no_tts = {
                        'sampling': True,
                        'max_new_tokens': 128,
                        'use_tts_template': False,  # 完全禁用 TTS 模板
                        'generate_audio': False,   # 完全禁用音频生成
                        'temperature': 0.3,
                    }
                    try:
                        res = self.model.chat(
                            msgs=msgs,
                            tokenizer=self.tokenizer,
                            **sampling_params_no_tts
                        )
                        output_audio_path = None
                        print("Successfully generated text response without TTS.")
                    except Exception as e2:
                        # 如果禁用 TTS 后仍然失败，抛出原始错误
                        raise ValueError(f"Failed to generate response even without TTS: {e2}") from e
                else:
                    raise ValueError(error_msg) from e
            else:
                raise
        
        # 更新历史记录：将当前消息和回复都添加到历史中
        # 注意：msgs 已经包含了当前用户消息，只需要添加助手回复
        self.history = msgs.copy()
        if isinstance(res, str):
            self.history.append({'role': 'assistant', 'content': res})
        else:
            # 如果返回的是其他格式，尝试提取文本
            self.history.append({'role': 'assistant', 'content': str(res)})
        
        # 返回结果
        result = {
            "text": res if isinstance(res, str) else str(res),
            "speech": output_audio_path if self.output_speech and output_audio_path else None
        }
        
        return result


if __name__ == "__main__":
    # 测试代码
    # TTS 模块不可用，需要修改库
    # model = MiniCPMo26(
    #     model_path='../weight/MiniCPM-o-2_6',
    #     input_speech=True,
    #     output_speech=False,
    #     ref_audio_path='./prompt_zh.wav',
    #     language='en'
    # )
    # model.conv_id = 0
    
    # print(model.reply([
    #     {
    #         "from": "user",
    #         "value": "",
    #         "speech": "./test2.mp3"
    #     }
    # ]))

    model = MiniCPMo26(
        model_path='../weight/MiniCPM-o-2_6',
        input_speech=False,
        output_speech=False,
        ref_audio_path='./prompt_zh.wav',
        language='en'
    )
    model.conv_id = 0
    
    print(model.reply([
        {
            "from": "user",
            "value": "医生，我有个单项选择题想考你一下。\n下列有关鳃弓的描述，错误的是A: 由间充质增生形成 B: 人胚第4周出现 C: 相邻鳃弓之间为鳃沟 D: 共5对鳃弓 E: 位于头部两侧\n请你直接回答正确答案选项和选项内容。"
        }
    ]))

