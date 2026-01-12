# use env Qwen25Omni
import os
import sys
import torch
import soundfile as sf
import logging
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

logger = logging.getLogger(__name__)


def sharegpt2Qwen25Omni(messages):
    new_messages = []

    for item in messages:
        role = "user" if item["from"] in ["human", "user"] else "assistant"
        content = []

        # 音频（仅 user 可能有 speech）
        if role == "user" and item.get("speech") is not None and item["speech"]:
            content.append({
                "type": "audio",
                "audio": item["speech"]
            })

        # 文本
        if "value" in item and item["value"]:
            content.append({
                "type": "text",
                "text": item["value"]
            })

        # 加入新格式对话
        new_messages.append({
            "role": role,
            "content": content
        })

    return new_messages


class Qwen25Omni:
    def __init__(self, model_path, input_speech=True, output_speech=False, speech_output_dir="Qwen25Omni_output"):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        self.round_idx = 0

    def reply(self, messages, round_idx=0):
        # 根据是否需要音频输出决定如何处理音频输入
        # 关键修复：如果 output_speech=False，即使 input_speech=True，我们也不在输入中包含音频
        # 因为模型内部可能会根据输入中是否有音频来决定是否生成音频，导致 CUDA 错误
        # 这是一个权衡：我们选择避免 CUDA 错误，而不是让模型理解音频输入
        use_audio_for_input = self.input_speech and self.output_speech  # 只有同时需要输入和输出时才处理音频
        use_audio_for_output = self.output_speech  # 是否生成音频输出
        
        # 转换消息格式，如果不需要音频输入，不包含音频内容
        # 这样可以避免 processor 尝试处理音频 token 但 audio_lengths 为空的问题
        if use_audio_for_input:
            new_messages = sharegpt2Qwen25Omni(messages)
        else:
            # 不包含音频内容，只保留文本
            new_messages = []
            for item in messages:
                role = "user" if item["from"] in ["human", "user"] else "assistant"
                content = []
                # 只添加文本，不添加音频
                if "value" in item and item["value"]:
                    content.append({
                        "type": "text",
                        "text": item["value"]
                    })
                if content:  # 只有当有内容时才添加
                    new_messages.append({
                        "role": role,
                        "content": content
                    })

        # 处理文本模板
        text = self.processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)
        
        # 处理多媒体信息（音频、图像、视频）
        # 如果不需要音频输出，不处理音频，避免模型内部尝试使用
        if use_audio_for_input:
            audios, images, videos = process_mm_info(new_messages, use_audio_in_video=True)
        else:
            # 不处理音频，只处理图像和视频（如果有）
            mm_info = process_mm_info(new_messages, use_audio_in_video=False)
            audios = None
            if len(mm_info) >= 2:
                images, videos = mm_info[1], mm_info[2] if len(mm_info) > 2 else []
            else:
                images, videos = [], []
        
        # 处理输入
        # 如果不需要音频输出，不传入音频，避免模型内部尝试处理
        # 注意：如果 use_audio_for_input=False，audios 应该是 None 或空列表
        inputs = self.processor(
            text=text,
            audio=audios if use_audio_for_input else None,  # 不需要音频时传入 None，避免空列表导致错误
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_for_input  # 根据是否需要音频输入决定
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # 根据 output_speech 参数决定是否生成音频
        return_audio = use_audio_for_output
        
        # 生成输出，添加错误处理
        try:
            # 生成参数设置
            # 如果不需要音频输出，完全禁用音频生成以避免 CUDA 错误
            if not return_audio:
                # 只进行文本生成，禁用所有音频输出相关处理
                # 关键：即使设置了 use_audio_in_video=False，如果输入包含音频特征，
                # 模型内部可能仍会尝试使用它们，导致 CUDA 错误
                # 因此我们需要在生成前就移除所有音频相关的输入
                generate_inputs = {}
                # 扩展音频相关键列表，包括所有可能的变体
                audio_keys_to_skip = [
                    'audio_features', 'audio_feature_lengths', 'audio', 'audios',
                    'audio_input_ids', 'audio_attention_mask', 'audio_position_ids',
                    'audio_values', 'audio_values_lengths', 'pixel_values',  # 某些模型可能使用这些
                    'audio_embeds', 'audio_embeddings', 'audio_tokens'
                ]
                for key, value in inputs.items():
                    # 跳过所有音频相关的键，以及任何包含 'audio' 的键
                    if key not in audio_keys_to_skip and 'audio' not in key.lower():
                        generate_inputs[key] = value
                    else:
                        logger.debug(f"移除了音频相关输入键: {key}")
                
                # 记录是否移除了音频特征
                removed_keys = [k for k in inputs.keys() if k in audio_keys_to_skip or 'audio' in k.lower()]
                if removed_keys:
                    logger.debug(f"移除了 {len(removed_keys)} 个音频相关输入以避免 CUDA 错误（output_speech=False）")
                
                # 在生成前清理 CUDA 缓存，确保没有残留的音频相关状态
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import time
                        time.sleep(0.1)  # 短暂等待，让 CUDA 状态稳定
                    except:
                        pass
                
                generate_kwargs = {
                    "use_audio_in_video": False,  # 生成时禁用音频处理
                    "return_audio": False
                }
            else:
                generate_inputs = inputs
                generate_kwargs = {
                    "use_audio_in_video": True,
                    "return_audio": True
                }
            output = self.model.generate(**generate_inputs, **generate_kwargs)
            
            # 只提取新生成的部分（去掉输入部分）
            generated_ids = output[0] if isinstance(output, tuple) else output
            # 使用 generate_inputs 中的 input_ids，因为可能已经移除了某些键
            input_length = generate_inputs['input_ids'].size(1)
            generated_ids = generated_ids[:, input_length:]

            # 解码文本和音频
            reply = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            audio = output[1] if return_audio and isinstance(output, tuple) and len(output) > 1 else None

            # 保存音频（如果需要）
            speech = None
            if self.output_speech and audio is not None:
                os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
                output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
                # 将音频tensor转换为numpy并保存
                if isinstance(audio, torch.Tensor):
                    audio_numpy = audio.detach().cpu().view(-1).numpy()
                else:
                    audio_numpy = audio
                sf.write(output_path, audio_numpy, 24000)
                speech = output_path

            return {"text": reply, "speech": speech}
            
        except RuntimeError as e:
            error_str = str(e)
            # 捕获所有 CUDA 相关错误，包括 device-side assert 和 CUBLAS 错误
            if any(keyword in error_str for keyword in ["CUDA error", "device-side assert", "indexSelectLargeIndex", 
                                                         "cuda", "CUBLAS", "cublas"]):
                logger.warning(f"生成时发生 CUDA 错误，尝试回退到仅文本生成: {error_str[:300]}")
                # 清理 CUDA 缓存和同步，更彻底地清理状态
                if torch.cuda.is_available():
                    try:
                        # 同步所有 CUDA 操作
                        torch.cuda.synchronize()
                        # 清空缓存
                        torch.cuda.empty_cache()
                        # 再次同步确保清理完成
                        torch.cuda.synchronize()
                        # 如果可能，重置 CUDA 设备状态
                        import time
                        time.sleep(0.1)  # 短暂等待，让 CUDA 状态稳定
                    except Exception as cleanup_error:
                        logger.debug(f"CUDA 清理过程中出现错误（可忽略）: {cleanup_error}")
                
                # 回退到仅文本生成（不返回音频）
                try:
                    # 强制禁用音频输出，只生成文本
                    # 使用更保守的参数设置
                    # 如果输入包含音频，尝试移除音频相关输入以避免索引问题
                    # inputs 可能是 BatchEncoding 对象，需要转换为字典并过滤
                    safe_inputs = {}
                    audio_keys_to_skip = ['audio_features', 'audio_feature_lengths', 'audio', 'audios',
                                          'audio_input_ids', 'audio_attention_mask', 'audio_position_ids']
                    for key, value in inputs.items():
                        if key not in audio_keys_to_skip:
                            safe_inputs[key] = value
                    
                    if any(key in inputs for key in audio_keys_to_skip):
                        logger.warning("检测到音频输入导致 CUDA 错误，尝试移除音频特征后重新生成")
                    
                    output = self.model.generate(
                        **safe_inputs, 
                        use_audio_in_video=False,  # 完全禁用音频相关处理
                        return_audio=False
                    )
                    generated_ids = output[0] if isinstance(output, tuple) else output
                    input_length = safe_inputs['input_ids'].size(1)
                    generated_ids = generated_ids[:, input_length:]
                    reply = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    logger.info("成功回退到仅文本生成模式（已禁用音频处理）")
                    return {"text": reply, "speech": None}
                except Exception as e2:
                    logger.error(f"文本生成也失败: {e2}")
                    # 再次清理缓存
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except:
                            pass
                    raise
            else:
                # 其他类型的错误直接抛出
                raise
        except Exception as e:
            # 捕获其他可能的异常
            error_str = str(e)
            if "CUDA" in error_str or "cuda" in error_str:
                logger.warning(f"捕获到 CUDA 相关异常: {error_str[:300]}")
                # 清理 CUDA 缓存，更彻底地清理状态
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import time
                        time.sleep(0.1)  # 短暂等待，让 CUDA 状态稳定
                    except Exception as cleanup_error:
                        logger.debug(f"CUDA 清理过程中出现错误（可忽略）: {cleanup_error}")
                # 尝试回退到仅文本生成
                try:
                    # 如果输入包含音频，尝试移除音频相关输入
                    safe_inputs = {}
                    audio_keys_to_skip = ['audio_features', 'audio_feature_lengths', 'audio', 'audios',
                                          'audio_input_ids', 'audio_attention_mask', 'audio_position_ids']
                    for key, value in inputs.items():
                        if key not in audio_keys_to_skip:
                            safe_inputs[key] = value
                    
                    if any(key in inputs for key in audio_keys_to_skip):
                        logger.warning("检测到音频输入导致 CUDA 错误，尝试移除音频特征后重新生成")
                    
                    output = self.model.generate(
                        **safe_inputs, 
                        use_audio_in_video=False,
                        return_audio=False
                    )
                    generated_ids = output[0] if isinstance(output, tuple) else output
                    input_length = safe_inputs['input_ids'].size(1)
                    generated_ids = generated_ids[:, input_length:]
                    reply = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    logger.info("成功回退到仅文本生成模式")
                    return {"text": reply, "speech": None}
                except Exception as e3:
                    logger.error(f"回退到仅文本生成也失败: {e3}")
                    # 最后尝试：完全移除所有可能的音频相关输入，只使用文本
                    try:
                        text_only_inputs = {
                            'input_ids': inputs.input_ids,
                            'attention_mask': inputs.attention_mask
                        }
                        output = self.model.generate(
                            **text_only_inputs,
                            use_audio_in_video=False,
                            return_audio=False
                        )
                        generated_ids = output[0] if isinstance(output, tuple) else output
                        input_length = text_only_inputs['input_ids'].size(1)
                        generated_ids = generated_ids[:, input_length:]
                        reply = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        logger.info("成功使用纯文本输入生成回复")
                        return {"text": reply, "speech": None}
                    except:
                        raise e3
            else:
                raise


if __name__ == "__main__":
    # model = Qwen25Omni(model_path="../weight/Qwen2.5-Omni-7B", output_speech=True)
    # model.conv_id = 0
    # print(model.reply([
    #     {
    #         "from": "user",
    #         "value": "",
    #         "speech": "./test2.mp3"
    #     }
    # ]))

    model = Qwen25Omni(model_path="../weight/Qwen2.5-Omni-7B", input_speech=False, output_speech=False)
    model.conv_id = 0
    print(model.reply([
        {
            "from": "user",
            "value": "下列有关鳃弓的描述，错误的是A: 由间充质增生形成 B: 人胚第4周出现 C: 相邻鳃弓之间为鳃沟 D: 共5对鳃弓 E: 位于头部两侧\n请你直接回答正确答案选项和选项内容。"
        }
    ]))

