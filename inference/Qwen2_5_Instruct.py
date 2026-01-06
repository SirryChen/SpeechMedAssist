import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchaudio
from .utils import ASRModel, TTSModel, sharegpt_old2new


class Qwen2_5_Instruct:
    def __init__(self, model_path, input_speech=False, output_speech=False, speech_output_dir="Qwen2_5_Instruct_output"):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        if input_speech:
            self.asr_model = ASRModel()

        if output_speech:
            self.TTSModel = TTSModel()


    def reply(self, messages, round_idx=0):
        if self.input_speech:
            for i, turn in enumerate(messages):
                if i % 2 == 0:
                    turn["value"] = self.asr_model.speech2text(turn["speech"])

        # 转换消息格式为标准格式
        new_messages = sharegpt_old2new(messages)
        # 确保角色映射正确（Qwen2.5 需要 user/assistant）
        for msg in new_messages:
            if msg["role"] in ["human", "user"]:
                msg["role"] = "user"
            elif msg["role"] in ["gpt", "assistant"]:
                msg["role"] = "assistant"

        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            new_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        # 只提取新生成的部分
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        reply = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if self.input_speech:
            if self.output_speech:
                audio = self.TTSModel.synthesize_speech(reply)
                speech = self.save_audio(audio, round_idx)
                return {"text": reply, "asr": messages[0]["value"], "speech": speech}
            else:
                return {"text": reply, "asr": messages[0]["value"], "speech": None}
        else:
            if self.output_speech:
                audio = self.TTSModel.synthesize_speech(reply)
                speech = self.save_audio(audio, round_idx)
                return {"text": reply, "speech": speech}
            else:
                return {"text": reply, "speech": None}

    def save_audio(self, audio_tensor, round_idx):
        os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
        output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
        torchaudio.save(output_path, audio_tensor.cpu(), 24000)

        return output_path


if __name__ == "__main__":
    print("="*50)
    print("测试 Qwen2.5-7B-Instruct 模型")
    print("="*50)
    
    # 初始化模型（请根据实际情况修改模型路径）
    model = Qwen2_5_Instruct(
        model_path="../weight/Qwen2.5-7B-Instruct",
        input_speech=False,
        output_speech=False
    )
    model.conv_id = 0
    
    # 测试1: 医学单选题
    print("\n测试1: 医学单选题")
    result = model.reply([
        {
            "from": "user",
            "value": "下列有关鳃弓的描述，错误的是A: 由间充质增生形成 B: 人胚第4周出现 C: 相邻鳃弓之间为鳃沟 D: 共5对鳃弓 E: 位于头部两侧\n请你直接回答正确答案选项和选项内容。"
        }
    ])
    print(f"用户: 下列有关鳃弓的描述，错误的是A: 由间充质增生形成 B: 人胚第4周出现 C: 相邻鳃弓之间为鳃沟 D: 共5对鳃弓 E: 位于头部两侧\n请你直接回答正确答案选项和选项内容。")
    print(f"模型回复: {result['text']}")
    
    # 测试2: 医学咨询问题
    print("\n" + "="*50)
    print("测试2: 医学咨询问题")
    print("="*50)
    result = model.reply([
        {
            "from": "user",
            "value": "最近我右下眼老是跳，有时候还会牵动到嘴角，这是怎么回事？"
        }
    ])
    print(f"用户: 最近我右下眼老是跳，有时候还会牵动到嘴角，这是怎么回事？")
    print(f"模型回复: {result['text']}")
    
    # 测试3: 多轮对话
    print("\n" + "="*50)
    print("测试3: 多轮对话")
    print("="*50)
    
    # 第一轮
    print("\n--- 第1轮 ---")
    messages = [
        {
            "from": "user",
            "value": "我最近经常头疼，特别是一侧，有时还会恶心。"
        }
    ]
    result1 = model.reply(messages, round_idx=0)
    print(f"用户: 我最近经常头疼，特别是一侧，有时还会恶心。")
    print(f"模型回复: {result1['text']}")
    
    # 第二轮
    print("\n--- 第2轮 ---")
    messages.append({
        "from": "assistant",
        "value": result1['text']
    })
    messages.append({
        "from": "user",
        "value": "这种情况大概持续了一个星期了，是不是偏头痛？"
    })
    result2 = model.reply(messages, round_idx=1)
    print(f"用户: 这种情况大概持续了一个星期了，是不是偏头痛？")
    print(f"模型回复: {result2['text']}")
    
    print("\n" + "="*50)
    print("测试完成！")
    print("="*50)

