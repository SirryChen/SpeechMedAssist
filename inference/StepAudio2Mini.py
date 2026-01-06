# use env StepAudio2
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../Step-Audio2")
project_path = os.path.abspath(project_path)

# 规范化路径以便比较
def normalize_path(path):
    """规范化路径，处理空字符串和相对路径"""
    if not path:
        return None
    try:
        return os.path.normpath(os.path.abspath(path))
    except (OSError, ValueError):
        return None

current_dir_norm = normalize_path(current_dir)
project_path_norm = normalize_path(project_path)

# 从 sys.path 中移除 inference 目录，避免导入冲突
removed_paths = []
if current_dir_norm:
    for i in range(len(sys.path) - 1, -1, -1):
        path_norm = normalize_path(sys.path[i])
        if path_norm == current_dir_norm:
            removed_paths.append(sys.path.pop(i))

# 将 Step-Audio2 路径插入到最前面
if project_path_norm and project_path_norm not in [normalize_path(p) for p in sys.path]:
    sys.path.insert(0, project_path_norm)

try:
    from stepaudio2 import StepAudio2
finally:
    # 确保 Step-Audio2 路径在最前面
    if project_path_norm and (normalize_path(sys.path[0]) if sys.path else None) != project_path_norm:
        if project_path_norm in sys.path:
            sys.path.remove(project_path_norm)
        sys.path.insert(0, project_path_norm)
    # 将被移除的路径添加回去
    for path in removed_paths:
        if path not in sys.path:
            sys.path.append(path)


def sharegpt2StepAudio2(messages):
    """将 sharegpt 格式的消息转换为 StepAudio2 格式"""
    new_messages = []
    
    for item in messages:
        if item.get("from") == "system":
            if "value" in item and item["value"]:
                new_messages.append({"role": "system", "content": item["value"]})
            continue
        
        if item["from"] in ["human", "user"]:
            if item.get("speech") is not None and item["speech"]:
                new_messages.append({
                    "role": "human",
                    "content": [{"type": "audio", "audio": item["speech"]}]
                })
            elif "value" in item and item["value"]:
                new_messages.append({"role": "human", "content": item["value"]})
        else:
            if "value" in item and item["value"]:
                new_messages.append({"role": "assistant", "content": item["value"]})
    
    return new_messages


class StepAudio2Mini:
    def __init__(self, model_path, input_speech=True, output_speech=False, speech_output_dir="StepAudio2Mini_output"):
        """
        初始化 StepAudio2-mini 模型
        
        Args:
            model_path: 模型路径或模型名称
            input_speech: 是否支持语音输入
            output_speech: 是否支持语音输出（目前仅支持文本输出）
            speech_output_dir: 语音输出目录（未使用，保留接口兼容性）
        """
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None
        
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)
        
        self.model = StepAudio2(model_path)
        self.sampling_params = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
        }
        
        self.round_idx = 0

    def reply(self, messages, round_idx=0):
        """
        生成回复
        
        Args:
            messages: sharegpt 格式的消息列表
            round_idx: 当前轮次索引
            
        Returns:
            {"text": reply_text, "speech": None}
        """
        new_messages = sharegpt2StepAudio2(messages)
        
        # 添加 assistant 占位符
        if not new_messages or new_messages[-1].get("role") != "assistant":
            new_messages.append({"role": "assistant", "content": None})
        
        # 调用模型生成文本
        tokens, text, _ = self.model(new_messages, **self.sampling_params)
        
        return {"text": text, "speech": None}


if __name__ == "__main__":
    # 测试代码
    # model = StepAudio2Mini(model_path='../weight/Step-Audio-2-mini', output_speech=False)
    # model.conv_id = 0
    # print(model.reply([
    #     {
    #         "from": "user",
    #         "value": "",
    #         "speech": "./test2.mp3"
    #     }
    # ]))

    model = StepAudio2Mini(model_path='../weight/Step-Audio-2-mini', input_speech=False, output_speech=False)
    model.conv_id = 0
    print(model.reply([
        {
            "from": "user",
            "value": "医生，我有个单项选择题想考你一下。\n下列有关鳃弓的描述，错误的是A: 由间充质增生形成 B: 人胚第4周出现 C: 相邻鳃弓之间为鳃沟 D: 共5对鳃弓 E: 位于头部两侧\n请你直接回答正确答案选项和选项内容。"
        }
    ]))


