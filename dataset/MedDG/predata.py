import json
import re
from tqdm import tqdm

def process_meddg_to_sharegpt(input_file, output_file, start_id=0):
    """
    将MedDG格式的对话数据转换为sharegpt格式
    
    Args:
        input_file: 输入的txt文件路径
        output_file: 输出的sharegpt格式文件路径
        start_id: 对话ID的起始值
    """
    conversations = []
    current_dialog = []
    dialog_id = start_id
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 使用tqdm包装range来显示进度条
    for i in tqdm(range(len(lines)), desc=f"处理 {input_file}"):
        line = lines[i].strip()
        
        # 跳过空行
        if not line:
            continue
        
        # 检查是否是新的对话开始
        if line.startswith('dialog'):
            # 如果之前有对话内容，保存它
            if current_dialog:
                match = re.search(r'（(男|女)[，,](\d+)(岁)?）', current_dialog[0]["value"])
                if match:
                    gender = match.group(1)
                    age = int(match.group(2))
                else:
                    gender = None
                    age = None
                conversations.append({
                    "idx": f"MedDG-{dialog_id}",
                    "age": age,
                    "gender": gender,
                    "conversations": current_dialog
                })
                dialog_id += 1
                current_dialog = []
            
            continue
        
        # 处理对话行
        if line.startswith('{'):
            try:
                # 解析JSON行
                data = json.loads(line)
                speaker = data.get('id', '')
                sentence = data.get('Sentence', '')
                
                if speaker and sentence:
                    # 将Patients映射为user，Doctor映射为gpt
                    role = 'user' if speaker == 'Patients' else 'gpt'
                    
                    current_dialog.append({
                        "from": role,
                        "value": sentence
                    })
            except json.JSONDecodeError:
                print(f"无法解析JSON行: {line}")
    
    # 添加最后一个对话
    if current_dialog:
        conversations.append({
            "idx": f"MedDG-{dialog_id}",
            "conversations": current_dialog
        })
    
    return conversations

def merge_and_save(train_conversations, dev_conversations, output_file):
    """
    合并训练集和验证集的对话并保存
    
    Args:
        train_conversations: 训练集对话列表
        dev_conversations: 验证集对话列表
        output_file: 输出文件路径
    """
    # 合并所有对话
    all_conversations = train_conversations + dev_conversations
    
    # 保存为sharegpt格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成！")
    print(f"训练集对话数: {len(train_conversations)}")
    print(f"验证集对话数: {len(dev_conversations)}")
    print(f"总对话数: {len(all_conversations)}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # 处理训练集
    print("处理训练集...")
    train_conversations = process_meddg_to_sharegpt("train.txt", "MedDG-sharegpt-train.json", start_id=0)
    
    # 处理验证集
    print("处理验证集...")
    dev_conversations = process_meddg_to_sharegpt("dev.txt", "MedDG-sharegpt-dev.json", start_id=len(train_conversations))
    
    # 合并并保存
    print("合并数据集...")
    merge_and_save(train_conversations, dev_conversations, "MedDG-sharegpt.json")

    print("处理测试集...")
    test_conversations = process_meddg_to_sharegpt("test.txt", "MedDG-sharegpt-test.json", start_id=0)
    with open("MedDG-sharegpt-test.json", 'w', encoding='utf-8') as f:
        json.dump(test_conversations, f, ensure_ascii=False, indent=4)
