import os
import json
import pandas as pd
from pathlib import Path

def process_csv_to_sharegpt(csv_dir, output_file):
    """
    处理tcsv_dir下的所有CSV文件，转换为sharegpt格式
    """
    csv_dir = Path(csv_dir)
    # 存储所有转换后的数据
    all_conversations = []
    
    # 获取所有CSV文件
    csv_files = list(csv_dir.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        print(f"处理文件: {csv_file.name}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查必要的列是否存在
            if 'harmful_medical_request' not in df.columns or 'safe_response' not in df.columns:
                print(f"警告: {csv_file.name} 缺少必要的列")
                continue
            
            # 处理每一行数据
            for index, row in df.iterrows():
                # 跳过空值
                if pd.isna(row['harmful_medical_request']) or pd.isna(row['safe_response']):
                    continue
                
                # 创建对话格式
                conversation = {
                    "idx": f"med_safety_{csv_file.stem}_{index}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": str(row['harmful_medical_request']).strip()
                        },
                        {
                            "from": "gpt", 
                            "value": str(row['safe_response']).strip()
                        }
                    ]
                }
                
                all_conversations.append(conversation)
                
        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {e}")
            continue
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共处理 {len(all_conversations)} 个对话")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    process_csv_to_sharegpt("train/gpt4", "med_safety_sharegpt-train.json")
    process_csv_to_sharegpt("test/gpt4", "med_safety_sharegpt-test.json")
