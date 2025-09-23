import json
import os
from collections import defaultdict

spk_info_txt_path = 'spk-info.txt'
data_dirs = ['train', 'test']
wav_base_dir = './dataset/ref_audio/Aishell3/'

# 读取文本内容
spk_file_content_dict = defaultdict(list)

for data_dir in data_dirs:
    wav_dir = os.path.join(data_dir, 'wav')
    text_path = os.path.join(data_dir, 'content.txt')

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            filename, text = parts
            text_chinese_only = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff'])
            spk_id = filename[:7]
            wav_path = os.path.join(wav_base_dir, data_dir, 'wav', spk_id, filename)
            spk_file_content_dict[spk_id].append([wav_path, text_chinese_only])

# 年龄组字母到中文映射
age_group_map = {
    "A": "少年",
    "B": "青年",
    "C": "成年",
    "D": "老年"
}

# 构建最终嵌套结构（含性别/年龄段/说话人/语音路径和文本）
result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

with open(spk_info_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            spk_id, age_group, gender, accent = parts
            if spk_id not in spk_file_content_dict:
                continue
            age_group_chinese = age_group_map.get(age_group, age_group)  # 替换为中文
            gender_label = '女' if gender == 'female' else '男'
            result[gender_label][age_group_chinese][spk_id].extend(spk_file_content_dict[spk_id])

# 构建 "未知" 字段：合并男女的说话人
result["未知"] = defaultdict(dict)
all_age_groups = set(result["男"].keys()) | set(result["女"].keys())
for age_group in all_age_groups:
    male_speakers = result["男"].get(age_group, {})
    female_speakers = result["女"].get(age_group, {})
    combined = {**male_speakers, **female_speakers}
    result["未知"][age_group] = combined

# 保存为 JSON 文件
with open("spk_info.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
