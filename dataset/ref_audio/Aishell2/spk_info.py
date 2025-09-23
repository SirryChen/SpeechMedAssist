import json
import os
from collections import defaultdict
import torchaudio
from tqdm import tqdm

# 数据根目录
data_dir = 'data'
trans_path = os.path.join(data_dir, 'trans.txt')
spk_info_path = os.path.join(data_dir, 'spk_info.txt')
wav_base_dir = os.path.join("./dataset/ref_audio/Aishell2/iOS/", data_dir, 'wav')

# 年龄组字母到中文映射
age_group_map = {
    "A": "少年",
    "B": "青年",
    "C": "成年",
    "D": "老年"
}

# 读取文本内容，构建 spk_file_content_dict：spk_id -> List[[wav_path, 中文文本]]
spk_file_content_dict = defaultdict(list)
invalid_wav_files = []

with open(trans_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        filename, text = parts
        text_chinese_only = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff'])
        spk_id = filename[1:6]
        wav_path = os.path.join(data_dir, "wav", spk_id, filename + '.wav')

        # 检查文件是否存在且可读
        if not os.path.exists(wav_path):
            print(f"❌ 缺失文件: {wav_path}")
            continue

        try:
            # 用 torchaudio 检查是否能正常加载
            torchaudio.load(wav_path)
            spk_file_content_dict[spk_id].append([os.path.join(wav_base_dir, spk_id, filename + '.wav'), text_chinese_only])
        except Exception as e:
            print(f"❌ 文件损坏: {wav_path} ({e})")
            invalid_wav_files.append(wav_path)
            # 可选择是否删除损坏的音频

print("构建最终嵌套结构")
result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

with open(spk_info_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            spk_id, age_group, gender, accent = parts
            if spk_id not in spk_file_content_dict:
                continue
            age_group_chinese = age_group_map.get(age_group, age_group)
            gender_label = '女' if gender == 'female' else '男'
            result[gender_label][age_group_chinese][spk_id].extend(spk_file_content_dict[spk_id])

# 添加“未知”字段：性别合并
result["未知"] = defaultdict(dict)
all_age_groups = set(result["男"].keys()) | set(result["女"].keys())
for age_group in all_age_groups:
    male_speakers = result["男"].get(age_group, {})
    female_speakers = result["女"].get(age_group, {})
    combined = {**male_speakers, **female_speakers}
    result["未知"][age_group] = combined

# 保存为 JSON
with open("spk_info.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"✅ spk_info.json 已生成。跳过或删除了 {len(invalid_wav_files)} 个损坏的音频文件。")
