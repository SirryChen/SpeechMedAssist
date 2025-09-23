import json

file_list = ["./Aishell3/spk_info.json", "./Aishell2/iOS/spk_info.json"]

with open(file_list[0], "r", encoding='utf-8') as f:
    final_data = json.load(f)

for file in file_list[1:]:
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    for gender in final_data.keys():
        for age in final_data[gender].keys():
            if data[gender].get(age):
                final_data[gender][age].update(data[gender][age])

with open("./spk_info.json", "w", encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
