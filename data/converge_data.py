import os
import json
import re


def save(data_path_list, output_path):
    converged_data = []

    for data_path in data_path_list:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "s2t" in data_path:
            data = replace2(data)
        elif "t2t" in data_path:
            if "Encyclopedia" in data_path:
                data = replace(data)

        print(f"{os.path.basename(data_path)}: {len(data)}")
        converged_data += data


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converged_data, f, ensure_ascii=False, indent=4)

def replace(data):
    for item in data:
        item["conversations"][1]["value"].replace("~", "至").replace("-", "至").replace("mg", "毫克").replace("g", "克").replace("ml", "毫升")
        if "%" in item["conversations"][1]["value"]:
            item["conversations"][1]["value"] = re.sub(r'(\d+(\.\d+)?)%', r'百分之\1', item["conversations"][1]["value"])
        if len(item["conversations"]) > 2:
            item["conversations"] = item["conversations"][:2]
    return data

def replace2(data):
    for item in data:
        for turn in item["conversations"]:
            if turn["from"] == "assistant":
                turn["value"] = turn["value"].replace("~", "至").replace("-", "至").replace("mg", "毫克").replace("g", "克").replace("ml", "毫升")
                if "%" in turn["value"]:
                    turn["value"] = re.sub(r'(\d+(\.\d+)?)%', r'百分之\1', turn["value"])
    return data

def train_t2t():
    data_path_list = [
        "../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2.json",
        "../dataset/SpeechMedDataset/train_t2t_CMtMedQA.json",
        "../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2_pretrain.json",
        "../dataset/SpeechMedDataset/train_t2t_Med_Safety.json",
        "../dataset/SpeechMedDataset/train_t2t_MedDG.json",
        "../dataset/CMB/CMB-train-sharegpt.json",
        "../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2_Pretrain_Meidcal_Encyclopedia.json"
    ]
    output_path = "../dataset/SpeechMedDataset/train_t2t.json"

    save(data_path_list, output_path)

def annotate_t2t():
    data_path_list = [
        "../dataset/SpeechMedDataset/annotated_HuatuoGPT2_t2t.json",
        "../dataset/SpeechMedDataset/annotated_CMtMedQA_t2t.json",
        "../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2_pretrain.json"
    ]
    output_path = "../dataset/SpeechMedDataset/annotated_t2t.json"

    save(data_path_list, output_path)


def train_s2t():
    data_path_list = [
        "../dataset/SpeechMedDataset/train_s2t_normal.json",
        "../dataset/SpeechMedDataset/train_s2t_Encyclopedia.json"
        "../dataset/SpeechMedDataset/train_s2t_MedSafety.json"
    ]
    output_path = "../dataset/SpeechMedDataset/train_s2t.json"

    save(data_path_list, output_path)

if __name__ == "__main__":

    train_t2t()

    annotate_t2t()

    # train_s2t()