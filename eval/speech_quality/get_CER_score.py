from jiwer import cer
import json
import difflib

"""
python get_CER_score.py --test_model Zhongjing
python get_CER_score.py --test_model HuatuoGPT2 --use_lcs
python get_CER_score.py --test_model Baichuan2 --use_lcs
python get_CER_score.py --test_model DISC_MedLLM --use_lcs
python get_CER_score.py --test_model SpeechGPT2
python get_CER_score.py --test_model KimiAudio
python get_CER_score.py --test_model GLM4-Voice
python get_CER_score.py --test_model Llama-Omni2-7B
python get_CER_score.py --test_model SpeechMedAssist2-final
python get_CER_score.py --test_model Qwen2-Audio
"""



def extract_ref_segment(ref: str, hyp: str) -> str:
    """
    在 ref 中找到与 hyp 最接近的片段
    使用最长公共子串匹配
    """
    seq = difflib.SequenceMatcher(None, ref, hyp)
    match = seq.find_longest_match(0, len(ref), 0, len(hyp))
    return ref[match.a: match.a + match.size] if match.size > 0 else ref[:len(hyp)]


def get_cer(audio_path, use_lcs=False):
    with open(audio_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cer_list = []
    for item in data:
        reply = item["conversations"][1]
        asr_text = reply["ASR"]
        ref_text = reply["value"]

        if len(ref_text) - len(asr_text) > 5:
            continue

        if use_lcs:
            ref_text = extract_ref_segment(ref_text, asr_text)

        cer_score = cer(asr_text, ref_text)
        cer_list.append(cer_score)

    average_score = sum(cer_list) / len(cer_list)
    print(f"{args.test_model} 平均CER：{average_score:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model", type=str, default="KimiAudio")
    parser.add_argument("--data_base_path", type=str, default="../single_round")
    parser.add_argument("--use_lcs", action="store_true", help="是否启用最长公共子串匹配")

    args = parser.parse_args()
    data_path = f"{args.data_base_path}/dialog_s2t_{args.test_model}.json"

    get_cer(data_path, args.use_lcs)

