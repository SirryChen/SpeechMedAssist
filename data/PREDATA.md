
### 3.1 Filter
<details open>
<summary>filter the raw data</summary>

```bash
# -------- 1. HuatuoGPT2-GPT4-SFT-140K.json --------
python filter.py \
  --model_path ../weight/Qwen2.5-14B-Instruct/ \
  --data_path ../dataset/HuatuoGPT2-SFT/HuatuoGPT2-GPT4-SFT-140K.json \
  --selected_ratio 0.1 \
  --question_len 5 \
  --image-remove False \
  --selected_data_path ../dataset/SpeechMedDataset/selected_huatuo_pretrain_t2t.json \
  --output_path ../dataset/SpeechMedDataset/filtered_t2t.json \
  --removed_data_path ../dataset/SpeechMedDataset/removed_huatuo_pretrain_t2t.json \
  --save_steps 5000

```
</details>


### 3.2 Rewrite

<details open>
<summary>rewrite for speech featured conversations</summary>

```bash
# -------- 1. CMtMedQA-sharegpt.json --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/CMtMedQA/CMtMedQA-sharegpt.json \
  --selected_ratio 0.5 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_cmtmedqa_t2t.json \
  --output_path ../dataset/SpeechMedDataset/train_t2t_CMtMedQA.json \
  --save_steps 2000


# -------- 2. filtered_t2t.json --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/SpeechMedDataset/filtered_t2t.json \
  --selected_ratio 0.5 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_t2t.json \
  --output_path ../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2.json \
  --save_steps 2000


# -------- 3. filtered_huatuo_pretrain_t2t.json --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/SpeechMedDataset/filtered_huatuo_pretrain_t2t.json \
  --selected_ratio 0.5 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_filtered_huatuo_pretrain_t2t.json \
  --output_path ../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2_pretrain.json \
  --save_steps 2000


# -------- 4. MedDG-sharegpt.json --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/MedDG/MedDG-sharegpt.json \
  --selected_ratio 1 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_MedDG-sharegpt.json \
  --output_path ../dataset/SpeechMedDataset/train_t2t_MedDG.json \
  --save_steps 2000


# -------- 5. med_safety_sharegpt-train.json --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/med-safety-bench-datasets/med_safety_sharegpt-train.json \
  --selected_ratio 1 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_Med_Safety.json \
  --output_path ../dataset/SpeechMedDataset/train_t2t_Med_Safety.json \
  --save_steps 2000


# -------- 6. med_safety_sharegpt-test.json --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/med-safety-bench-datasets/med_safety_sharegpt-test.json \
  --selected_ratio 1 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_Med_Safety.json \
  --output_path ../dataset/SpeechMedDataset/test_t2t_Med_Safety.json \
  --save_steps 2000


# -------- 7. HuatuoGPT2_Pretrain_Medical_Encyclopedia_cn.json (train) --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/HuatuoGPT2-Pretraining-Instruction/data/HuatuoGPT2_Pretrain_Medical_Encyclopedia_cn.json \
  --selected_ratio 0.1 \
  --selected_data_path ../dataset/SpeechMedDataset/HuatuoGPT2_Pretrain_Medical_Encyclopedia.json \
  --output_path ../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2_Pretrain_Medical_Encyclopedia.json \
  --save_steps 2000


# -------- 8. HuatuoGPT2_Pretrain_Medical_Encyclopedia_cn.json (test, small ratio) --------
python rewrite.py \
  --model_path ../weight/Qwen2.5-32B-Instruct/ \
  --data_path ../dataset/HuatuoGPT2-Pretraining-Instruction/data/HuatuoGPT2_Pretrain_Medical_Encyclopedia_cn.json \
  --selected_ratio 0.0005 \
  --exist_data_path ../dataset/SpeechMedDataset/HuatuoGPT2_Pretrain_Medical_Encyclopedia.json \
  --selected_data_path ../dataset/SpeechMedDataset/test_HuatuoGPT2_Pretrain_Medical_Encyclopedia.json \
  --output_path ../dataset/SpeechMedDataset/test_t2t_Medical_Encyclopedia.json \
  --save_steps 2000

```
</details>


### 3.3 Get Patient Info

<details open>
<summary>get the info of patients for synthesizing</summary>

```bash
# -------- 1. train_t2t_CMtMedQA.json --------
python your_script.py \
  --model_path ../weight/Qwen2.5-14B-Instruct/ \
  --data_path ../dataset/SpeechMedDataset/train_t2t_CMtMedQA.json \
  --selected_ratio 1 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_t2t.json \
  --output_path ../dataset/SpeechMedDataset/annotated_CMtMedQA_t2t.json \
  --save_steps 2000


# -------- 2. train_t2t_HuatuoGPT2.json --------
python your_script.py \
  --model_path ../weight/Qwen2.5-14B-Instruct/ \
  --data_path ../dataset/SpeechMedDataset/train_t2t_HuatuoGPT2.json \
  --selected_ratio 1 \
  --selected_data_path ../dataset/SpeechMedDataset/selected_t2t.json \
  --output_path ../dataset/SpeechMedDataset/annotated_HuatuoGPT2_t2t.json \
  --save_steps 2000

```
</details>





### 3.4 Synthesize

run the function `train_t2t()` and `annotate_t2t()` in [`converge.py`](./data.converge.py)

<details open>
<summary>synthesize the speech used in train and test</summary>

You can also use the [`split_data_for_parallel.py`](./data/split_data_for_parallel.py) for parallel synthesizing.
```bash
# -------- 1. train_s2t_normal.json --------
python synthesize.py \
  --cosyvoice_path ../weight/CosyVoice2-0.5B \
  --fish_ckpt_path ../weight/openaudio-s1-mini \
  --fish_codec_ckpt ../weight/openaudio-s1-mini/codec.pth \
  --data_path ../dataset/SpeechMedDataset/annotated_t2t.json \
  --selected_data_path ../dataset/SpeechMedDataset/selected_s2t.json \
  --output_path ../dataset/SpeechMedDataset/train_s2t_normal.json \
  --wav_save_path ../dataset/SpeechMedDataset/wav/ \
  --spk_info_path ../dataset/ref_audio/spk_info.json \
  --selected_ratio 1 \
  --save_interval 2000

# -------- 2. test_s2t_Med_Safety.json --------
python synthesize.py \
  --cosyvoice_path ../weight/CosyVoice2-0.5B \
  --fish_ckpt_path ../weight/openaudio-s1-mini \
  --fish_codec_ckpt ../weight/openaudio-s1-mini/codec.pth \
  --data_path ../dataset/SpeechMedDataset/test_t2t_Med_Safety.json \
  --selected_data_path ../dataset/SpeechMedDataset/selected_s2t.json \
  --output_path ../dataset/SpeechMedDataset/test_s2t_Med_Safety.json \
  --wav_save_path ../dataset/SpeechMedDataset/test_wav/ \
  --spk_info_path ../dataset/ref_audio/Aishell-2018A-EVAL/spk_info.json \
  --selected_ratio 1 \
  --save_interval 2000

# -------- 3. train_s2t_Med_Safety.json --------
python synthesize.py \
  --cosyvoice_path ../weight/CosyVoice2-0.5B \
  --fish_ckpt_path ../weight/openaudio-s1-mini \
  --fish_codec_ckpt ../weight/openaudio-s1-mini/codec.pth \
  --data_path ../dataset/SpeechMedDataset/train_t2t_Med_Safety.json \
  --selected_data_path ../dataset/SpeechMedDataset/selected_s2t.json \
  --output_path ../dataset/SpeechMedDataset/train_s2t_Med_Safety.json \
  --wav_save_path ../dataset/SpeechMedDataset/wav/ \
  --spk_info_path ../dataset/ref_audio/spk_info.json \
  --selected_ratio 1 \
  --save_interval 2000

# -------- 4. test_s2t_Medical_Encyclopedia.json --------
python synthesize.py \
  --cosyvoice_path ../weight/CosyVoice2-0.5B \
  --fish_ckpt_path ../weight/openaudio-s1-mini \
  --fish_codec_ckpt ../weight/openaudio-s1-mini/codec.pth \
  --data_path ../dataset/SpeechMedDataset/test_t2t_Medical_Encyclopedia.json \
  --selected_data_path ../dataset/SpeechMedDataset/selected_s2t.json \
  --output_path ../dataset/SpeechMedDataset/test_s2t_Medical_Encyclopedia.json \
  --wav_save_path ../dataset/SpeechMedDataset/test_wav/ \
  --spk_info_path ../dataset/ref_audio/Aishell-2018A-EVAL/spk_info.json \
  --selected_ratio 1 \
  --save_interval 2000

```
</details>

run the function `train_s2t()` in [`converge.py`](./data.converge.py)

