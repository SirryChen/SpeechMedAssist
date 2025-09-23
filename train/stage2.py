from model.builder import create_model
import argparse
import torch
from torch.utils.data import Dataset
import json
from transformers import TrainingArguments
from transformers import Trainer
import whisper
import random
from model.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SYSTEM_PROMPT
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    input_ids = [instance["input_ids"] for instance in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=151645)

    labels = [instance["labels"] for instance in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=151645)

    if batch[0].get("speech") is None:
        return {"input_ids": input_ids, "labels": labels}

    all_speech = [speech for instance in batch for speech in instance["speech"]]

    speech_tensors = pad_sequence(
        all_speech,
        batch_first=True,
        padding_value=0     # Whisper 训练时通常也对无声部分 pad 为 0
    )
    speech_lengths = torch.LongTensor([len(speech) for speech in all_speech])
    return {"input_ids": input_ids, "labels": labels, "speech": speech_tensors, "speech_lengths": speech_lengths}


class MultiTurnSpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_config, assist_data_path, assist_ratio, select_ratio):
        self.data_path = data_path
        self.assist_data_path = assist_data_path
        self.assist_ratio = assist_ratio
        self.select_ratio = select_ratio
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.model_config = model_config

        self.system_ids = self.tokenizer("<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        self.user_prefix_ids = self.tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        self.assistant_prefix_ids = self.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.end_ids = self.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"语音数据：{len(data)}")

        with open(self.assist_data_path, "r", encoding="utf-8") as f:
            assist_data = json.load(f)
            assist_data = random.sample(assist_data, int(len(assist_data) * self.assist_ratio))
            print(f"辅助数据：{len(assist_data)}")

        # 合并
        all_data = data + assist_data
        if self.select_ratio < 1:
            total_len = len(all_data)
            selected = random.sample(all_data, int(total_len * self.select_ratio))
            all_data = (selected * (total_len // len(selected) + 1))[:total_len]

        return all_data

    def load_speech(self, path):
        speech = whisper.load_audio(path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        return speech

    def process_messages(self, messages):
        input_ids_list = []
        labels_list = []
        input_ids_list.extend(self.system_ids)
        labels_list.extend([-100] * len(self.system_ids))

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # 手动构建模板：<|im_start|>role\ncontent<|im_end|>\n
            role_prefix_ids = self.user_prefix_ids if role != "assistant" else self.assistant_prefix_ids
            content_ids = self.tokenizer(content, add_special_tokens=False)["input_ids"]

            # 拼接整个片段
            input_ids_list.extend(role_prefix_ids + content_ids + self.end_ids)

            # label 设置：user 段全为 -100，assistant 段仅内容部分为 label
            if role == "assistant":
                labels_list.extend([-100] * len(role_prefix_ids))  # prefix
                labels_list.extend(content_ids + self.end_ids)  # keep assistant reply
            else:
                labels_list.extend([-100] * (len(role_prefix_ids) + len(content_ids) + len(self.end_ids)))

        # 转成 tensor
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)

        # 替换语音 token
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX

        return input_ids, labels

    def __getitem__(self, index):
        item = self.data[index]
        messages = []
        if item["conversations"][0].get("speech") is None:
            for i, turn in enumerate(item["conversations"]):
                if i % 2 == 0:
                    messages.append({
                        "role": "user",
                        "content": turn["value"],
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": turn["value"],
                    })
            input_ids, labels = self.process_messages(messages)
            return {
                "input_ids": input_ids,
                "labels": labels
            }
        else:
            speech_list = []
            for i, turn in enumerate(item["conversations"]):
                if i % 2 == 0:
                    messages.append({
                        "role": "user",
                        "content": DEFAULT_SPEECH_TOKEN,
                    })
                    speech_list.append(self.load_speech(turn["speech"]))
                else:
                    messages.append({
                        "role": "assistant",
                        "content": turn["value"],
                    })
            input_ids, labels = self.process_messages(messages)
            return {
                "input_ids": input_ids,
                "labels": labels,
                "speech": speech_list
            }

    def __len__(self):
        return len(self.data)


def train_model(args):
    tokenizer, model, context_len = create_model(args.model_path, s2s=args.s2s, args=args)

    dataset = MultiTurnSpeechDataset(args.data_path, tokenizer, model.config, args.assist_data_path, args.assist_ratio, args.select_ratio)
    # 初始化Trainer
    training_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        adam_beta2=0.999,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',
        logging_steps=10,
        logging_dir=f"../log/stage2",
        report_to="tensorboard",
        num_train_epochs=1,
        save_steps=5000,
        seed=2025,
        bf16=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )

    trainer.train()

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../weight/stage1/")
    parser.add_argument("--output-path", type=str, default="../weight/stage2/")
    parser.add_argument("--local_rank", type=int, default=-1, help="Used for distributed training")
    parser.add_argument("--data-path", type=str, default="../dataset/SpeechMedDataset/train_s2t.json")
    parser.add_argument("--assist-data-path", type=str, default="../dataset/CMB/CMB-train-sharegpt.json")
    parser.add_argument("--assist_ratio", type=float, default=1)
    parser.add_argument("--select_ratio", type=float, default=1)
    parser.add_argument("--epoch", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--t2t", type=bool, default=False)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--speech_encoder_type", type=str, default="whisper")
    parser.add_argument("--speech_encoder", type=str, default="../weight/whisper/large-v3.pt")
    train_model(parser.parse_args())
