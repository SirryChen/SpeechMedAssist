from model.builder import create_model
import argparse
import torch
from torch.utils.data import Dataset
import json
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Trainer,
    TrainingArguments
)
from model.constants import SYSTEM_PROMPT


def collate_fn(batch):
    input_ids = [instance["input_ids"] for instance in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=151645)

    labels = [instance["labels"] for instance in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=151645)

    return {"input_ids": input_ids, "labels": labels}


class MultiTurnSpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_config):
        self.data_path = data_path
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
        return data

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


        return input_ids, labels

    def __getitem__(self, index):
        item = self.data[index]
        messages = []
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

    def __len__(self):
        return len(self.data)


def train_model(args):

    tokenizer, model, context_len = create_model(args.model_path, s2s=args.s2s, args=args)

    tokenizer.chat_template = tokenizer.chat_template.replace(
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        SYSTEM_PROMPT
    )

    dataset = MultiTurnSpeechDataset(args.data_path, tokenizer, model.config)
    # 初始化Trainer
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        logging_dir="../log/stage1-embedding",
        logging_steps=10,
        lr_scheduler_type='cosine',
        save_steps=500,
        report_to="tensorboard",
        load_best_model_at_end=False,
        bf16=True,
        weight_decay=0.01,
        warmup_ratio=0.1,
        learning_rate=5e-5,
        adam_beta2=0.95,
        save_total_limit=2,
        seed=2025,
        ddp_find_unused_parameters=True,
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
    parser.add_argument("--model-path", type=str, default="../weight/LLaMA-Omni2-7B-Bilingual")
    parser.add_argument("--output-path", type=str, default="../weight/stage1-embedding/")
    parser.add_argument("--local_rank", type=int, default=-1, help="Used for distributed training")
    parser.add_argument("--data-path", type=str, default="../dataset/SpeechMedDataset/train_t2t.json")
    parser.add_argument("--t2t", type=bool, default=True)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--speech_encoder_type", type=str, default="whisper")
    parser.add_argument("--speech_encoder", type=str, default="../weight/whisper/large-v3.pt")
    train_model(parser.parse_args())
