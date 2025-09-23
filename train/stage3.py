from model.builder import create_model
import argparse
import torch
from torch.utils.data import Dataset
import json
import os
from transformers import TrainingArguments
from transformers import Trainer
import whisper
from model.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from torch.nn.utils.rnn import pad_sequence
from model.constants import IGNORE_INDEX


def collate_fn(batch):
    input_ids = [instance["input_ids"] for instance in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=151645)

    labels = [instance["labels"] for instance in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=151645)

    all_speech = [speech for instance in batch for speech in instance["speech"]]
    speech_tensors = pad_sequence(
        all_speech,
        batch_first=True,
        padding_value=0     # Whisper 训练时通常也对无声部分 pad 为 0
    )
    speech_lengths = torch.LongTensor([len(speech) for speech in all_speech])

    tgt_units = [instance["tgt_units"] for instance in batch]
    tgt_units = pad_sequence(tgt_units, batch_first=True, padding_value=IGNORE_INDEX)

    return {"input_ids": input_ids, "labels": labels, "speech": speech_tensors, "speech_lengths": speech_lengths, "tgt_units": tgt_units}


class MultiTurnSpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_config, tts_tokenizer):
        self.data_path = data_path
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.tts_tokenizer = tts_tokenizer

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_speech(self, path):
        path = os.path.join("../dataset/SpeechMedBenchMark/", path)
        speech = whisper.load_audio(path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        return speech

    def process_messages(self, messages):
        # assert len(messages) % 2 == 1, "Number of history messages must be odd"
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
        labels = input_ids.clone()
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        return input_ids, labels

    def __getitem__(self, index):
        item = self.data[index]
        messages = []
        speech_list = []
        units_list = []
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
                    "content": turn["text"],
                })
                units_list.append(turn["unit"])

        # 文本 prompt 仅用于语音模型理解上下文，不用于训练 loss
        input_ids, labels = self.process_messages(messages)

        return {
            "input_ids": input_ids,
            "speech": speech_list,
            "labels": labels,
            "tgt_units": self.tokenizer(units_list, return_tensors="pt", padding=True)["input_ids"]        # 后续会tokenizer，转为id，作为TTS LM的label
        }

    def __len__(self):
        return len(self.data)


def train_model(args):
    tokenizer, model, context_len = create_model(args.model_path, s2s=args.s2s, args=args)

    dataset = MultiTurnSpeechDataset(args.data_path, tokenizer, model.config, model.speech_generator.tokenizer)
    # 初始化Trainer
    training_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=6,
        learning_rate=5e-5,
        weight_decay=0.01,
        adam_beta2=0.95,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        report_to=None,
        num_train_epochs=1,
        save_steps=5000,
        save_total_limit=1,
        seed=2025,
        bf16=True
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../weight/stage2/")
    parser.add_argument("--output-path", type=str, default="../weight/stage3/")
    parser.add_argument("--data-path", type=str, default="../data/SpeechMedBenchmark/train_s2s.json")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--s2s", type=bool, default=True)
    parser.add_argument("--speech_encoder_type", type=str, default="whisper")
    parser.add_argument("--speech_encoder", type=str, default="../weight/whisper/large-v3.pt")
    parser.add_argument("--tune_speech_generator_only", type=bool, default=True)
    train_model(parser.parse_args())
