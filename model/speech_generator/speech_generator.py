import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from model.constants import IGNORE_INDEX


def lengths_to_attention_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return ~mask


class LLMSpeechGenerator(nn.Module):
    def __init__(self, config):
        super(LLMSpeechGenerator, self).__init__()
        self.model = Qwen2ForCausalLM(Qwen2Config(**config.speech_generator))
        self.tokenizer = AutoTokenizer.from_pretrained(config.tts_tokenizer)
        self.input_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, self.model.config.hidden_size)
        )
        self.stream_params = config.stream_params
        self.gate = nn.Sequential(
            nn.Linear(2 * self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Sigmoid()
        )

    def fusion(self, rep, emb):
        gate = self.gate(torch.cat([rep, emb], dim=-1))
        return rep * gate + emb * (1 - gate)

    def forward(self, hidden_states, text_tokens, tgt_units):
        B = hidden_states.size(0)

        # Step 1: embedding & fusion
        hidden_states_proj = self.input_proj(hidden_states)  # (B, N1, D_model)
        text_tokens[text_tokens == IGNORE_INDEX] = self.tokenizer.pad_token_id
        text_emb = self.model.get_input_embeddings()(text_tokens)  # (B, N2, D_model)  use text labels since using teacher forcing
        fused_rep = self.fusion(hidden_states_proj, text_emb)  # (B, N2, D_model)

        tgt_emb = self.model.get_input_embeddings()(tgt_units)  # (B, N3, D_model)

        # Step 2: Interleave fused and tgt_emb according to R:W
        r, w = eval(self.stream_params)
        input_embeddings = []
        labels = []

        for b in range(B):
            fused_emb = fused_rep[b]  # shape: (N1, D)
            speech_emb = tgt_emb[b][0]  # shape: (N3, D)
            label_ids = tgt_units[b][0]  # shape: (N3,)

            fused_ptr, speech_ptr = 0, 0
            input_seq = []
            label_seq = []

            while fused_ptr < fused_emb.size(0):
                # Add R reps
                for _ in range(r):
                    if fused_ptr < fused_emb.size(0):
                        input_seq.append(fused_emb[fused_ptr])
                        label_seq.append(IGNORE_INDEX)
                        fused_ptr += 1
                        if fused_ptr == fused_emb.size(0):  # 确保在Read完后，继续Write时label能错开一位
                            label_seq = label_seq[1:].append(IGNORE_INDEX)

                # Add W target units
                for _ in range(w):
                    if speech_ptr < speech_emb.size(0):
                        input_seq.append(speech_emb[speech_ptr])
                        label_seq.append(label_ids[speech_ptr].item())
                        speech_ptr += 1

            while speech_ptr < speech_emb.size(0):
                input_seq.append(speech_emb[speech_ptr])
                label_seq.append(label_ids[speech_ptr].item())
                speech_ptr += 1
            label_seq = label_seq[1:].append(IGNORE_INDEX)

            # 转换为Tensor
            input_seq = torch.stack(input_seq, dim=0)  # (L, D)
            label_seq = torch.tensor(label_seq, dtype=torch.long, device=hidden_states.device)  # (L,)

            input_embeddings.append(input_seq)
            labels.append(label_seq)

        # Step 3: pad all sequences
        input_embeddings = torch.nn.utils.rnn.pad_sequence(input_embeddings, batch_first=True)  # (B, L_max, D)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)  # (B, L_max)

        # Step 4: forward to decoder
        outputs = self.model(
            inputs_embeds=input_embeddings,
            labels=labels,
        )
        return outputs.loss, outputs

    def generate_units(self, tts_inputs, new_hidden_states, new_tokens, is_finished=False):
        # only for batch size = 1
        new_hidden_states = self.input_proj(new_hidden_states)
        new_token_embeddings = self.model.get_input_embeddings()(new_tokens)
        new_hidden_states = self.fusion(new_hidden_states, new_token_embeddings)
        if tts_inputs is not None:
            tts_inputs = torch.cat([tts_inputs, new_hidden_states], dim=0)
        else:
            tts_inputs = new_hidden_states
        if is_finished:
            device = tts_inputs.device
            sep_id = torch.LongTensor([self.tokenizer.convert_tokens_to_ids("<sep>")]).to(device)
            sep_emb = self.model.get_input_embeddings()(sep_id)
            tts_inputs = torch.cat([tts_inputs, sep_emb], dim=0)

        _, M = eval(self.stream_params)
        max_new_tokens = M if not is_finished else 1024
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=tts_inputs.unsqueeze(0),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0]
        generated_units = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_tokens_embeds = self.model.get_input_embeddings()(generated_tokens)
        tts_inputs = torch.cat([tts_inputs, generated_tokens_embeds], dim=0)
        return tts_inputs, generated_units