import sys
sys.path.append('../FishSpeech')  # 添加 Fish-speech 路径
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import torchaudio

from fish_speech.models.text2semantic.inference import init_model, generate_long
from fish_speech.models.dac.inference import load_model


def synthesize_with_fish_speech(text, model, decode_one_token, codec_model, device):
    # === 使用 generate_long 生成语义代码 ===
    torch.manual_seed(2025)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,
        top_p=0.8,
        repetition_penalty=1.1,
        temperature=0.8,
        compile=False,
        iterative_prompt=True,
        chunk_length=300,
        prompt_text=[],
        prompt_tokens=[],
    )

    codes = []
    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
        elif response.action == "next":
            break

    if not codes:
        raise RuntimeError("未生成任何语义编码！")

    codes_tensor = torch.cat(codes, dim=1).to(device).long()

    # === 使用 codec_model 解码音频 ===
    indices_lens = torch.tensor([codes_tensor.shape[1]], device=device, dtype=torch.long)
    fake_audios, _ = codec_model.decode(codes_tensor, indices_lens)
    waveform = fake_audios[0, 0].detach()

    waveform_resampled = torchaudio.functional.resample(
        waveform.unsqueeze(0), orig_freq=codec_model.sample_rate, new_freq=16000
    )

    return waveform_resampled


def main():
    text = "我的思路是航天一次焰火一次飞升。然后长弓蹲铁穹，有人接就让他做。趴个15分钟，没人接就跑过去接了做。"  # 输入文本
    output_wav_path = "./embedding_speech5.wav"  # 输出路径

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.bfloat16

    # ==== 路径配置（请根据本地修改） ====
    model_ckpt_path = "../weight/openaudio-s1-mini"
    codec_ckpt_path = "../weight/openaudio-s1-mini/codec.pth"

    # ==== 加载语义生成模型 ====
    model, decode_one_token = init_model(
        checkpoint_path=model_ckpt_path,
        device=device,
        precision=precision,
        compile=False
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype
        )

    # ==== 加载语音解码模型 ====
    codec_model = load_model(
        config_name="modded_dac_vq",
        checkpoint_path=codec_ckpt_path,
        device=device
    )

    # ==== 合成语音 ====
    waveform = synthesize_with_fish_speech(
        text=text,
        model=model,
        decode_one_token=decode_one_token,
        codec_model=codec_model,
        device=device
    )

    # ==== 保存语音 ====
    torchaudio.save(output_wav_path, waveform.cpu(), sample_rate=16000)
    # sf.write(output_wav_path, waveform, samplerate=sample_rate)
    print(f"✅ 语音已保存到 {output_wav_path}")


if __name__ == "__main__":
    main()
