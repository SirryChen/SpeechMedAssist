# use env KimiAudio
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, "../../Kimi-Audio")
sys.path.append(project_path)

import soundfile as sf
from kimia_infer.api.kimia import KimiAudio as KimiAudioModel


def sharegpt2KimiAudio(messages):
    new_messages = []

    for item in messages:

        role = "user" if item["from"] in ["human", "user"] else "assistant"
        if role == "user":
            new_messages.append({"role": role, "message_type": "audio", "content": item["speech"]})
        else:
            new_messages.append({"role": "assistant", "message_type": "text", "content": item["value"]})

    return new_messages


# KimiAudio only support S2T and S2S according to https://github.com/MoonshotAI/Kimi-Audio/tree/master#quick-start
class KimiAudio:
    def __init__(self, model_path, input_speech=True, output_speech=False, speech_output_dir="KimiAudio_output"):
        self.input_speech = input_speech
        self.output_speech = output_speech
        self.speech_output_dir = speech_output_dir
        self.conv_id = None
        if output_speech:
            os.makedirs(speech_output_dir, exist_ok=True)

        self.model = KimiAudioModel(model_path=model_path, load_detokenizer=True)
        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

        self.round_idx = 0


    def reply(self, messages, round_idx=0):
        new_messages = sharegpt2KimiAudio(messages)

        # S2T
        if not self.output_speech:
            _, reply = self.model.generate(new_messages, **self.sampling_params, output_type="text")
            speech = None
        else:
            os.makedirs(os.path.join(self.speech_output_dir, str(self.conv_id)), exist_ok=True)
            output_path = os.path.join(self.speech_output_dir, str(self.conv_id), f"doctor_{round_idx}.wav")
            wav, reply = self.model.generate(new_messages, **self.sampling_params, output_type="both")
            sf.write(output_path, wav.detach().cpu().view(-1).numpy(), 24000)
            speech = output_path

        return {"text": reply, "speech": speech}


if __name__ == "__main__":
    # Fix the path in "Kimi-Audio/kimia_infer/api/prompt_manager.py" line 17
    model = KimiAudio(model_path="../weight/Kimi-Audio-7B-Instruct", output_speech=True)
    model.conv_id = 0
    print(model.reply([
            {
                "from": "user",
                "value": "",
                "speech": "./test2.mp3"
            }
    ]))