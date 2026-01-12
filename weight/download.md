Here we provide the download command for the datasets used in our work.

```shell

huggingface-cli download --resume-download ICTNLP/LLaMA-Omni2-7B-Bilingual --local-dir ./weight/LLaMA-Omni2-7B-Bilingual
huggingface-cli download --resume-download ICTNLP/cosy2_decoder --local-dir ./weight/cosy2_decoder


git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git ./weight/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git ./weight/CosyVoice-ttsfrd
cd ./weight/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl 

huggingface-cli download --resume-download fishaudio/openaudio-s1-mini --local-dir ./weight/openaudio-s1-mini

huggingface-cli download --resume-download Qwen/Qwen2-Audio-7B-Instruct --local-dir ./weight/Qwen2-Audio-7B-Instruct

huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir ./weight/Qwen2.5-32B-Instruct
huggingface-cli download --resume-download Qwen/Qwen2.5-14B-Instruct --local-dir ./weight/Qwen2.5-32B-Instruct
huggingface-cli download --resume-download Qwen/Qwen2.5-32B-Instruct --local-dir ./weight/Qwen2.5-32B-Instruct
huggingface-cli download --resume-download Qwen/Qwen2.5-72B-Instruct --local-dir ./weight/Qwen2.5-32B-Instruct

huggingface-cli download --resume-download zai-org/glm-4-voice-9b --local-dir ./weight/GLM4-Voice/glm-4-voice-9b
huggingface-cli download --resume-download zai-org/glm-4-voice-tokenizer --local-dir ./weight/GLM4-Voice/glm-4-voice-tokenizer
cd ./weight/GLM4-Voice/; git clone https://huggingface.co/THUDM/glm-4-voice-decoder

huggingface-cli download --resume-download FunAudioLLM/SenseVoiceSmall --local-dir ./weight/SenseVoiceSmall

huggingface-cli download --resume-download moonshotai/Kimi-Audio-7B-Instruct --local-dir ./weight/Kimi-Audio-7B-Instruct

huggingface-cli download --resume-download openai/whisper-large-v3 --local-dir ./weight/whisper-large-v3

huggingface-cli download --resume-download FreedomIntelligence/ShizhenGPT-7B-Omni --local-dir ./weight/ShizhenGPT-7B-Omni

huggingface-cli download --resume-download BioMistral/BioMistral-7B --local-dir ./weight/BioMistral-7B

huggingface-cli download --resume-download baichuan-inc/Baichuan-Audio-Instruct --local-dir ./weight/Baichuan-Audio-Instruct

huggingface-cli download --resume-download Flmc/DISC-MedLLM --local-dir ./weight/DISC-MedLLM

huggingface-cli download --resume-download fnlp/SpeechGPT-2.0-preview-7B --local-dir ./weight/SpeechGPT-2.0-preview
huggingface-cli download --resume-download fnlp/SpeechGPT-2.0-preview-Codec --local-dir ./weight/SpeechGPT-2.0-preview-Codec

```