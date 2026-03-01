import sys
import types
import os
import torchaudio

try:
    import pkg_resources as _pr
    _pr.get_distribution("setuptools")
except (ImportError, AttributeError):
    _stub = types.ModuleType("pkg_resources")
    _stub.declare_namespace = lambda name: None
    try:
        from importlib.metadata import distribution as _dist
        _stub.get_distribution = _dist
    except ImportError:
        _dummy = type("_D", (), {"version": "0.0.0"})()
        _stub.get_distribution = lambda name: _dummy
    sys.modules["pkg_resources"] = _stub

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

#AUDIO_PATH   = 'dataset/english_free_speech/files_cut_by_sentences/01_M_native/a boy looking at the frog.wav'
AUDIO_PATH =  'asset/zero_shot_prompt.wav'
TRANSCRIPTION = 'a boy looking at the frog'
OUTPUT_DIR   = 'audio_generate_zeroshot'
N_SAMPLES    = 8   # 生成样本数量

# CosyVoice3 zero-shot 的 prompt_text 格式：system prompt + <|endofprompt|> + 参考音频文字
PROMPT_TEXT = f'You are a helpful assistant.<|endofprompt|>{TRANSCRIPTION}'

os.makedirs(OUTPUT_DIR, exist_ok=True)

cosyvoice = AutoModel(
    model_dir='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    load_trt=True, load_vllm=False, fp16=False
)

for i in range(N_SAMPLES):
    set_all_random_seed(i * 7)   # 不同 seed → 相同音色下的韵律/节奏自然变体
    chunks = []
    for output in cosyvoice.inference_zero_shot(
        TRANSCRIPTION,
        PROMPT_TEXT,
        AUDIO_PATH,
        stream=False,
    ):
        chunks.append(output['tts_speech'])

    import torch
    audio = torch.cat(chunks, dim=1)
    out_path = os.path.join(OUTPUT_DIR, f'sample_{i:02d}.wav')
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)
    print(f'[sample_{i:02d}] saved -> {out_path}')

print('Done. All audio saved to:', OUTPUT_DIR)
