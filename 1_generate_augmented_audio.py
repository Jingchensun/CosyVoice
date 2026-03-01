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

AUDIO_PATH = 'dataset/english_free_speech/files_cut_by_sentences/01_M_native/a boy looking at the frog.wav'
TRANSCRIPTION = 'it’s broken'
OUTPUT_DIR = 'audio_generate'

# -----------------------------------------------------------------------
# 只使用模型 instruct_list 中真实支持的指令（cosyvoice/utils/common.py）
# 英文情绪指令（如 happy/curious）未在训练数据中出现，会导致生成质量极差
# 情绪/风格用中文指令，音量/语速可用英文或中文
# (instruct_text, speed)
# -----------------------------------------------------------------------
STYLES = [
    ('happy',       'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>',        1.0),
    ('sad',         'You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>',        0.9),
    ('angry',       'You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>',        1.1),
    ('fast',        'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',  1.3),
    ('slow',        'You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>',  0.7),
    ('loud',        'You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>', 1.0),
    ('soft',        'You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>',  0.9),
    ('peppa_style', 'You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>', 1.0),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

cosyvoice = AutoModel(
    model_dir='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    load_trt=True, load_vllm=False, fp16=False
)

for idx, (style_name, instruct_text, speed) in enumerate(STYLES):
    set_all_random_seed(idx * 10)   # 每种风格用不同 seed，增加多样性
    chunks = []
    for output in cosyvoice.inference_instruct2(
        TRANSCRIPTION,
        instruct_text,
        AUDIO_PATH,
        stream=False,
        speed=speed,
    ):
        chunks.append(output['tts_speech'])

    import torch
    audio = torch.cat(chunks, dim=1)
    out_path = os.path.join(OUTPUT_DIR, f'{style_name}.wav')
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)
    print(f'[{style_name}] saved -> {out_path}')

print('Done. All audio saved to:', OUTPUT_DIR)
