import sys
import types
import os
import random
import torch
import torchaudio
from pathlib import Path

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

SOURCE_ROOT  = Path('dataset/english_free_speech/files_cut_by_sentences')
TEMPLATE     = Path('generation/template.txt')
OUTPUT_DIR   = Path('audio_generate_zeroshot')

# 解析 template.txt → [(编号, 句子), ...]
template_sentences = [
    (int(line.split('.', 1)[0]), line.split('.', 1)[1].strip())
    for line in TEMPLATE.read_text().splitlines()
    if line.strip() and '.' in line and line.split('.', 1)[0].strip().isdigit()
]

cosyvoice = AutoModel(
    model_dir='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    load_trt=True, load_vllm=False, fp16=False
)

# 过滤掉单词数 < 4 的短句（避免生成音频过短导致卷积核报错）
valid_templates = [(idx, txt) for idx, txt in template_sentences if len(txt.split()) >= 4]

# 提前分配所有随机模板，避免 set_all_random_seed 重置 Python random 状态
all_tasks = [
    (audio_path, random.choice(valid_templates))
    for speaker_dir in sorted(d for d in SOURCE_ROOT.iterdir() if d.is_dir())
    for audio_path in sorted(speaker_dir.glob('*.wav'))
]

for audio_path, (tmpl_idx, tmpl_text) in all_tasks:
    out_dir = OUTPUT_DIR / audio_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_transcription = audio_path.stem.strip()
    prompt_text = f'You are a helpful assistant.<|endofprompt|>{prompt_transcription}'

    set_all_random_seed(0)
    chunks = []
    for output in cosyvoice.inference_zero_shot(
        tmpl_text, prompt_text, str(audio_path), stream=False
    ):
        chunks.append(output['tts_speech'])

    audio = torch.cat(chunks, dim=1)
    out_path = out_dir / f'{tmpl_idx:03d}_{audio_path.stem}.wav'
    torchaudio.save(str(out_path), audio, cosyvoice.sample_rate)
    print(f'[{tmpl_idx:03d}] {tmpl_text!r} -> {out_path}')

print('Done. All audio saved to:', OUTPUT_DIR)
