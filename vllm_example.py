import sys
import types

# 部分环境无 pkg_resources（或仅有残缺），需注入最小桩以兼容 lightning / pyworld 等
try:
    import pkg_resources as _pr
    _pr.get_distribution("setuptools")  # 确保 get_distribution 可用
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
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm


def cosyvoice2_example():
    """ CosyVoice2 推理示例（从 Hugging Face 自动下载）。load_vllm=True 需 CUDA 12+，否则用 load_vllm=False。
    """
    cosyvoice = AutoModel(model_dir='FunAudioLLM/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=False, fp16=True)
    set_all_random_seed(0)
    for _ in cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False):
        pass


def cosyvoice3_example():
    """ CosyVoice3 推理示例（从 Hugging Face 自动下载：FunAudioLLM/Fun-CosyVoice3-0.5B-2512）。
    load_vllm=True 需要 CUDA 12+（flashinfer 依赖 cuda/functional），当前为 CUDA 11.8 时请用 load_vllm=False。
    """
    cosyvoice = AutoModel(model_dir='FunAudioLLM/Fun-CosyVoice3-0.5B-2512', load_trt=True, load_vllm=False, fp16=False)
    set_all_random_seed(0)
    for _ in cosyvoice.inference_zero_shot(
        '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        stream=False,
    ):
        pass  # 流式逐段产出，这里仅消费；若需保存可在此收集并拼接


def main():
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()
