# CosyVoice 推理参数说明

## 参数含义

| 参数 | 含义 | 是否自定义 |
|---|---|---|
| `tts_text` | 想要生成的文本内容（你自己指定说什么）| 完全自定义 |
| `prompt_wav` | 参考音频文件（提供音色/说话风格）| 你的原始音频 |

---

## `prompt_text` 说明（zero_shot 用）

`prompt_text` 就是 `prompt_wav` 的转写文字，**没有固定选项，就是参考音频说的内容**。

**CosyVoice3 格式（必须包含 system prompt）：**
```
You are a helpful assistant.<|endofprompt|>{参考音频的转写}
```

示例：
```python
'You are a helpful assistant.<|endofprompt|>a boy looking at the frog'
```

**CosyVoice1/2 格式（直接写转写）：**
```python
'希望你以后能够做的比我还好呦。'
```

---

## `instruct_text` 全部选项（instruct2 用）

> 只有以下 26 条是训练集中出现过的，超出范围的描述（如英文情绪 `"Speak like a happy child"`）模型无法理解，生成质量会很差。

### 情绪类（3条）

```
You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>
You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>
You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>
```

### 语速 / 音量类（4条）

```
You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>
You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>
You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>
You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>
```

### 风格类（2条）

```
You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>
You are a helpful assistant. 你可以尝试用机器人的方式解答吗？<|endofprompt|>
```

### 方言类（17条）

```
You are a helpful assistant. 请用广东话表达。<|endofprompt|>
You are a helpful assistant. 请用东北话表达。<|endofprompt|>
You are a helpful assistant. 请用四川话表达。<|endofprompt|>
You are a helpful assistant. 请用上海话表达。<|endofprompt|>
You are a helpful assistant. 请用天津话表达。<|endofprompt|>
You are a helpful assistant. 请用山东话表达。<|endofprompt|>
You are a helpful assistant. 请用河南话表达。<|endofprompt|>
You are a helpful assistant. 请用湖南话表达。<|endofprompt|>
You are a helpful assistant. 请用湖北话表达。<|endofprompt|>
You are a helpful assistant. 请用江西话表达。<|endofprompt|>
You are a helpful assistant. 请用闽南话表达。<|endofprompt|>
You are a helpful assistant. 请用陕西话表达。<|endofprompt|>
You are a helpful assistant. 请用山西话表达。<|endofprompt|>
You are a helpful assistant. 请用云南话表达。<|endofprompt|>
You are a helpful assistant. 请用贵州话表达。<|endofprompt|>
You are a helpful assistant. 请用甘肃话表达。<|endofprompt|>
You are a helpful assistant. 请用宁夏话表达。<|endofprompt|>
```

---

## 5 种推理模式对比

| 模式 | 方法 | 需要 Audio | 需要 Text | 说明 |
|---|---|---|---|---|
| **SFT** | `inference_sft` | 否 | tts_text + spk_id | 使用预训练内置音色，仅 CosyVoice1 支持 |
| **Zero-Shot** | `inference_zero_shot` | prompt_wav | tts_text + prompt_text | 克隆参考音色，需要音频及其转写 |
| **Cross-Lingual** | `inference_cross_lingual` | prompt_wav | tts_text（只需生成文本）| 克隆音色，不需要 prompt_text，适合跨语言 |
| **Instruct** | `inference_instruct2` | prompt_wav | tts_text + instruct_text | 克隆音色 + 风格指令 |
| **VC（变声）** | `inference_vc` | source_wav + prompt_wav | 不需要 | 将 source_wav 内容转换为 prompt_wav 的音色 |
