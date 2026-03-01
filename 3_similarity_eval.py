"""
评估 audio_generate_zeroshot/ 中生成音频与原始音频的说话人相似度。

生成文件命名规则：{tmpl_idx:03d}_{original_stem}.wav
映射关系：去掉三位数字前缀和下划线，即可找到原始文件。

使用 ECAPA-TDNN（SpeechBrain）计算说话人余弦相似度，输出到 similarity_zeroshot.txt。
"""

from pathlib import Path
import torch
import librosa

GENERATED_DIR = Path("audio_generate_zeroshot")
ORIGINAL_DIR  = Path("dataset/english_free_speech/files_cut_by_sentences")
OUTPUT_TXT    = Path("similarity_zeroshot.txt")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
ECAPA_SOURCE  = "speechbrain/spkrec-ecapa-voxceleb"


def load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source=ECAPA_SOURCE, run_opts={"device": DEVICE})


@torch.inference_mode()
def get_embedding(path: Path, classifier) -> torch.Tensor:
    wav, _ = librosa.load(str(path), sr=16000, mono=True)
    sig = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
    emb = classifier.encode_batch(sig).squeeze()
    return torch.nn.functional.normalize(emb, dim=0).cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a * b).clamp(-1.0, 1.0).item())


def main():
    classifier = load_ecapa()

    written = skipped = 0
    with OUTPUT_TXT.open("w", encoding="utf-8") as f_out:
        f_out.write("speaker,generated_file,original_file,similarity_score\n")

        for gen_audio in sorted(GENERATED_DIR.rglob("*.wav")):
            speaker = gen_audio.parent.name
            # 去掉 "XXX_" 前缀还原原始文件名
            stem = gen_audio.stem
            original_stem = stem[4:] if (len(stem) > 4 and stem[:3].isdigit() and stem[3] == "_") else stem
            orig_audio = ORIGINAL_DIR / speaker / f"{original_stem}.wav"

            if not orig_audio.is_file():
                print(f"[WARN] 找不到原始文件，跳过：{orig_audio}")
                skipped += 1
                continue

            gen_emb  = get_embedding(gen_audio, classifier)
            orig_emb = get_embedding(orig_audio, classifier)
            score    = cosine(gen_emb, orig_emb)

            f_out.write(f"{speaker},{gen_audio.name},{orig_audio.name},{score:.6f}\n")
            print(f"[{written+1:04d}] {speaker} | {gen_audio.name} -> {score:.4f}")
            written += 1

    print(f"\n完成：写入 {written} 行，跳过 {skipped} 个。结果保存至：{OUTPUT_TXT}")


if __name__ == "__main__":
    main()
