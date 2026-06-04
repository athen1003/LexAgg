"""下载 fastText 中文模型到 models/ 目录。

默认从 Hugging Face 镜像下载小型压缩版（~100MB）。
若用户已有 cc.zh.300.bin 全量版，可放入 models/ 跳过下载。

注：Task 4 已用 gensim 替代 fasttext-wheel。gensim 通过
KeyedVectors.load_word2vec_format 同样能加载 Facebook 官方 .bin 文件。
"""
import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "cc.zh.300.bin"

# 备选 URL（Hugging Face 镜像 + 官方源）
URLS = [
    "https://huggingface.co/facebook/fasttext-zh-vectors/resolve/main/cc.zh.300.bin",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz",
]


def main() -> int:
    if MODEL_PATH.exists():
        print(f"模型已存在: {MODEL_PATH}")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for url in URLS:
        print(f"尝试下载: {url}")
        try:
            if url.endswith(".gz"):
                import gzip
                tmp_gz = MODEL_PATH.with_suffix(".bin.gz")
                urllib.request.urlretrieve(url, tmp_gz)
                print("解压中...")
                with gzip.open(tmp_gz, "rb") as f_in, open(MODEL_PATH, "wb") as f_out:
                    f_out.writelines(f_in)
                tmp_gz.unlink()
            else:
                urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"下载完成: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
            return 0
        except Exception as e:
            print(f"失败: {e}")
            continue

    print("所有下载源均失败，请手动放置模型到 models/cc.zh.300.bin")
    return 1


if __name__ == "__main__":
    sys.exit(main())
