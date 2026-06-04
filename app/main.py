import io
import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.embedding import ModelNotFoundError, get_model
from app.normalizer import Normalizer
from app.vocabulary import Vocabulary, VocabularyLoadError

DEFAULT_VOCAB_PATH = os.environ.get("VOCAB_PATH", "data/vocabulary.csv")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app = FastAPI(title="Word Normalizer", version="0.1.0")

logger = logging.getLogger("wordtest")

# 启动时加载
_state: dict = {}


def _load_state(vocab_path: str) -> None:
    vocab = Vocabulary.load(vocab_path)
    # 加载 aliases.json（如存在）
    aliases_path = Path(vocab_path).parent / "aliases.json"
    if aliases_path.exists():
        with open(aliases_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 把 aliases.json 中人工维护的别名合并到 vocab
        for variant, std in data.items():
            std_polarity = vocab.polarity_map.get(std)
            if std_polarity is None:
                raise VocabularyLoadError(
                    f"aliases.json: 目标词 '{std}' 不在词库中（变体 '{variant}'）"
                )
            existing_polarity = vocab.polarity_map.get(variant)
            if existing_polarity is not None and existing_polarity != std_polarity:
                raise VocabularyLoadError(
                    f"aliases.json: 变体 '{variant}' 已存在（极性 {existing_polarity}），"
                    f"不能同时作为 '{std}'（极性 {std_polarity}）的别名"
                )
            if variant not in vocab.polarity_map:
                vocab.buckets[std_polarity].append(variant)
                vocab.polarity_map[variant] = std_polarity
            vocab.alias_map[variant] = std

    _state["vocab"] = vocab
    _state["vocab_path"] = vocab_path
    # 默认模型预加载
    try:
        default_emb = get_model("bge")
        _state["default_normalizer"] = Normalizer(default_emb, vocab)
    except Exception:
        _state["default_normalizer"] = None


@app.on_event("startup")
def startup():
    _load_state(DEFAULT_VOCAB_PATH)


@app.get("/api/v1/health")
async def health():
    vocab = _state.get("vocab")
    return {
        "status": "ok",
        "default_model": "bge",
        "vocab_size": (
            len(vocab.buckets["正面"]) + len(vocab.buckets["负面"])
            if vocab
            else 0
        ),
    }


@app.post("/api/v1/normalize")
async def normalize(
    file: UploadFile = File(...),
    model: str = Query("bge"),
    debug: int = Query(0),
):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "file_too_large"})

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "invalid_encoding"})

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return JSONResponse(status_code=400, content={"error": "empty_file"})

    vocab = _state["vocab"]
    try:
        embedding = get_model(model)
    except ModelNotFoundError:
        return JSONResponse(
            status_code=400,
            content={"error": "unknown_model", "supported": ["bge", "fasttext"]},
        )

    normalizer = Normalizer(embedding, vocab)
    t0 = time.time()
    results = [normalizer.normalize(line) for line in lines]
    elapsed = time.time() - t0

    layer_counts = {"L1": 0, "L2": 0, "L3": 0, "FALLBACK": 0}
    for r in results:
        layer_counts[r.matched_layer] += 1

    logger.info(
        f"POST /normalize {len(lines)} lines model={model} "
        f"matched={layer_counts} total={elapsed:.1f}s"
    )

    sep = "\t"
    if debug:
        output = "\n".join(
            f"{r.original}{sep}{r.normalized}{sep}{r.matched_layer}{sep}{r.score:.3f}"
            for r in results
        )
    else:
        output = "\n".join(f"{r.original}{sep}{r.normalized}" for r in results)

    return StreamingResponse(
        io.StringIO(output),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=normalized.txt"},
    )


@app.post("/api/v1/admin/reload")
async def reload_vocab():
    try:
        _load_state(_state.get("vocab_path", DEFAULT_VOCAB_PATH))
    except VocabularyLoadError as e:
        return JSONResponse(
            status_code=500, content={"error": "reload_failed", "msg": str(e)}
        )
    vocab = _state["vocab"]
    return {
        "status": "ok",
        "正面": len(vocab.buckets["正面"]),
        "负面": len(vocab.buckets["负面"]),
    }
