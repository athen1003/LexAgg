import io
import ipaddress
import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.embedding import ModelNotFoundError, get_model
from app.normalizer import Normalizer
from app.vocabulary import Vocabulary, VocabularyLoadError

DEFAULT_VOCAB_PATH = os.environ.get("VOCAB_PATH", "data/vocabulary.csv")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")  # None → 仅允许 loopback

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
    # 按模型名缓存 Normalizer：默认 bge 启动时建，其他模型首次请求时懒建
    _state["normalizers"] = {}
    try:
        default_emb = get_model("bge")
        _state["normalizers"]["bge"] = Normalizer(default_emb, vocab)
    except Exception:
        # 启动时模型加载失败时，不阻塞；后续请求再尝试
        pass


def _get_normalizer(model: str) -> Normalizer:
    """按 model 名取缓存的 Normalizer；缺失则构建并缓存。"""
    cached = _state["normalizers"].get(model)
    if cached is not None:
        return cached
    embedding = get_model(model)
    normalizer = Normalizer(embedding, _state["vocab"])
    _state["normalizers"][model] = normalizer
    return normalizer


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
        normalizer = _get_normalizer(model)
    except ModelNotFoundError:
        return JSONResponse(
            status_code=400,
            content={"error": "unknown_model", "supported": ["bge", "fasttext"]},
        )

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


def _is_loopback(client_host: str | None) -> bool:
    """判断请求来源是否为 loopback（127.0.0.0/8 或 ::1）。"""
    if not client_host:
        return False
    try:
        ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    return ip.is_loopback


def _check_admin_auth(client_host: str | None, auth_header: str | None) -> JSONResponse | None:
    """校验管理端权限。返回 None 表示通过；返回 JSONResponse 表示拒绝。

    client_host: 来自 request.client.host，可为 None（无连接信息时按非 loopback 处理）
    auth_header: 来自 request.headers.get('authorization')，可为空
    """
    token = ADMIN_TOKEN
    if token is None:
        # 未配置 ADMIN_TOKEN → 仅 loopback 允许
        if _is_loopback(client_host):
            return None
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    presented = auth_header[len("Bearer "):].strip()
    if presented != token:
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return None


@app.post("/api/v1/admin/reload")
async def reload_vocab(request: Request):
    denied = _check_admin_auth(
        request.client.host if request.client else None,
        request.headers.get("authorization"),
    )
    if denied is not None:
        return denied
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
