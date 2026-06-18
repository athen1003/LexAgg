import io
import ipaddress
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.embedding import ModelNotFoundError, get_model
from app.normalizer import Normalizer
from app.vocabulary import Vocabulary, VocabularyLoadError

from typing import Union

from pydantic import BaseModel


class VocabItem(BaseModel):
    word: str
    polarity: str  # "正面" | "负面"
    category: str = ""


class WordItem(BaseModel):
    word: str
    polarity: str = ""  # "" = 自动推断, "正面"/"负面" = 锁定单桶, 其他值 = 强制 FALLBACK


class NormalizeJsonRequest(BaseModel):
    words: list[Union[str, WordItem]]  # 每个元素可以是纯字符串或 {word, polarity}
    vocab: list[VocabItem]
    model: str = "m3e"

    def get_words_and_hints(self) -> tuple[list[str], list[str], list[str]]:
        """返回 (word_list, hint_list, raw_polarity_list)。
        实现与 Excel 端点一致的极性处理：
        - "正面"/"负面" → 只在对应单桶搜
        - "" → 自动推断极性 + 双桶对比
        - 其他值 → 强制 FALLBACK,但不损失建议计算
        """
        w, h, raw = [], [], []
        for item in self.words:
            if isinstance(item, str):
                w.append(item)
                h.append("")
                raw.append("")
            else:
                w.append(item.word)
                hint = item.polarity
                raw.append(hint)
                if hint in ("正面", "负面"):
                    h.append(hint)
                elif hint == "":
                    h.append("")  # auto-infer
                else:
                    h.append("__FALLBACK__")  # 非法极性 → force FALLBACK
        return w, h, raw


DEFAULT_VOCAB_PATH = os.environ.get("VOCAB_PATH", "data/vocabulary.csv")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_EXCEL_ROWS = 50_000
MAX_JSON_ITEMS = 50_000
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")  # None → 仅允许 loopback

logger = logging.getLogger("wordtest")

# 启动时加载
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state(DEFAULT_VOCAB_PATH)
    yield


app = FastAPI(title="Word Normalizer", version="0.1.0", lifespan=lifespan)


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
    # 按模型名缓存 Normalizer:默认 m3e 启动时建,其他模型首次请求时懒建
    _state["normalizers"] = {}
    try:
        default_emb = get_model("m3e")
        _state["normalizers"]["m3e"] = Normalizer(default_emb, vocab)
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


@app.get("/api/v1/health")
async def health():
    vocab = _state.get("vocab")
    return {
        "status": "ok",
        "default_model": "m3e",
        "vocab_size": (
            len(vocab.buckets["正面"]) + len(vocab.buckets["负面"])
            if vocab
            else 0
        ),
    }


@app.post("/api/v1/normalize")
async def normalize(
    file: UploadFile = File(...),
    model: str = Query("m3e"),
    debug: int = Query(0),
):
    # 流式读取，避免大文件一次性缓冲到内存；累积超过 MAX_FILE_SIZE 立即 413
    CHUNK = 64 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            return JSONResponse(status_code=413, content={"error": "file_too_large"})
        chunks.append(chunk)
    content = b"".join(chunks)

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
            content={"error": "unknown_model", "supported": ["bge", "bge_base", "m3e", "fasttext"]},
        )

    t0 = time.time()
    # 用 batch 一次 encode,避免逐条 GPU launch overhead(单条调 GPU 慢 10-100×)
    results = normalizer.normalize_batch(lines, [""] * len(lines))
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


def _cell_to_str(v) -> str:
    """把 pandas 单元格值规整成字符串：None / NaN → ''。"""
    if v is None:
        return ""
    # pandas 把缺失单元格读成 float('nan')
    if isinstance(v, float):
        return ""
    return str(v)


@app.post("/api/v1/normalize/excel")
async def normalize_excel(
    file: UploadFile = File(...),
    model: str = Query("m3e"),
    debug: int = Query(0),  # 接受但暂未使用（与 txt 端点参数对齐）
):
    # 1. 流式读取 + 大小限制
    CHUNK = 64 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            return JSONResponse(status_code=413, content={"error": "file_too_large"})
        chunks.append(chunk)
    content = b"".join(chunks)

    # 2. 校验 xlsx：按文件名后缀或 content_type 至少一项命中
    fname = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    is_xlsx = fname.endswith(".xlsx") or "spreadsheet" in ctype or "openxmlformats" in ctype
    if not is_xlsx:
        return JSONResponse(status_code=400, content={"error": "invalid_file_type"})

    # 3. 解析 xlsx（无表头，列 0 = 原词，列 1 = 极性提示）
    try:
        df = pd.read_excel(io.BytesIO(content), dtype=str, header=None)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_file_type"})

    if len(df) > MAX_EXCEL_ROWS:
        return JSONResponse(
            status_code=400, content={"error": "too_many_rows", "limit": MAX_EXCEL_ROWS}
        )

    # 4. 取 normalizer
    try:
        normalizer = _get_normalizer(model)
    except ModelNotFoundError:
        return JSONResponse(
            status_code=400,
            content={"error": "unknown_model", "supported": ["bge", "bge_base", "m3e", "fasttext"]},
        )

    # 5. 逐行归一
    t0 = time.time()
    out_rows: list[dict] = []
    layer_counts = {"L1": 0, "L2": 0, "L3": 0, "FALLBACK": 0}

    # 5a. 收集全部有效行,极性提示非 {正面, 负面} → 归一为 "__FALLBACK__" 哨兵
    #     (normalizer 看到后会强制 FALLBACK 但仍算建议)
    batch_idx: list[int] = []
    batch_words: list[str] = []
    batch_hints: list[str] = []
    for _, row in df.iterrows():
        word = _cell_to_str(row.iloc[0]).strip() if len(row) >= 1 else ""
        polarity_hint = _cell_to_str(row.iloc[1]) if len(row) >= 2 else ""
        if not word:
            continue
        effective_hint = polarity_hint if polarity_hint in {"正面", "负面"} else "__FALLBACK__"
        batch_idx.append(len(out_rows))
        batch_words.append(word)
        batch_hints.append(effective_hint)
        out_rows.append(
            {
                "原词": word,
                "归一词": word,  # 占位,后面覆盖
                "命中层级": "FALLBACK",  # 占位
                "分数": 0.0,
                "输入极性": polarity_hint,
                "归一-大类": "",
                "建议归一词": "",
                "建议分数": 0.0,
                "建议-大类": "",
            }
        )

    # 5b. 批量归一 + 累计 FALLBACK
    fb_acc = _state.setdefault("fallback_acc", {})
    if batch_words:
        results = normalizer.normalize_batch(batch_words, batch_hints)
        for j, row_pos in enumerate(batch_idx):
            r = results[j]
            out_rows[row_pos]["归一词"] = r.normalized
            out_rows[row_pos]["命中层级"] = r.matched_layer
            out_rows[row_pos]["分数"] = round(r.score, 4)
            out_rows[row_pos]["归一-大类"] = r.matched_category
            if r.matched_layer == "FALLBACK":
                out_rows[row_pos]["建议归一词"] = r.best_candidate
                out_rows[row_pos]["建议分数"] = round(r.best_candidate_score, 4)
                out_rows[row_pos]["建议-大类"] = r.best_candidate_category
                # 累计到 fallback_acc(运营期聚类用)
                entry = fb_acc.get(r.original)
                if entry is None:
                    fb_acc[r.original] = {"freq": 1, "first_seen": time.time()}
                else:
                    entry["freq"] += 1
            layer_counts[r.matched_layer] += 1

    elapsed = time.time() - t0

    logger.info(
        f"POST /normalize/excel {len(out_rows)} rows model={model} elapsed={elapsed:.1f}s"
    )

    # 6. 写 xlsx
    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "原词", "归一词", "命中层级", "分数", "输入极性", "归一-大类",
            "建议归一词", "建议分数", "建议-大类",
        ],
    )
    buf = io.BytesIO()
    out_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    summary = {"total": len(out_rows), **layer_counts}
    return StreamingResponse(
        io.BytesIO(buf.getvalue()),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=normalized.xlsx",
            "X-Summary": json.dumps(summary, ensure_ascii=False),
        },
    )


@app.get("/api/v1/normalize/excel/template")
async def normalize_excel_template():
    """下载 Excel 导入模板：2 列（原词 / 极性），含表头 + 2 行示例。"""
    df = pd.DataFrame(
        [
            ["原词", "极性"],
            ["舒适", "正面"],
            ["破洞", "负面"],
        ]
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False, header=False, engine="openpyxl")
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue()),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=template.xlsx"},
    )


@app.post("/api/v1/normalize/json")
async def normalize_json(body: NormalizeJsonRequest):
    """JSON 归一：传入待分析词数组 + 标准词库数组，返回归一结果。

    words 每个元素可以是纯字符串，也可以是 {word, polarity}：

    {
      "words": [
        {"word": "好用的", "polarity": "正面"},
        "很差",
        {"word": "不明", "polarity": "?"}
      ],
      "vocab": [
        {"word": "质量好", "polarity": "正面", "category": "质量"},
        {"word": "质量差", "polarity": "负面", "category": "质量"}
      ],
      "model": "m3e"
    }
    """
    if not body.words:
        return JSONResponse(status_code=400, content={"error": "empty_words"})
    if not body.vocab:
        return JSONResponse(status_code=400, content={"error": "empty_vocab"})
    if len(body.words) > MAX_JSON_ITEMS:
        return JSONResponse(status_code=400, content={"error": "too_many_words", "limit": MAX_JSON_ITEMS})
    if len(body.vocab) > MAX_JSON_ITEMS:
        return JSONResponse(status_code=400, content={"error": "too_many_vocab", "limit": MAX_JSON_ITEMS})

    word_list, hints, raw_polarities = body.get_words_and_hints()

    # 用传入的 voca 构建临时词库
    try:
        vocab = Vocabulary.from_json([item.model_dump() for item in body.vocab])
    except VocabularyLoadError as e:
        return JSONResponse(status_code=400, content={"error": "invalid_vocab", "msg": str(e)})

    # 复用已加载的 embedding 模型(权重复用),只为本次请求构造 Normalizer
    try:
        embedding = get_model(body.model)
    except ModelNotFoundError:
        return JSONResponse(
            status_code=400,
            content={"error": "unknown_model", "supported": ["bge", "bge_base", "m3e", "fasttext"]},
        )

    normalizer = Normalizer(embedding, vocab)

    t0 = time.time()
    results = normalizer.normalize_batch(word_list, hints)
    elapsed_ms = (time.time() - t0) * 1000

    layer_counts = {"L1": 0, "L2": 0, "L3": 0, "FALLBACK": 0}
    out = []
    for i, r in enumerate(results):
        layer_counts[r.matched_layer] += 1
        out.append({
            "original": r.original,
            "normalized": r.normalized,
            "layer": r.matched_layer,
            "score": round(r.score, 4),
            "category": r.matched_category,
            "input_polarity": raw_polarities[i],
            # FALLBACK 时带建议
            **(
                {"suggestion": r.best_candidate,
                 "suggestion_score": round(r.best_candidate_score, 4),
                 "suggestion_category": r.best_candidate_category}
                if r.matched_layer == "FALLBACK" else {}
            ),
        })

    logger.info(
        f"POST /normalize/json {len(body.words)} words vocab={len(body.vocab)} "
        f"model={body.model} elapsed={elapsed_ms:.0f}ms matched={layer_counts}"
    )

    return {
        "results": out,
        "summary": {"total": len(results), **layer_counts},
        "model": body.model,
        "elapsed_ms": round(elapsed_ms, 1),
    }


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


@app.get("/api/v1/admin/fallbacks")
async def get_fallbacks(
    request: Request,
    model: str = Query("m3e"),
    min_freq: int = Query(1, ge=1),
    limit: int = Query(1000, ge=1, le=10000),
):
    """累计 FALLBACK 词的运营期分析。返回按「建议归一词」分组、按总频次降序的 JSON。"""
    denied = _check_admin_auth(
        request.client.host if request.client else None,
        request.headers.get("authorization"),
    )
    if denied is not None:
        return denied

    fb_acc = _state.get("fallback_acc", {})
    total_unique = len(fb_acc)
    total_freq = sum(info["freq"] for info in fb_acc.values())

    items = [(w, info) for w, info in fb_acc.items() if info["freq"] >= min_freq]
    items.sort(key=lambda x: -x[1]["freq"])
    items = items[:limit]

    if not items:
        return {
            "total_unique": total_unique,
            "total_freq": total_freq,
            "filtered": 0,
            "by_suggestion": [],
        }

    try:
        normalizer = _get_normalizer(model)
    except ModelNotFoundError:
        return JSONResponse(
            status_code=400,
            content={"error": "unknown_model", "supported": ["bge", "bge_base", "m3e", "fasttext"]},
        )

    words = [w for w, _ in items]
    vecs = normalizer.embedding.encode(words)

    groups: dict[str, dict] = {}
    for j, (word, info) in enumerate(items):
        cand, score, cat = normalizer._suggest_candidate(vecs[j])
        key = cand if cand else "__no_suggestion__"
        g = groups.get(key)
        if g is None:
            g = {
                "suggested_vocab": cand,
                "suggestion_category": cat,
                "suggestion_score": round(score, 4),
                "fallbacks": [],
                "total_freq": 0,
            }
            groups[key] = g
        g["fallbacks"].append({
            "word": word,
            "freq": info["freq"],
            "score": round(score, 4),
        })
        g["total_freq"] += info["freq"]

    by_suggestion = sorted(groups.values(), key=lambda g: -g["total_freq"])

    return {
        "total_unique": total_unique,
        "total_freq": total_freq,
        "filtered": len(items),
        "by_suggestion": by_suggestion,
    }


@app.post("/api/v1/admin/fallbacks/reset")
async def reset_fallbacks(request: Request):
    """清空累计的 FALLBACK 词(测试 + 运营期重置)。"""
    denied = _check_admin_auth(
        request.client.host if request.client else None,
        request.headers.get("authorization"),
    )
    if denied is not None:
        return denied
    _state["fallback_acc"] = {}
    return {"status": "ok", "cleared": True}


# 静态前端（必须放在所有 API 路由之后，避免 shadow）
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
