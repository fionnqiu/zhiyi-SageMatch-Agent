"""Naive knowledge ingest + lexical recall. Numeric knobs live in backend/config/rag.yaml."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from app.core.rag_config import get_rag_config

# 与切块窗口同一套计数：英文词 / 数字串 / 单汉字 / 标点各 1 token；空白不计。
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\sA-Za-z0-9_\u4e00-\u9fff]+|\s+")
_DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def parse_bytes(filename: str, data: bytes) -> str:
    """Extract text from txt/md/pdf/docx. Binary office files are never decoded as UTF-8."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _pdf_text(data)
    # docx 是 zip。直接 decode 会把 PK 头和 XML 当正文，所以先按段落抽出 w:t。
    if name.endswith(".docx") or _is_zip(data):
        return _docx_text(data)
    if name.endswith(".doc"):
        raise ValueError("暂不支持旧版 .doc，请另存为 .docx")
    text = data.decode("utf-8", errors="ignore")
    if not text.strip():
        text = data.decode("gb18030", errors="ignore")
    return text.replace("\x00", " ").strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def token_estimate(text: str) -> int:
    """Count non-whitespace tokens; same unit as chunk.size / chunk.overlap."""
    n = sum(1 for t in tokenize(text) if not t.isspace())
    return max(1, n) if text.strip() else 0


def split_chunks(text: str) -> list[str]:
    """Slide a token window so recall has overlapping units."""
    chunk = get_rag_config().chunk
    size = max(1, chunk.size)
    overlap = max(0, min(chunk.overlap, size - 1))
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return []
    toks = tokenize(cleaned)
    weights = [0 if t.isspace() else 1 for t in toks]
    n = len(toks)
    parts: list[str] = []
    i = 0
    while i < n:
        acc = 0
        j = i
        while j < n and acc < size:
            acc += weights[j]
            j += 1
        piece = "".join(toks[i:j]).strip()
        if piece:
            parts.append(piece)
        if j >= n:
            break
        need = max(1, size - overlap)
        stepped = 0
        ni = i
        while ni < j and stepped < need:
            stepped += weights[ni]
            ni += 1
        i = max(ni, i + 1)
    return parts


def lexical_score(query: str, text: str) -> float:
    cfg = get_rag_config().recall
    query_terms = terms(query)
    hay = terms(text)
    if not query_terms or not hay:
        score = 0.0
    else:
        score = len(query_terms & hay) / len(query_terms)
        if query.strip() and query.strip().lower() in text.lower():
            score += cfg.exact_match_bonus
    return min(score, 1.0)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def recall(query: str, rows: list[dict], k: int | None = None, query_vec: list[float] | None = None) -> list[dict]:
    """Lexical overlap, optionally mixed with cosine when query_vec and chunk embeddings exist."""
    cfg = get_rag_config()
    top_k = k if k is not None else cfg.recall.top_k
    mix = cfg.embedding.lexical_weight if query_vec else 1.0
    scored: list[dict] = []
    for row in rows:
        lex = lexical_score(query, row["text"])
        vec = row.get("embedding") if isinstance(row.get("embedding"), list) else None
        if query_vec and vec:
            score = cosine(query_vec, vec) * (1 - mix) + lex * mix
        else:
            score = lex
        if score <= cfg.recall.score_floor:
            continue
        item = dict(row)
        item.pop("embedding", None)
        item["score"] = round(min(score, 1.0), 4)
        scored.append(item)
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[: max(1, top_k)]


def terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text.lower())
    return set(words)


def _is_zip(data: bytes) -> bool:
    return data[:2] == b"PK"


def _docx_text(data: bytes) -> str:
    """Read document.xml paragraphs. Stdlib only, so chat upload does not need python-docx."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("无法读取这份 Word 文件") from exc
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("无法读取这份 Word 文件") from exc
    lines: list[str] = []
    for para in root.iterfind(".//w:p", _DOCX_NS):
        bits = [node.text or "" for node in para.iterfind(".//w:t", _DOCX_NS)]
        line = "".join(bits).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _pdf_text(data: bytes) -> str:
    """Pull printable strings from a PDF stream without adding a PDF library."""
    raw = data.decode("latin-1", errors="ignore")
    bits = re.findall(r"\((?:\\.|[^\\)]){3,}\)", raw)
    out: list[str] = []
    for bit in bits:
        inner = bit[1:-1]
        inner = inner.replace("\\n", "\n").replace("\\r", "").replace("\\t", " ")
        inner = re.sub(r"\\[0-9]{1,3}", "", inner)
        inner = inner.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
        if any(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" for ch in inner):
            out.append(inner)
    text = "\n".join(out)
    if len(text) < 40:
        # Fallback: keep high-bit / letter runs from the raw stream.
        text = " ".join(re.findall(r"[\x20-\x7e\u4e00-\u9fff]{5,}", raw))
    return text.strip()


def upload_path(dest_dir: Path, material_id: str, filename: str) -> Path:
    # 和写入时用同一套文件名，后台任务才能按 id 把原件读回来。
    safe = re.sub(r"[^\w.\-]+", "_", filename)[:80] or "upload.txt"
    return dest_dir / f"{material_id}_{safe}"


def write_upload(dest_dir: Path, material_id: str, filename: str, data: bytes) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = upload_path(dest_dir, material_id, filename)
    path.write_bytes(data)
    return path
