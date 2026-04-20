#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""默认流程（无参数或仅传一个 .pdf）：① PDF→同主名 .md ② jieba→*_jieba_seg.md
③ text-embedding-v4→FAISS ④ 用查询向量检索相似片段，将命中结果与完整
*_faiss_meta.json 一并交给 DeepSeek（默认 deepseek-chat，即 V3 对话模型）生成回答。

环境变量：DASHSCOPE_API_KEY（嵌入与检索查询向量）、DEEPSEEK_API_KEY（对话）。
可选：DEEPSEEK_BASE_URL（默认 https://api.deepseek.com/v1）、DEEPSEEK_MODEL（默认
deepseek-chat）、RAG_QUERY、RAG_TOP_K、RAG_MAX_META_CHARS。

另：对 .md 为分词+FAISS+④；--faiss 仅重建索引；--rag 仅执行④（默认浦发索引与问题）。

依赖：pip install docling jieba openai faiss-cpu numpy

FAISS 向量（text-embedding-v4，需百炼 API）：
  • 环境变量：DASHSCOPE_API_KEY
  • python pdf_extract_docling_to_md.py --faiss
  • python pdf_extract_docling_to_md.py --faiss /path/to/xxx_jieba_seg.md

首次运行会从 Hugging Face 下载模型（需联网）。解析失败时直接抛出异常（无其它后备引擎）。

国内访问默认走镜像（可在运行前覆盖）：
  • 默认：HF_ENDPOINT=https://hf-mirror.com（未设置时才生效）
  • 自定义：export HF_ENDPOINT=https://你的镜像域名

证书问题可任选：
  • python -m pip install -U certifi
  • Python 3.10+：pip install truststore
  • 调试时临时：export DOCMLING_ALLOW_INSECURE_SSL=1（对 Session.request 默认 verify=False）

用法：
  python pdf_extract_docling_to_md.py   # 默认浦发：①②③④（含 DeepSeek）
  DOCMLING_ALLOW_INSECURE_SSL=1 python pdf_extract_docling_to_md.py
  python pdf_extract_docling_to_md.py /path/to/file.pdf   # 同上全流程
  python pdf_extract_docling_to_md.py /path/to/file.md   # jieba 分词后自动 FAISS
  python pdf_extract_docling_to_md.py a.md b.md
  python pdf_extract_docling_to_md.py --faiss
  python pdf_extract_docling_to_md.py --faiss /path/to/xxx_jieba_seg.md
  python pdf_extract_docling_to_md.py --rag
  RAG_QUERY="你的问题" python pdf_extract_docling_to_md.py --rag
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# 证书（须在 import huggingface_hub / docling 之前）
# ---------------------------------------------------------------------------
try:
    import certifi

    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
except ImportError:
    pass

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Hugging Face Hub 国内镜像（须在 import huggingface_hub / docling 之前）
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _patch_requests_verify_disabled() -> None:
    if os.environ.get("DOCMLING_ALLOW_INSECURE_SSL") != "1":
        return
    warnings.warn(
        "DOCMLING_ALLOW_INSECURE_SSL=1：HTTPS 验证已关闭，仅用于调试。",
        UserWarning,
        stacklevel=2,
    )
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    import requests

    _orig_request = requests.Session.request

    def _request(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_request(self, method, url, **kwargs)

    requests.Session.request = _request  # type: ignore[method-assign]


_patch_requests_verify_disabled()

if os.environ.get("DOCLING_INSECURE_SSL") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf"
DEFAULT_JIEBA_SEG_MD = (
    BASE_DIR / "浦发上海浦东发展银行西安分行个金客户经理考核办法_jieba_seg.md"
)
EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# DashScope text-embedding-v4：单次 input 条数上限为 10（再大返回 400 InvalidParameter）
EMBED_BATCH_SIZE = 10

DEFAULT_FAISS_PATH = (
    BASE_DIR / "浦发上海浦东发展银行西安分行个金客户经理考核办法_jieba_seg.faiss"
)
DEFAULT_META_PATH = (
    BASE_DIR / "浦发上海浦东发展银行西安分行个金客户经理考核办法_jieba_seg_faiss_meta.json"
)
DEFAULT_RAG_QUERY = (
    "请结合制度原文，说明与「服务质量考核办法」相关的内容、条款及考核要点。"
)
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_MAX_META_CHARS = int(os.getenv("RAG_MAX_META_CHARS", "120000"))


def _convert_pdf(pdf_path: Path) -> None:
    from docling.document_converter import DocumentConverter

    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"不是 PDF 文件: {pdf_path}")
    if not pdf_path.is_file():
        raise SystemExit(f"未找到 PDF 文件: {pdf_path}")

    out_md = pdf_path.with_suffix(".md")
    print(f"正在转换: {pdf_path.name} …")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    md = result.document.export_to_markdown()

    out_md.write_text(md, encoding="utf-8")
    print(f"已写入: {out_md}")


def _md_lines_for_embedding(md_path: Path, *, max_chars: int = 12000) -> list[dict]:
    """按非空行切分为片段，供 embedding；过长单行再按字符切段。"""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict] = []
    seq = 0
    for line_no, raw in enumerate(text.splitlines()):
        s = raw.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            rows.append({"line_no": line_no, "seq": seq, "text": s})
            seq += 1
            continue
        for off in range(0, len(s), max_chars):
            rows.append({"line_no": line_no, "seq": seq, "text": s[off : off + max_chars]})
            seq += 1
    return rows


def _embed_md_to_faiss(md_path: Path) -> tuple[Path, Path]:
    import numpy as np
    import faiss
    from openai import OpenAI

    md_path = md_path.expanduser().resolve()
    if not md_path.is_file():
        raise SystemExit(f"未找到 Markdown 文件: {md_path}")
    if md_path.suffix.lower() != ".md":
        raise SystemExit(f"FAISS 模式仅支持 .md: {md_path}")

    chunks = _md_lines_for_embedding(md_path)
    if not chunks:
        raise SystemExit(f"文件无可用文本行: {md_path}")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("请设置环境变量 DASHSCOPE_API_KEY（百炼 API Key）。")

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    vectors_list: list[list[float]] = []
    metadata_store: list[dict] = []

    print(f"正在生成向量（{EMBED_MODEL}，维度 {EMBED_DIM}），共 {len(chunks)} 条 …")
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        inputs = [c["text"] for c in batch]
        completion = client.embeddings.create(
            model=EMBED_MODEL,
            input=inputs,
            dimensions=EMBED_DIM,
            encoding_format="float",
        )
        emb_sorted = sorted(completion.data, key=lambda d: d.index)
        if len(emb_sorted) != len(batch):
            raise SystemExit(
                f"Embedding 返回条数与请求不一致: 期望 {len(batch)}，实际 {len(emb_sorted)}"
            )
        for j, item in enumerate(emb_sorted):
            vectors_list.append(item.embedding)
            c = batch[j]
            metadata_store.append(
                {
                    "faiss_id": len(metadata_store),
                    "source": md_path.name,
                    "line_no": c["line_no"],
                    "seq": c["seq"],
                    "text": c["text"],
                }
            )
        done = min(start + EMBED_BATCH_SIZE, len(chunks))
        print(f"  - 已完成 {done}/{len(chunks)}")

    vectors_np = np.array(vectors_list, dtype="float32")
    ids_np = np.arange(len(vectors_list), dtype=np.int64)
    index_flat = faiss.IndexFlatL2(EMBED_DIM)
    index = faiss.IndexIDMap(index_flat)
    index.add_with_ids(vectors_np, ids_np)

    faiss_path = md_path.with_suffix(".faiss")
    meta_path = md_path.parent / f"{md_path.stem}_faiss_meta.json"
    faiss.write_index(index, str(faiss_path))
    meta_path.write_text(
        json.dumps(metadata_store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"FAISS 索引已写入: {faiss_path}（ntotal={index.ntotal}）")
    print(f"元数据已写入: {meta_path}")
    return faiss_path, meta_path


def _retrieve_similar_chunks(
    faiss_path: Path,
    meta_path: Path,
    query: str,
    *,
    top_k: int | None = None,
) -> list[dict]:
    import numpy as np
    import faiss
    from openai import OpenAI

    k_req = top_k if top_k is not None else RAG_TOP_K
    faiss_path = faiss_path.expanduser().resolve()
    meta_path = meta_path.expanduser().resolve()
    if not faiss_path.is_file():
        raise SystemExit(f"未找到 FAISS 索引: {faiss_path}")
    if not meta_path.is_file():
        raise SystemExit(f"未找到元数据 JSON: {meta_path}")

    meta_store: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(faiss_path))

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("FAISS 检索需 DASHSCOPE_API_KEY（用于将查询转为向量）。")
    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    completion = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=EMBED_DIM,
        encoding_format="float",
    )
    qv = np.array([completion.data[0].embedding], dtype="float32")
    k = min(k_req, int(index.ntotal))
    if k < 1:
        raise SystemExit("FAISS 索引为空，无法检索。")
    dist, ids = index.search(qv, k)
    hits: list[dict] = []
    for rank in range(k):
        fid = int(ids[0][rank])
        if fid < 0 or fid >= len(meta_store):
            continue
        row = dict(meta_store[fid])
        row["retrieval_rank"] = rank + 1
        row["l2_distance"] = float(dist[0][rank])
        hits.append(row)
    return hits


def _rag_answer_with_deepseek(
    query: str,
    faiss_path: Path,
    meta_path: Path,
    *,
    top_k: int | None = None,
) -> str:
    from openai import OpenAI

    hits = _retrieve_similar_chunks(faiss_path, meta_path, query, top_k=top_k)
    meta_full = json.loads(meta_path.expanduser().resolve().read_text(encoding="utf-8"))
    meta_json_str = json.dumps(meta_full, ensure_ascii=False, indent=2)
    if len(meta_json_str) > RAG_MAX_META_CHARS:
        meta_json_str = (
            meta_json_str[:RAG_MAX_META_CHARS]
            + f"\n\n…（元数据 JSON 已截断至约 {RAG_MAX_META_CHARS} 字符，完整文件见 {meta_path}）"
        )

    sk = os.getenv("DEEPSEEK_API_KEY")
    if not sk:
        raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY。")

    ds = OpenAI(api_key=sk, base_url=DEEPSEEK_BASE_URL)
    system = (
        "你是银行业务与考核制度解读助手。请严格依据用户给出的「向量检索命中的片段」"
        "以及「完整元数据 JSON」作答，使用简洁的简体中文，并尽量引用原文要点。"
        "若材料中找不到依据，请明确说明。"
    )
    user_body = (
        f"## 用户问题\n{query}\n\n"
        f"## FAISS 向量检索到的相似片段（共 {len(hits)} 条，L2 距离越小越相似）\n"
        f"{json.dumps(hits, ensure_ascii=False, indent=2)}\n\n"
        "## 完整元数据（与索引逐条对齐的 _faiss_meta.json）\n"
        f"```json\n{meta_json_str}\n```\n\n"
        "请基于上述材料回答用户问题。"
    )
    resp = ds.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_body},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _print_deepseek_answer(answer: str) -> None:
    print(f"\n--- DeepSeek（{DEEPSEEK_MODEL}）回答 ---\n")
    print(answer)


def _pipeline_pdf_md_jieba_faiss(pdf_path: Path) -> None:
    """① PDF→MD ② jieba→*_jieba_seg.md ③ embedding→FAISS ④ 检索 + DeepSeek。"""
    pdf_path = pdf_path.expanduser().resolve()
    print("【1/4】PDF → Markdown …")
    _convert_pdf(pdf_path)
    md_out = pdf_path.with_suffix(".md")
    print("【2/4】Markdown → jieba 分词 …")
    seg_out = _segment_md_to_jieba_file(md_out)
    print(f"已写入: {seg_out}")
    print("【3/4】text-embedding-v4 → FAISS …")
    faiss_path, meta_path = _embed_md_to_faiss(seg_out)
    print("【4/4】FAISS 检索 → DeepSeek …")
    q = os.getenv("RAG_QUERY", DEFAULT_RAG_QUERY)
    ans = _rag_answer_with_deepseek(q, faiss_path, meta_path)
    _print_deepseek_answer(ans)


def _segment_md_to_jieba_file(md_path: Path, *, sep: str = " ") -> Path:
    import jieba

    md_path = md_path.expanduser().resolve()
    if not md_path.is_file():
        raise SystemExit(f"未找到文件: {md_path}")
    if md_path.suffix.lower() != ".md":
        raise SystemExit(f"分词模式仅支持 .md: {md_path}")

    text = md_path.read_text(encoding="utf-8", errors="replace")
    out_path = md_path.with_name(f"{md_path.stem}_jieba_seg.md")
    out_path.write_text(sep.join(jieba.cut(text)), encoding="utf-8")
    return out_path


def main() -> None:
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] == "--faiss":
        if len(raw_args) > 2:
            raise SystemExit("用法: python pdf_extract_docling_to_md.py --faiss [某个.md]")
        md_for_faiss = (
            Path(raw_args[1]).expanduser()
            if len(raw_args) == 2
            else DEFAULT_JIEBA_SEG_MD
        )
        _embed_md_to_faiss(md_for_faiss)
        return

    if raw_args and raw_args[0] == "--rag":
        if len(raw_args) > 1:
            raise SystemExit(
                "用法: python pdf_extract_docling_to_md.py --rag\n"
                "自定义问题请用环境变量 RAG_QUERY=...；可选 RAG_TOP_K、RAG_MAX_META_CHARS。"
            )
        q = os.getenv("RAG_QUERY", DEFAULT_RAG_QUERY)
        ans = _rag_answer_with_deepseek(q, DEFAULT_FAISS_PATH, DEFAULT_META_PATH)
        _print_deepseek_answer(ans)
        return

    argv_paths = [Path(a) for a in raw_args]
    if not argv_paths:
        _pipeline_pdf_md_jieba_faiss(DEFAULT_PDF)
        return

    md_paths = [p for p in argv_paths if p.suffix.lower() == ".md"]
    pdf_paths = [p for p in argv_paths if p.suffix.lower() == ".pdf"]
    rest = [
        p
        for p in argv_paths
        if p.suffix.lower() not in (".md", ".pdf")
    ]
    if rest:
        raise SystemExit(
            "仅支持 .pdf（含 FAISS+DeepSeek）或 .md（jieba+FAISS+DeepSeek），多余参数: "
            + ", ".join(str(p) for p in rest)
        )
    if md_paths and pdf_paths:
        raise SystemExit("请勿在同一命令中混传 PDF 与 MD，请分开运行。")

    if md_paths:
        for p in md_paths:
            print(f"正在分词: {p.name} …")
            out = _segment_md_to_jieba_file(p)
            print(f"已写入: {out}")
            print("【3/4】正在对分词结果生成向量并写入 FAISS …")
            faiss_path, meta_path = _embed_md_to_faiss(out)
            print("【4/4】FAISS 检索 → DeepSeek …")
            q = os.getenv("RAG_QUERY", DEFAULT_RAG_QUERY)
            ans = _rag_answer_with_deepseek(q, faiss_path, meta_path)
            _print_deepseek_answer(ans)
        return

    if len(pdf_paths) != 1:
        raise SystemExit("PDF 模式一次只支持一个 .pdf 文件路径。")
    _pipeline_pdf_md_jieba_faiss(pdf_paths[0])


if __name__ == "__main__":
    main()
