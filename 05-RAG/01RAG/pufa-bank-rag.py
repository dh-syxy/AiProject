#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""浦发制度 PDF：用 Docling 解析，按文档元素返回文本与所在页码；可选切分、嵌入与 FAISS。

依赖：
  pip install docling
  pip install langchain langchain-community faiss-cpu

问答：DashScope 上的 ``deepseek-v3``（``langchain_community.llms.Tongyi``），需 ``DASHSCOPE_API_KEY``。

用法示例（注意文件名含连字符时需用 importlib 或改名后再 import）：
  python3 pufa-bank-rag.py
  若 ``pufa_bank_faiss/`` 下已有 ``index.faiss``、``index.pkl``、``pageinfo.pkl``，则**不会**再跑
  Docling，只 ``load_faiss_with_pageinfo`` 后 ``query_faiss_deepseek_stuff``（需 ``DASHSCOPE_API_KEY``）。
  强制从 PDF 重建：``PUFA_REBUILD_FAISS=1 python3 pufa-bank-rag.py``
  自定义问题：``PUFA_RAG_QUERY='你的问题' python3 pufa-bank-rag.py``

仅问答（不加载 Docling）：``python3 pufa_query_only.py``
"""

from __future__ import annotations

import os

# macOS 上 faiss 与其它带 OpenMP 的库并存时可能 OMP Error #15 并 abort；见 ``pufa_rag_runtime`` 注释。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

# 须在 import huggingface_hub / docling 之前（与同目录流程脚本一致）
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

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FAISS_DIR = BASE_DIR / "pufa_bank_faiss"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf"


def _faiss_rag_bundle_ready(faiss_dir: Path) -> bool:
    """是否已有可加载的 FAISS 目录（LangChain 默认 ``index.faiss`` / ``index.pkl`` + ``pageinfo.pkl``）。"""
    if not faiss_dir.is_dir():
        return False
    return (
        (faiss_dir / "index.faiss").is_file()
        and (faiss_dir / "index.pkl").is_file()
        and (faiss_dir / "pageinfo.pkl").is_file()
    )


def _primary_page_no(item: Any) -> Optional[int]:
    """从 DocItem 的 provenance 取页码（Docling 为 1-based）。无 provenance 时返回 None。"""
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    return int(prov[0].page_no)


def read_pdf_text_with_pages(
    pdf_path: Union[str, Path],
) -> list[dict[str, Any]]:
    """使用 Docling 读取 PDF，返回若干条记录，每条为一段文本及其页码。

    每条字典包含：
      - ``text`` (str)：该元素的正文（与 Docling ``TextItem.text`` 一致，含标题、段落、列表项等）
      - ``page_no`` (int)：该文本在 PDF 中的页码（从 1 起）

    仅包含带可读正文的 ``TextItem`` 及其子类（标题、章节标题、列表项等）；表格内嵌文本由
    Docling 以结构化节点表示，若需表格可再扩展遍历 ``TableItem``。

    :param pdf_path: PDF 文件路径
    :return: ``[{"text": "...", "page_no": 1}, ...]``
    :raises FileNotFoundError: 文件不存在
    :raises ValueError: 路径不是 ``.pdf``
    """
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import TextItem

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到 PDF: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"需要 .pdf 文件: {path}")

    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document

    out: list[dict[str, Any]] = []
    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue
        text = (item.text or "").strip()
        if not text:
            continue
        page_no = _primary_page_no(item)
        if page_no is None:
            continue
        out.append({"text": text, "page_no": page_no})

    return out


def _rows_to_flat_text_and_char_pages(
    rows: list[dict[str, Any]],
) -> tuple[str, list[int]]:
    """将 ``read_pdf_text_with_pages`` 的输出拼成单一字符串，并为每个字符标注页码。"""
    parts: list[str] = []
    mapping: list[int] = []
    for i, row in enumerate(rows):
        p = int(row["page_no"])
        if i > 0:
            parts.append("\n\n")
            mapping.extend([p, p])
        t = str(row.get("text", "") or "")
        parts.append(t)
        mapping.extend([p] * len(t))
    return "".join(parts), mapping

def _page_for_document(page_info: dict[str, Any], doc: Any) -> Optional[int]:
    """用 ``page_content`` 在 ``page_info`` 中解析页码（兼容 strip 与键不完全一致）。"""
    raw = getattr(doc, "page_content", "") or ""
    text = raw.strip()
    if text in page_info:
        return int(page_info[text])
    if raw in page_info:
        return int(page_info[raw])
    for k, v in page_info.items():
        if isinstance(k, str) and k.strip() == text:
            return int(v)
    return None

def _mode_page(pages: list[int]) -> int:
    """页码列表的众数；平局时取较小页码以便稳定。"""
    if not pages:
        return 1
    cnt = Counter(pages)
    best_n = max(cnt.values())
    candidates = [p for p, n in cnt.items() if n == best_n]
    return min(candidates)


def _chunk_spans_in_full_text(
    full_text: str,
    chunks: list[str],
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """根据与分割器一致的滑动起点，在原文中定位每个 chunk 的 ``[start, end)``。"""
    spans: list[tuple[int, int]] = []
    search_lo = 0
    for i, chunk in enumerate(chunks):
        pos = full_text.find(chunk, search_lo)
        if pos < 0:
            pos = full_text.find(chunk)
        if pos < 0:
            raise ValueError(f"无法在原文中定位第 {i} 个文本块，请检查分割结果与原文一致性。")
        end = pos + len(chunk)
        spans.append((pos, end))
        search_lo = pos + len(chunk) - chunk_overlap
    return spans


def build_faiss_from_read_pdf_rows(
    rows: list[dict[str, Any]],
    *,
    save_dir: Optional[Union[str, Path]] = None,
    dashscope_api_key: Optional[str] = None,
) -> Any:
    """对 ``read_pdf_text_with_pages`` 的输出做递归切分、DashScope 嵌入并建立 FAISS 索引。

    1. 将各条 ``text`` 用 ``\\n\\n`` 连接成全文，并建立逐字符页码序列；再用递归字符分割器切分
       （chunk_size=1000，chunk_overlap=200）。分隔符优先级：段落 ``\\n\\n``、换行 ``\\n``、
       句末标点、英文句号、空格、单字符。
    2. 对每个 chunk，用其在原文中的字符区间截取页码序列，取**众数**作为该块页码（平局取较小页码）。
    3. 使用 ``text-embedding-v1`` 向量化，写入 FAISS；调用 ``save_local`` 保存向量库，并将
       ``chunk_text_to_page`` 与 ``pages_by_index`` 写入 ``pageinfo.pkl``（与向量顺序一致）。

    :param rows: ``read_pdf_text_with_pages`` 的返回值
    :param save_dir: 持久化目录；``None`` 时使用 ``BASE_DIR / "pufa_bank_faiss"``
    :param dashscope_api_key: 默认读环境变量 ``DASHSCOPE_API_KEY``
    :return: ``langchain_community.vectorstores.FAISS`` 实例，并设置 ``page_info``（``dict[chunk, page]``）
    """
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.vectorstores import FAISS

    if not rows:
        raise ValueError("rows 为空，无法建立索引。")

    api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或传入 dashscope_api_key。")

    chunk_size = 1000
    chunk_overlap = 200
    separators = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        ".",
        " ",
        "",
    ]
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    full_text, char_pages = _rows_to_flat_text_and_char_pages(rows)
    if not full_text.strip():
        raise ValueError("拼接后全文为空。")

    chunks = splitter.split_text(full_text)
    spans = _chunk_spans_in_full_text(full_text, chunks, chunk_overlap)

    page_info: dict[str, int] = {}
    pages_by_index: list[int] = []
    for chunk, (start, end) in zip(chunks, spans):
        slice_pages = char_pages[start:end]
        page_no = _mode_page(slice_pages)
        page_info[chunk] = page_no
        pages_by_index.append(page_no)

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=api_key,
    )
    store = FAISS.from_texts(chunks, embeddings)
    store.page_info = page_info

    out_dir = (
        Path(save_dir).expanduser().resolve()
        if save_dir is not None
        else (BASE_DIR / "pufa_bank_faiss")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))
    payload = {
        "chunk_text_to_page": page_info,
        "pages_by_index": pages_by_index,
    }
    with open(out_dir / "pageinfo.pkl", "wb") as f:
        pickle.dump(payload, f)

    return store

def load_faiss_with_pageinfo(
    faiss_dir: Union[str, Path],
    *,
    dashscope_api_key: Optional[str] = None,
) -> Any:
    """从 ``build_faiss_from_read_pdf_rows`` 写入的目录加载 FAISS，并挂载 ``pageinfo.pkl`` 中的页码映射。"""
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.vectorstores import FAISS

    api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或传入 dashscope_api_key。")

    path = Path(faiss_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"未找到 FAISS 目录: {path}")

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=api_key,
    )
    store = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    pkl = path / "pageinfo.pkl"
    if pkl.is_file():
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "chunk_text_to_page" in data:
            store.page_info = data["chunk_text_to_page"]
        else:
            store.page_info = data
    else:
        store.page_info = {}
    return store

def query_faiss_deepseek_stuff(
    question: str,
    vectorstore: Any,
    *,
    top_k: int = 10,
    dashscope_api_key: Optional[str] = None,
    print_answer: bool = True,
    print_sources: bool = True,
) -> dict[str, Any]:
    """基于已构建的 FAISS 库做 RAG 问答（检索 top_k → stuff 链 → DeepSeek-V3）。

    1. 使用与建库相同的嵌入在 ``vectorstore`` 中做相似度检索，返回 top ``top_k`` 条文档。
    2. 使用 ``langchain.chains.question_answering.load_qa_chain``、``chain_type="stuff"``，
       通过 DashScope 的 ``Tongyi(model_name="deepseek-v3")`` 将合并上下文与问题发给模型。
    3. 根据 ``vectorstore.page_info``（或 ``pageinfo.pkl`` 加载结果）解析每条检索文档的页码，
       去重排序后打印并写入返回值。

    :param question: 用户问题
    :param vectorstore: ``build_faiss_from_read_pdf_rows`` 的返回值，或 ``load_faiss_with_pageinfo`` 的返回值
    :param top_k: 检索条数，默认 10
    :param dashscope_api_key: 默认读 ``DASHSCOPE_API_KEY``（嵌入与 LLM 共用百炼 Key）
    :param print_answer: 是否打印模型回答
    :param print_sources: 是否打印来源页码
    :return: ``{"answer": str, "source_pages": list[int], "retrieved_docs": list}``
    """
    from langchain.chains.question_answering import load_qa_chain
    from langchain_community.llms import Tongyi

    if not (question or "").strip():
        raise ValueError("question 不能为空。")

    api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或传入 dashscope_api_key。")

    docs = vectorstore.similarity_search(question.strip(), k=top_k)

    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.invoke({"input_documents": docs, "question": question.strip()})
    answer = response.get("output_text") or response.get("text") or ""

    page_info: dict[str, Any] = getattr(vectorstore, "page_info", None) or {}
    per_doc_pages: list[int] = []
    for d in docs:
        pg = _page_for_document(page_info, d)
        if pg is not None:
            per_doc_pages.append(pg)
    source_pages = sorted(set(per_doc_pages))

    if print_answer:
        print(answer)
    if print_sources:
        if source_pages:
            print("来源页码:", source_pages)
        else:
            print("来源页码: 未能从 page_info 解析（请确认已加载 pageinfo.pkl 或建库时写入 page_info）。")

    return {
        "answer": answer,
        "source_pages": source_pages,
        "retrieved_docs": docs,
        "source_pages_per_doc": per_doc_pages,
    }


if __name__ == "__main__":
    # 默认示例问题；可用环境变量 PUFA_RAG_QUERY 覆盖
    _default_demo_query = "客户经理每年评聘申报时间是怎样的？"
    demo_q = os.getenv("PUFA_RAG_QUERY", _default_demo_query).strip()
    force_rebuild = os.getenv("PUFA_REBUILD_FAISS", "").strip() in ("1", "true", "yes")

    key = os.getenv("DASHSCOPE_API_KEY")
    out = DEFAULT_FAISS_DIR
    rag_ready = _faiss_rag_bundle_ready(out) and not force_rebuild

    if not rag_ready:
        pdf = DEFAULT_PDF
        rows = read_pdf_text_with_pages(pdf)
        vb = build_faiss_from_read_pdf_rows(rows, save_dir=out)
        print(f"已加载，向量数={vb.index.ntotal}")
        query_faiss_deepseek_stuff(demo_q, vb, top_k=10)
    else:
        vb = load_faiss_with_pageinfo(out)
        print(f"FAISS 已构建，向量数={vb.index.ntotal}，已保存到 {out}")
        query_faiss_deepseek_stuff(demo_q, vb, top_k=10)
