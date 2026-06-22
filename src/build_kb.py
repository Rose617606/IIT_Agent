"""知识库构建器 — 加载政策文档 → 切片 → BGE-M3 编码 → 存入 Chroma。

使用方式：
    python -m src.build_kb          # 读取 data/*.md，输出到 knowledge_base/
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.schemas import Chunk, ChunkMeta, DocumentMeta, TaxSubCategory

# BGE-M3: 延迟加载，仅 build_kb() 调用时加载（避免 import 时报错）
_BGE_MODEL = None

_logger = logging.getLogger("build_kb")


# ── 工具函数 ───────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Markdown 开头的 YAML frontmatter，返回 (meta_dict, body_text)。"""
    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return meta, body
    return {}, text


def _resolve_tax_subcategory(heading: str) -> str:
    """根据章节标题映射到 tax_subcategory。"""
    heading = heading.strip().lstrip("#").strip()
    mapping = [
        (["子女教育", "婴幼儿", "3岁以下"], TaxSubCategory.CHILD_EDUCATION),
        (["继续教育"], TaxSubCategory.CONTINUING_EDUCATION),
        (["大病医疗"], TaxSubCategory.MAJOR_MEDICAL),
        (["住房贷款利息", "住房贷款"], TaxSubCategory.HOUSING_LOAN),
        (["住房租金", "租房", "房租"], TaxSubCategory.HOUSING_RENT),
        (["赡养老人", "赡养"], TaxSubCategory.ELDERLY_SUPPORT),
        (["婴幼儿照护", "婴儿"], TaxSubCategory.INFANT_CARE),
        (["年终奖", "一次性奖金", "全年一次性"], TaxSubCategory.ANNUAL_BONUS),
        (["综合所得", "应纳税所得额", "税率"], TaxSubCategory.COMPREHENSIVE_INCOME),
        (["汇算清缴", "年度汇算"], TaxSubCategory.ANNUAL_SETTLEMENT),
        (["免税", "免征", "减征"], TaxSubCategory.BASIC_DEDUCTION),
    ]
    for keywords, category in mapping:
        if any(kw in heading for kw in keywords):
            return category.value
    return TaxSubCategory.COMPREHENSIVE_INCOME.value  # 兜底


# ── 核心函数 ───────────────────────────────────────────

def load_documents(data_dir: str = "data/") -> list[tuple[DocumentMeta, str]]:
    """加载 data/*.md 文件，解析 frontmatter，返回 (DocumentMeta, 正文)。"""
    data_path = Path(data_dir)
    if not data_path.exists() or not list(data_path.glob("*.md")):
        raise FileNotFoundError(f"data/ 目录为空或不存在，请先放入 Markdown 政策文件。路径: {data_path.absolute()}")

    docs = []
    for md_file in sorted(data_path.glob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw_text)
        if not frontmatter:
            _logger.warning("跳过无 frontmatter 的文件: %s", md_file.name)
            continue
        meta = DocumentMeta(**frontmatter)
        docs.append((meta, body))
        _logger.info("已加载: %s (%s)", meta.title, meta.source)
    return docs


def split_by_section(doc_meta: DocumentMeta, body: str) -> list[tuple[str, str, str]]:
    """按 ## 标题拆分文档为节，返回 [(section_title, content, tax_subcategory), ...]。

    Returns:
        三元组列表：(章节标题, 正文, tax_subcategory)
    """
    sections = []
    # 匹配 ## 标题，保留标题作为 section_title
    parts = re.split(r"^(#{1,2}\s+.+)$", body, flags=re.MULTILINE)

    current_title = ""
    current_content: list[str] = []

    for part in parts:
        part = part.rstrip()
        if re.match(r"^#{1,2}\s+", part):
            # 保存上一个 section
            if current_content:
                combined = "\n".join(current_content).strip()
                if combined:
                    subcat = _resolve_tax_subcategory(current_title)
                    sections.append((current_title, combined, subcat))
            current_title = part
            current_content = []
        else:
            if part.strip():
                current_content.append(part)

    # 最后一段
    if current_content:
        combined = "\n".join(current_content).strip()
        if combined:
            subcat = _resolve_tax_subcategory(current_title)
            sections.append((current_title, combined, subcat))

    # 如果没有 ## 标题，整个文档作为一个 section
    if not sections:
        subcat = _resolve_tax_subcategory(doc_meta.title)
        sections.append((doc_meta.title, body.strip(), subcat))

    return sections


def chunk_section(
    section_title: str,
    section_content: str,
    tax_subcategory: str,
    min_chunk_size: int = 30,
    max_chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """按规则原子切分单节，返回 Chunk 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "；", "。", ";", ".", "，", ",", " "],
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    raw_chunks = splitter.split_text(section_content)

    chunks = []
    for i, raw in enumerate(raw_chunks):
        raw = raw.strip()
        if len(raw) < min_chunk_size:
            continue
        # 数字保护：跳过全是数字/符号的重叠片段
        if len(re.sub(r'[^一-鿿]', '', raw)) < 10:
            continue
        # 章节前缀拼接
        prefix = section_title.strip().lstrip("#").strip()
        content = f"[{prefix}] {raw}" if prefix else raw
        chunks.append(Chunk(content=content, meta=ChunkMeta(
            tax_subcategory=tax_subcategory,
            section_title=prefix,
            chunk_index=i,
            # document_source / effective_date 稍后由 build_kb 统一设置
        )))
    return chunks


def build_kb(
    data_dir: str = "data/",
    persist_dir: str = "knowledge_base/",
    model_name: str = "BAAI/bge-m3",
) -> Chroma:
    """主入口：加载文档 → 切片 → BGE-M3 编码 → 入库。

    Args:
        data_dir: 政策 Markdown 文件目录
        persist_dir: Chroma 持久化路径
        model_name: BGE-M3 模型名（HuggingFace hub 或本地路径）

    Returns:
        Chroma vectorstore 实例
    """
    global _BGE_MODEL

    # 1. 延迟加载 BGE-M3（首次加载会自动下载 ~2GB）
    from FlagEmbedding import BGEM3FlagModel
    if _BGE_MODEL is None:
        _logger.info("加载 BGE-M3 模型: %s ...", model_name)
        _BGE_MODEL = BGEM3FlagModel(model_name, use_fp16=True)
        _logger.info("BGE-M3 加载完成")

    # 2. 加载文档
    docs = load_documents(data_dir)
    _logger.info("共加载 %d 份文档", len(docs))

    # 3. 切分
    all_chunks: list[Chunk] = []
    for doc_meta, body in docs:
        sections = split_by_section(doc_meta, body)
        for section_title, section_content, tax_subcategory in sections:
            chunks = chunk_section(section_title, section_content, tax_subcategory)
            for ch in chunks:
                ch.meta.document_source = doc_meta.source
                ch.meta.effective_date = doc_meta.effective_date
                ch.meta.is_expired = (doc_meta.status == "expired")
            all_chunks.extend(chunks)

    _logger.info("共生成 %d 个切片", len(all_chunks))

    if not all_chunks:
        raise RuntimeError("未生成任何切片，请检查 data/ 中的文件内容")

    # 4. BGE-M3 编码
    texts = [ch.content for ch in all_chunks]
    meta_dicts = [ch.meta.model_dump() for ch in all_chunks]
    ids = [ch.meta.chunk_id for ch in all_chunks]

    # 日期字段转字符串（Chroma metadata 要求）
    for md in meta_dicts:
        md["effective_date"] = str(md["effective_date"])

    _logger.info("BGE-M3 编码 %d 个切片 (dense + sparse) ...", len(texts))
    output = _BGE_MODEL.encode(texts, return_dense=True, return_sparse=True, batch_size=32)
    dense_vecs = np.array(output["dense_vecs"], dtype=np.float32)
    sparse_vecs = output.get("lexical_weights", [])

    # 5. 存入 Chroma
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # 清空旧数据
    if list(persist_path.iterdir()):
        import shutil
        shutil.rmtree(persist_path)
        persist_path.mkdir()

    chroma = Chroma(persist_directory=str(persist_path), embedding_function=None)
    # 手动插入预计算的 embedding（Chroma 支持 add_embeddings）
    chroma.add_texts(texts=texts, metadatas=meta_dicts, ids=ids, embeddings=dense_vecs)
    _logger.info("Chroma 入库完成，路径: %s", persist_path.absolute())

    # 6. 存储 Sparse 索引
    sparse_index = {}
    for chunk_id, sw in zip(ids, sparse_vecs):
        # sw 格式: {token_id: weight, ...} 的 dict
        sparse_index[chunk_id] = {str(k): float(v) for k, v in sw.items()}

    sparse_path = persist_path / "sparse_index.json"
    sparse_path.write_text(json.dumps(sparse_index, ensure_ascii=False, indent=2), encoding="utf-8")
    _logger.info("Sparse 索引入库: %s", sparse_path)

    return chroma


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_kb()
    print("知识库构建完成。")
