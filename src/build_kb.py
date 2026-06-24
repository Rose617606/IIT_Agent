"""知识库构建器 — 加载政策文档 → 切片 → BGE-M3 编码 → 存入 Supabase pgvector。

使用方式：
    python -m src.build_kb          # 读取 data/*.md，输出到 Supabase pgvector
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.schemas import Chunk, ChunkMeta, DocumentMeta, TaxSubCategory

# BGE-M3: 延迟加载，仅 build_kb() 调用时加载（避免 import 时报错）
_BGE_MODEL = None

# BGE-M3 dense 向量维度
EMBEDDING_DIM = 1024

_logger = logging.getLogger("build_kb")


# ── 工具函数 ───────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Markdown 开头的 YAML frontmatter，返回 (meta_dict, body_text)。
    容错：source 为空时默认 ""，effective_date 为空时默认 date(1900,1,1)。
    """
    from datetime import date as date_cls
    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            if not meta.get("source"):
                meta["source"] = ""
            if not meta.get("effective_date"):
                meta["effective_date"] = date_cls(1900, 1, 1)
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
    return TaxSubCategory.COMPREHENSIVE_INCOME.value


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
    """按 ## 标题拆分文档为节，返回 [(section_title, content, tax_subcategory), ...]。"""
    sections = []
    parts = re.split(r"^(#{1,2}\s+.+)$", body, flags=re.MULTILINE)

    current_title = ""
    current_content: list[str] = []

    for part in parts:
        part = part.rstrip()
        if re.match(r"^#{1,2}\s+", part):
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

    if current_content:
        combined = "\n".join(current_content).strip()
        if combined:
            subcat = _resolve_tax_subcategory(current_title)
            sections.append((current_title, combined, subcat))

    if not sections:
        subcat = _resolve_tax_subcategory(doc_meta.title)
        sections.append((doc_meta.title, body.strip(), subcat))

    return sections


def _detect_content_type(content: str) -> str:
    """检测内容类型，返回 'case' | 'qa' | 'table' | 'law' | 'general'。"""
    lines = content.split("\n")
    # 表格类：表格行占比 > 30%
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    if table_lines > len(lines) * 0.3:
        return "table"
    # 案例类：含【案例X】标记
    if re.search(r"【案例[一二三四五六七八九十\d]+】", content):
        return "case"
    # 问答类：含 数字、标题 的 Q&A 结构
    if re.search(r"^#\s+[一二三四五六七八九十\d]+[,、]", content, re.MULTILINE) and len(content) > 2000:
        return "qa"
    # 法律条文体：含大量"第X条"
    if re.search(r"第[一二三四五六七八九十百\d]+条", content):
        return "law"
    return "general"


def _extract_tables(content: str) -> tuple[str, list[str]]:
    """从内容中提取 Markdown 表格，用占位符替换。
    返回 (替换后的内容, 表格列表)。"""
    lines = content.split("\n")
    tables: list[str] = []
    result_lines: list[str] = []
    in_table = False
    table_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        # 检测表格行：以 | 开头且包含分隔符 |---|
        is_table_line = bool(stripped.startswith("|") and "|" in stripped[1:])
        is_separator = bool(re.match(r"^\|[\s\-:|]+\|$", stripped))

        if is_table_line or is_separator:
            in_table = True
            table_buffer.append(line)
        else:
            if in_table:
                # 表格结束，保存
                tables.append("\n".join(table_buffer))
                table_buffer = []
                in_table = False
                result_lines.append(f"{{TABLE_{len(tables) - 1}}}")
            result_lines.append(line)

    # 收尾：最后一段如果是表格
    if in_table and table_buffer:
        tables.append("\n".join(table_buffer))
        result_lines.append(f"{{TABLE_{len(tables) - 1}}}")

    return "\n".join(result_lines), tables


def _clean_content(content: str) -> str:
    """基础清洗：去多余空行、统一换行、去页脚类噪声。"""
    # 合并 3 个以上连续换行为 2 个
    content = re.sub(r"\n{3,}", "\n\n", content)
    # 去掉只有空白符的空行
    content = re.sub(r"\n[ \t]+\n", "\n\n", content)
    # 去常见的重复分隔线
    content = re.sub(r"_{10,}", "", content)
    return content.strip()


def _table_chunk_to_text(table_str: str) -> str:
    """将 Markdown 表格转为结构化自然语言描述。"""
    lines = [l.strip() for l in table_str.split("\n") if l.strip()]
    if len(lines) < 2:
        return table_str

    # 解析表头
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    # 跳过分隔行
    data_start = 1
    if re.match(r"^\|[\s\-:|]+\|$", lines[1]):
        data_start = 2

    data_rows = lines[data_start:]
    # 转成自然语言：表头1: 值1，表头2: 值2；...
    descriptions = []
    for row in data_rows:
        if not row.startswith("|"):
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        parts = []
        for h, c in zip(headers, cells):
            if c and c != "-":
                parts.append(f"{h}：{c}")
        if parts:
            descriptions.append("；".join(parts))
    return "。\n".join(descriptions) if descriptions else table_str


def chunk_section(
    section_title: str,
    section_content: str,
    tax_subcategory: str,
    min_chunk_size: int = 30,
    max_chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """按规则原子切分单节 — 表格保护 + 分类型 + 清洗 + 兜底递归切分。

    策略（§4.1-4.3 + §6.3 + §9）：
    1. 基础清洗：去多余空行、统一换行
    2. 表格保护：先提取表格 → 非表格部分切分 → 表格整体作为独立 chunk
    3. 表格内容转自然语言保留表头语义
    4. 分类型：QA 按问答边界切，案例按【案例X】边界切，法律条文依赖 section 切
    5. 兜底：超长段落用 RecursiveCharacterTextSplitter
    """
    prefix = section_title.strip().lstrip("#").strip()

    # ── 1. 清洗 ──
    content = _clean_content(section_content)
    if len(content) < min_chunk_size:
        return []

    # ── 2. 表格保护 ──
    content_no_tables, tables = _extract_tables(content)

    chunks: list[Chunk] = []
    chunk_idx = 0

    def _make_chunk(text: str, subcat: str, idx: int) -> Chunk:
        text = text.strip()
        full = f"[{prefix}] {text}" if prefix else text
        return Chunk(content=full, meta=ChunkMeta(
            tax_subcategory=subcat,
            section_title=prefix,
            chunk_index=idx,
        ))

    # ── 3. 表格转为独立 chunk（自然语言形式） ──
    for table_str in tables:
        if not table_str.strip():
            continue
        # 转自然语言描述
        nl_text = _table_chunk_to_text(table_str)
        chunks.append(_make_chunk(nl_text, tax_subcategory, chunk_idx))
        chunk_idx += 1

    # ── 4. 分类型切分非表格内容 ──
    content_type = _detect_content_type(content_no_tables)
    _logger.debug("内容类型: %s, 标题: %s", content_type, prefix)

    if content_type in ("case", "qa"):
        # 案例：按【案例X】边界拆分，每个案例 + 温馨提示为一块
        segments = re.split(r"\n(?=【案例[一二三四五六七八九十\d]+】)", content_no_tables)
        for seg in segments:
            seg = seg.strip()
            if len(seg) < min_chunk_size:
                continue
            if len(seg) <= max_chunk_size:
                chunks.append(_make_chunk(seg, tax_subcategory, chunk_idx))
                chunk_idx += 1
            else:
                # 案例太长时仅做段落切分（不切碎故事）
                parts = seg.split("\n\n")
                for part in parts:
                    part = part.strip()
                    if len(part) < min_chunk_size:
                        continue
                    if len(part) <= max_chunk_size:
                        chunks.append(_make_chunk(part, tax_subcategory, chunk_idx))
                        chunk_idx += 1
                    else:
                        # 兜底递归切分
                        splitter = RecursiveCharacterTextSplitter(
                            separators=["\n\n", "\n", "；", "。"],
                            chunk_size=max_chunk_size, chunk_overlap=chunk_overlap,
                            length_function=len,
                        )
                        for sub in splitter.split_text(part):
                            chunks.append(_make_chunk(sub, tax_subcategory, chunk_idx))
                            chunk_idx += 1
    elif content_type == "law":
        # 法律条文：已在 split_by_section 按「第X条」分割，此处只对过长段落兜底
        if len(content_no_tables) <= max_chunk_size:
            chunks.append(_make_chunk(content_no_tables, tax_subcategory, chunk_idx))
            chunk_idx += 1
        else:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "；", "。", ";", "."],
                chunk_size=max_chunk_size, chunk_overlap=chunk_overlap,
                length_function=len,
            )
            for sub in splitter.split_text(content_no_tables):
                chunks.append(_make_chunk(sub, tax_subcategory, chunk_idx))
                chunk_idx += 1
    else:
        # general / table: 递归切分
        if len(content_no_tables.strip()) < min_chunk_size:
            pass  # 跳过纯空白
        else:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "；", "。", ";", ".", "，", ",", " "],
                chunk_size=max_chunk_size, chunk_overlap=chunk_overlap,
                length_function=len,
            )
            for sub in splitter.split_text(content_no_tables):
                sub = sub.strip()
                if len(sub) < min_chunk_size:
                    continue
                if len(re.sub(r'[^一-鿿]', '', sub)) < 10:
                    continue
                chunks.append(_make_chunk(sub, tax_subcategory, chunk_idx))
                chunk_idx += 1

    return chunks


# ── pgvector 数据库操作 ───────────────────────────────

async def _init_db(database_url: str):
    """初始化 pgvector 扩展和 chunks 表。"""
    import asyncpg
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)

        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tax_subcategory TEXT,
                document_source TEXT,
                effective_date DATE,
                is_expired BOOLEAN DEFAULT FALSE,
                section_title TEXT,
                chunk_index INTEGER,
                embedding vector({EMBEDDING_DIM}),
                sparse_embedding JSONB
            )
        """)
        _logger.info("pgvector 表 chunks 就绪")
    finally:
        await conn.close()


async def _store_chunks(database_url: str, chunks: list[Chunk], texts: list[str],
                        dense_vecs: np.ndarray, sparse_vecs: list[dict]):
    """批量写入切片和向量到 pgvector。"""
    import asyncpg
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(database_url)
    try:
        await register_vector(conn)

        # 清空旧数据
        await conn.execute("DELETE FROM chunks")
        _logger.info("已清空旧数据，开始写入 %d 条切片...", len(chunks))

        # 批量插入
        rows = []
        for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
            meta = chunk.meta
            rows.append((
                meta.chunk_id,
                chunk.content,
                meta.tax_subcategory,
                meta.document_source,
                meta.effective_date,
                meta.is_expired,
                meta.section_title,
                meta.chunk_index,
                dense.tolist(),
                json.dumps(
                    {str(k): round(float(v), 6) for k, v in sparse.items()}
                ) if sparse else "{}",
            ))

        await conn.executemany("""
            INSERT INTO chunks (id, content, tax_subcategory, document_source,
                               effective_date, is_expired, section_title, chunk_index,
                               embedding, sparse_embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, rows)

        count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        _logger.info("pgvector 入库完成，共 %d 条", count)

        # 数据量够大时建 IVF 索引加速检索
        if count >= 500:
            _logger.info("创建 IVF 索引...")
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {min(count // 10, 200)})
            """)
            _logger.info("索引创建完成")
    finally:
        await conn.close()


# ── 入口 ───────────────────────────────────────────────

def build_kb(
    data_dir: str = "data/",
    database_url: str | None = None,
    model_name: str = "BAAI/bge-m3",
):
    """主入口：加载文档 → 切片 → BGE-M3 编码 → 入库 pgvector。

    Args:
        data_dir: 政策 Markdown 文件目录
        database_url: Supabase PostgreSQL 连接串（默认从环境变量 DATABASE_URL 读取）
        model_name: BGE-M3 模型名
    """
    global _BGE_MODEL

    # 数据库连接
    if database_url is None:
        from dotenv import load_dotenv
        load_dotenv()
        database_url = os.environ["DATABASE_URL"]
    if not database_url:
        raise RuntimeError("未配置 DATABASE_URL，请在 .env 中设置 Supabase 连接串")

    # 1. 初始化 pgvector 表
    asyncio.run(_init_db(database_url))

    # 2. 加载 BGE-M3（优先本地路径，否则从 HuggingFace 下载）
    from FlagEmbedding import BGEM3FlagModel
    if _BGE_MODEL is None:
        local_path = os.environ.get("BGE_LOCAL", "")
        if local_path and Path(local_path).exists():
            _logger.info("加载本地 BGE-M3: %s ...", local_path)
            _BGE_MODEL = BGEM3FlagModel(local_path, use_fp16=True)
        else:
            _logger.info("加载 BGE-M3 模型: %s ...", model_name)
            _BGE_MODEL = BGEM3FlagModel(model_name, use_fp16=True)
        _logger.info("BGE-M3 加载完成")

    # 3. 加载文档
    docs = load_documents(data_dir)
    _logger.info("共加载 %d 份文档", len(docs))

    # 4. 切分
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

    # 5. BGE-M3 编码
    texts = [ch.content for ch in all_chunks]
    _logger.info("BGE-M3 编码 %d 个切片 (dense + sparse) ...", len(texts))
    output = _BGE_MODEL.encode(texts, return_dense=True, return_sparse=True, batch_size=32)
    dense_vecs = np.array(output["dense_vecs"], dtype=np.float32)
    sparse_vecs = output.get("lexical_weights", [])

    # 6. 存入 pgvector
    asyncio.run(_store_chunks(database_url, all_chunks, texts, dense_vecs, sparse_vecs))

    _logger.info("知识库构建完成！")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_kb()
    print("知识库构建完成。")
