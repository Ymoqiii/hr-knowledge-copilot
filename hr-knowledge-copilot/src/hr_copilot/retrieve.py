"""Keyword retrieval baseline using only the Python standard library.

Score is transparent: TF * IDF over interpretable tokens, plus a phrase bonus
when the full query string appears in the chunk. This is not semantic search;
synonyms (年假 vs 年休假) will miss unless both strings appear.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from hr_copilot.models import Chunk, QueryError, SearchHit, SearchResult

_LATIN = re.compile(r"[a-z0-9]+")
_CJK_BLOCK = re.compile(r"[\u4e00-\u9fff]+")

# Common conversational/document boilerplate that otherwise creates broad
# bigram matches across unrelated policy chunks.
_CJK_STOP = {
    "演示", "数据", "规定", "管理", "公司", "员工", "相关", "根据", "关于",
    "以及", "可以", "需要", "必须", "进行", "提供", "工作", "一个", "这个",
}

# Query glue words that rarely appear as policy terms. Kept tiny on purpose.
_STOP = {
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "吧",
    "请",
    "请问",
    "如何",
    "怎么",
    "什么",
    "一下",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "and",
    "is",
    "are",
}


def search_chunks(
    chunks: Sequence[Chunk],
    query: str,
    top_k: int = 3,
) -> SearchResult:
    if top_k <= 0:
        raise QueryError("top_k must be positive.")

    cleaned = query.strip()
    if not cleaned:
        raise QueryError("Empty question. Type a question about a policy, for example: 年假如何申请")

    if not chunks:
        return SearchResult(
            query=cleaned,
            hits=(),
            message="No indexed chunks. Add TXT/PDF files under the demo data folder and ingest again.",
        )

    query_terms = tokenize(cleaned)
    if not query_terms:
        return SearchResult(
            query=cleaned,
            hits=(),
            message="No searchable keywords in the question after removing filler words.",
        )

    document_tokens = [tokenize(chunk.text) for chunk in chunks]
    df: Counter[str] = Counter()
    for tokens in document_tokens:
        df.update(set(tokens))
    n_docs = len(chunks)

    scored: list[SearchHit] = []
    lowered_query = cleaned.lower()
    for chunk, tokens in zip(chunks, document_tokens, strict=True):
        tf = Counter(tokens)
        matched: list[str] = []
        tfidf = 0.0
        parts: list[str] = []
        for term in dict.fromkeys(query_terms):
            count = tf.get(term, 0)
            if count <= 0:
                continue
            idf = math.log((n_docs + 1) / (df[term] + 1)) + 1.0
            contrib = count * idf
            tfidf += contrib
            matched.append(term)
            parts.append(f"{term}(tf={count},idf={idf:.2f})")

        # A phrase match may boost a result, but must not create a hit on its
        # own: filler-heavy questions should return no results when no policy
        # keyword matches.
        phrase_bonus = 0.0
        if matched and lowered_query in chunk.text.lower():
            phrase_bonus = 2.0
            parts.append("phrase_bonus=2.00")

        score = tfidf + phrase_bonus
        if score <= 0:
            continue
        explanation = " + ".join(parts) + f" => {score:.3f}"
        scored.append(
            SearchHit(
                chunk=chunk,
                score=round(score, 4),
                matched_terms=tuple(matched),
                explanation=explanation,
            )
        )

    scored.sort(key=lambda hit: (-hit.score, hit.chunk.source_name, hit.chunk.chunk_id))
    top = tuple(scored[:top_k])
    if not top:
        return SearchResult(
            query=cleaned,
            hits=(),
            message="No matching policy snippet. The knowledge base may not cover this question.",
        )
    return SearchResult(query=cleaned, hits=top, message=f"Top {len(top)} chunk(s).")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    for word in _LATIN.findall(lowered):
        if word not in _STOP:
            tokens.append(word)
    for block in _CJK_BLOCK.findall(lowered):
        if len(block) == 1:
            if block not in _STOP:
                tokens.append(block)
            continue
        for index in range(len(block) - 1):
            gram = block[index : index + 2]
            if gram not in _STOP and gram not in _CJK_STOP:
                tokens.append(gram)
    return tokens
