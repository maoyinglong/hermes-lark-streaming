"""Markdown 文本处理 — 标题降级、表格降级、图片 key 剥离、长文本分块."""

from __future__ import annotations

import logging
import re

_logger = logging.getLogger("hermes_lark_streaming")

_MAX_CARD_TABLES = 5
_MAX_CHUNK_CHARS = 2400

__all__ = [
    "_downgrade_tables",
    "_find_tables_outside_code_blocks",
    "_normalize_indented_blockquotes",
    "_sanitize_math_symbols",
    "_split_long_text",
    "_strip_invalid_image_keys",
    "optimize_markdown_style",
]


def _find_tables_outside_code_blocks(text: str) -> list[tuple[int, int, str]]:
    """查找代码块外的 markdown 表格，返回 [(start, end, raw), ...]."""
    code_ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"```[\s\S]*?```", text):
        code_ranges.append((m.start(), m.end()))

    def _in_code(idx: int) -> bool:
        return any(s <= idx < e for s, e in code_ranges)

    results: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\|.+\|\n\|[-:| ]+\|[\s\S]*?(?=\n\n|\n(?!\|)|$)", text):
        if not _in_code(m.start()):
            results.append((m.start(), m.end(), m.group(0)))
    return results


def _downgrade_tables(text: str, limit: int = _MAX_CARD_TABLES) -> str:
    """超限表格降级为代码块（保留内容可见但飞书不渲染为表格元素）."""
    matches = _find_tables_outside_code_blocks(text)
    if len(matches) <= limit:
        return text
    result = text
    for start, end, raw in reversed(matches[limit:]):
        replacement = f"```\n{raw}\n```"
        result = result[:start] + replacement + result[end:]
    return result


def _strip_invalid_image_keys(text: str) -> str:
    """移除非 img_ 前缀的图片引用."""
    if "![" not in text:
        return text

    def _replace(m: re.Match) -> str:
        return m.group(0) if m.group(2).startswith("img_") else ""

    return re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", _replace, text)


def _sanitize_math_symbols(text: str) -> str:
    """清洗飞书 CardKit 不支持的 LaTeX 箭头及数学符号，转为纯文本 Unicode 符号."""
    text = re.sub(r"\$(?:\\to|\\rightarrow)\$", "→", text)
    text = re.sub(r"(?<!\\)\\(?:to|rightarrow)\b", "→", text)
    text = re.sub(r"\$(?:\\gets|\\leftarrow)\$", "←", text)
    text = re.sub(r"(?<!\\)\\(?:gets|leftarrow)\b", "←", text)
    text = re.sub(r"\$\\leftrightarrow\$", "↔", text)
    text = re.sub(r"\$\\Rightarrow\$", "⇒", text)
    text = re.sub(r"\$\\Leftarrow\$", "⇐", text)
    text = re.sub(r"\$\\Leftrightarrow\$", "⇔", text)
    return text


def _normalize_indented_blockquotes(text: str) -> str:
    """将嵌套在列表或缩进中的引用块 (如 '   > text') 顶格化为标准引用 ('> text').

    飞书 CardKit 富文本组件不支持缩进的 blockquote，遇到行首带空格的 '>' 会导致 AST 解析失败而静默吞掉内容。
    将其规整为顶格的独立引用块，并确保前后有换行分隔。
    """
    lines = text.split("\n")
    new_lines: list[str] = []
    in_quote = False

    for line in lines:
        m = re.match(r"^[ \t]+>(?:[ \t]?(.*))?$", line)
        if m:
            content = m.group(1) if m.group(1) is not None else ""
            if not in_quote:
                if new_lines and new_lines[-1].strip() != "":
                    new_lines.append("")
                in_quote = True
            new_lines.append(f"> {content}".rstrip() if content else ">")
        else:
            if in_quote:
                if line.strip() != "":
                    new_lines.append("")
                in_quote = False
            new_lines.append(line)

    return "\n".join(new_lines)


def optimize_markdown_style(text: str) -> str:
    """优化流式 Markdown 以适配飞书 CardKit 渲染.

    1. 提取代码块用占位符保护
    2. 标题降级: H1 -> H4, H2-H6 -> H5
    3. 数学符号与 LaTeX 消毒（如 $\\to$ -> →）
    4. 缩进引用块顶格规整（防飞书 CardKit AST 吞字）
    5. 压缩多余空行
    6. 还原代码块
    7. 剥离无效图片 key（非 img_xxx 格式）
    """
    try:
        # 1. 提取代码块
        mark = "___CB_"
        code_blocks: list[str] = []

        def _extract(m: re.Match) -> str:
            prefix = m.group(1) or ""
            block = m.group(0)[len(prefix) :]
            idx = len(code_blocks)
            code_blocks.append(block)
            return f"{prefix}{mark}{idx}___"

        r = re.sub(
            r"(^|\n)( {0,3})(`{3,}|~{3,})([^\n]*)\n[\s\S]*?\n {0,3}\3[ \t]*(?=\n|$)",
            _extract,
            text,
        )

        # 2. 标题降级（仅当存在 H1-H3 时）
        if re.search(r"^#{1,3} ", text, re.MULTILINE):
            r = re.sub(r"^#{2,6} (.+)$", r"##### \1", r, flags=re.MULTILINE)
            r = re.sub(r"^# (.+)$", r"#### \1", r, flags=re.MULTILINE)

        # 3. 数学符号与 LaTeX 消毒
        r = _sanitize_math_symbols(r)

        # 4. 缩进引用块顶格规整
        r = _normalize_indented_blockquotes(r)

        # 5. Compress prose only; blank lines inside code are significant.
        r = re.sub(r"\n{3,}", "\n\n", r)

        # 6. Restore code after prose normalization.
        for i, block in enumerate(code_blocks):
            r = r.replace(f"{mark}{i}___", block)

        # 7. 剥离无效图片 key
        r = _strip_invalid_image_keys(r)

        return r
    except Exception:
        _logger.debug("optimize_markdown_style failed", exc_info=True)
        return text


def _split_long_text(text: str, limit: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split independent Markdown elements, counting continuation markup in limit.

    Preserve source whitespace. Fences are closed/reopened and tables repeat their
    header. Inline constructs are atomic when they fit; oversized constructs/rows
    are shown as fenced source rather than emitting broken Markdown. This is not
    a full CommonMark parser (nested lists/blockquote fences are not supported).
    """
    if limit < 32:
        raise ValueError("Markdown chunk limit must be at least 32 characters")
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    pending = ""

    def flush() -> None:
        nonlocal pending
        if pending:
            chunks.append(pending)
            pending = ""

    def add(atom: str) -> None:
        nonlocal pending
        if len(pending) + len(atom) > limit:
            flush()
        pending += atom

    def fenced(body: str, opener: str, closer: str, final_closer: str | None = None) -> None:
        flush()
        final_closer = closer if final_closer is None else final_closer
        # Pathological info strings/fence runs cannot fit with any payload.
        overhead = len(opener) + max(len(closer), len(final_closer)) + 1
        if overhead >= limit:
            literal(opener + body + final_closer)
            return
        if not body:
            chunks.append(opener + final_closer)
        while body:
            capacity = limit - overhead
            cut = min(len(body), capacity)
            if cut < len(body):
                newline = body.rfind("\n", 0, cut)
                if newline >= capacity // 2:
                    cut = newline + 1
            piece, body = body[:cut], body[cut:]
            tail = closer if body else final_closer
            chunks.append(opener + piece + ("" if piece.endswith("\n") else "\n") + tail)

    def literal(source: str) -> None:
        # Choose a delimiter absent from the source, so even huge links or table
        # rows remain visible without accidentally starting Markdown structures.
        runs = [len(m.group()) for m in re.finditer(r"`+", source)]
        fence = "`" * max(3, max(runs, default=0) + 1)
        if 2 * len(fence) + 3 >= limit:
            flush()
            chunks.extend(source[i : i + limit] for i in range(0, len(source), limit))
        else:
            fenced(source, fence + "\n", fence)

    def prose(source: str) -> None:
        # Balanced link destinations (including escaped/nested parentheses),
        # inline code runs, emphasis, strike-through, and HTML/autolink tags.
        inline = re.compile(
            r"(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)"
            r"|!?\[(?:\\.|[^\]\\])*\]\("
            r"|(?<!\\)(\*\*|__|~~|\*|_)(?=\S)[\s\S]*?\2"
            r"|<[^<>\n]+>"
            r"|\\.",
        )
        pos = 0
        while pos < len(source):
            match = inline.search(source, pos)
            end = match.start() if match else len(source)
            # Non-markup text is freely splittable; prefer complete lines/words.
            while pos < end:
                capacity = limit - len(pending)
                if not capacity:
                    flush()
                    capacity = limit
                cut = min(end, pos + capacity)
                if cut < end:
                    boundary = max(source.rfind("\n", pos, cut), source.rfind(" ", pos, cut))
                    if boundary >= pos + capacity // 2:
                        cut = boundary + 1
                add(source[pos:cut])
                pos = cut
            if not match:
                break
            end = match.end()
            if source[end - 1 : end] == "(" and "](" in match.group():
                depth = 1
                while end < len(source) and depth:
                    if source[end] == "\\":
                        end = min(end + 2, len(source))
                        continue
                    depth += (source[end] == "(") - (source[end] == ")")
                    end += 1
            atom = source[pos:end]
            if len(atom) > limit:
                literal(atom)
            else:
                add(atom)
            pos = end

    lines = text.splitlines(keepends=True)
    fence_re = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*$")
    separator = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    i = 0
    plain: list[str] = []
    while i < len(lines):
        opening = fence_re.match(lines[i])
        table = i + 1 < len(lines) and "|" in lines[i] and separator.match(lines[i + 1])
        if not opening and not table:
            plain.append(lines[i])
            i += 1
            continue
        prose("".join(plain))
        plain = []
        if opening:
            marker = opening.group(1)
            if marker[0] == "`" and "`" in opening.group(2):
                plain.append(lines[i])
                i += 1
                continue
            start = i
            i += 1
            close_re = re.compile(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*[\r\n]*$")
            while i < len(lines) and not close_re.match(lines[i]):
                i += 1
            closed = i < len(lines)
            stop = i + int(closed)
            raw = "".join(lines[start:stop])
            if closed and len(raw) <= limit:
                add(raw)
            else:
                fenced(
                    "".join(lines[start + 1 : i]),
                    lines[start].rstrip("\r\n") + "\n",
                    marker,
                    lines[i] if closed else marker,
                )
            i = stop
        else:
            flush()
            header = lines[i] + lines[i + 1]
            i += 2
            rows: list[str] = []
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                rows.append(lines[i])
                i += 1
            if len(header) >= limit or any(len(header) + len(row) > limit for row in rows):
                literal(header + "".join(rows))
                continue
            pending = header
            for row in rows:
                if len(pending) + len(row) > limit:
                    flush()
                    pending = header
                pending += row
            flush()
    prose("".join(plain))
    flush()
    return chunks
