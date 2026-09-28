"""AST Patcher — 在 Hermes gateway/run.py 中注入 Hook 调用."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import hermes_home

_logger = logging.getLogger("hermes_lark_streaming")


PREFIX = "HERMES_LARK"

_HOOK_NAMES = [
    "NORMALIZE",
    "START",
    "COMPLETE",
    "FOLLOWUP_COMPLETE",
    "FOLLOWUP_RESULT",
    "FOLLOWUP_DELIVERY",
    "TOOL",
    "ANSWER",
    "THINKING",
    "REASONING",
    "BACKGROUND_REVIEW",
    "ABORT",
    "INTERRUPT",
    "BG_DELIVER",
]
MARKERS: list[tuple[str, str]] = [(f"# {PREFIX}_{n}_BEGIN", f"# {PREFIX}_{n}_END") for n in _HOOK_NAMES]

MK_NORMALIZE, MK_NORMALIZE_END = MARKERS[0]
MK_START, MK_START_END = MARKERS[1]
MK_COMPLETE, MK_COMPLETE_END = MARKERS[2]
MK_FOLLOWUP_COMPLETE, MK_FOLLOWUP_COMPLETE_END = MARKERS[3]
MK_FOLLOWUP_RESULT, MK_FOLLOWUP_RESULT_END = MARKERS[4]
MK_FOLLOWUP_DELIVERY, MK_FOLLOWUP_DELIVERY_END = MARKERS[5]
MK_TOOL, MK_TOOL_END = MARKERS[6]
MK_ANSWER, MK_ANSWER_END = MARKERS[7]
MK_THINKING, MK_THINKING_END = MARKERS[8]
MK_REASONING, MK_REASONING_END = MARKERS[9]
MK_BACKGROUND_REVIEW, MK_BACKGROUND_REVIEW_END = MARKERS[10]
MK_ABORT, MK_ABORT_END = MARKERS[11]
MK_INTERRUPT, MK_INTERRUPT_END = MARKERS[12]
MK_BG_DELIVER, MK_BG_DELIVER_END = MARKERS[13]

_BACKUP_SUFFIX = ".hermes_lark.bak"


def _valid_source(path: Path) -> Path | None:
    try:
        candidate = path.resolve()
        if candidate.is_file() and candidate.suffix == ".py":
            return candidate
    except (OSError, RuntimeError):
        pass
    return None


def _module_to_path(module_name: str) -> Path:
    """gateway.run → gateway/run.py."""
    return Path(*module_name.split(".")).with_suffix(".py")


# 候选代码根目录。生产默认 Home 下可能遗留旧 checkout；此时优先运行时
# import 根。显式切换 HERMES_HOME（测试、profile、独立安装）时则严格优先该 Home。
def _code_roots() -> list[Path]:
    home_root = hermes_home() / "hermes-agent"
    default_home = Path.home() / ".hermes"
    roots: list[Path] = [home_root] if hermes_home() != default_home else []
    try:
        spec = importlib.util.find_spec("gateway")
        locations = spec.submodule_search_locations if spec else None
        for location in locations or []:
            candidate = Path(location).parent
            if candidate not in roots:
                roots.append(candidate)
    except Exception:
        _logger.debug("Unable to resolve active gateway import root", exc_info=True)
    for candidate in (Path("/usr/local/lib/hermes-agent"), home_root):
        if candidate not in roots:
            roots.append(candidate)
    return roots


# venv 内 python 解释器候选（覆盖 venv/.venv 命名变体）。
_VENV_PYTHONS: tuple[tuple[str, ...], ...] = (
    ("venv", "bin", "python3"),
    ("venv", "bin", "python"),
    (".venv", "bin", "python3"),
    (".venv", "bin", "python"),
)


def _python_from_hermes_cli() -> Path | None:
    """从 which hermes 反推 venv 内的 python3。"""
    cli = shutil.which("hermes")
    if cli is None:
        return None
    cli_path = Path(cli)
    # 读脚本内容，依次试 exec 行(bash wrapper) 和 shebang(console_scripts)
    try:
        text = cli_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    # 1. bash wrapper: exec "venv/bin/hermes" → 同目录 python3
    m = re.search(r'''exec\s+["']([^"']+)["']''', text)
    if m:
        venv_bin = Path(m.group(1)).parent  # venv/bin
        for name in ("python3", "python"):
            py = venv_bin / name
            if py.exists():
                return py
    # 2. console_scripts: shebang #!/path/to/python3 直接指向 python
    m = re.match(r'^#!\s*(\S+)', text)
    if m:
        py = Path(m.group(1))
        if py.exists() and "python" in py.name.lower():
            return py
    return None


def hermes_python() -> Path | None:
    """定位 Hermes 的 Python: which hermes 优先, _code_roots 兜底."""
    # 1. which hermes (覆盖所有官方安装方式，跨平台)
    if py := _python_from_hermes_cli():
        return py
    # 2. 兜底: 已知代码根下的 venv
    for root in _code_roots():
        for parts in _VENV_PYTHONS:
            py = root.joinpath(*parts)
            if py.exists():
                return py
    return None


def hermes_install_dir() -> Path | None:
    """定位 Hermes 安装目录 (含 gateway/run.py): hermes_constants 优先, _code_roots 兜底."""
    # 1. 用 Hermes Python 调用官方 API (single source of truth)
    py = hermes_python()
    if py is not None:
        try:
            result = subprocess.run(
                [str(py), "-c", "from hermes_constants import get_hermes_home; print(get_hermes_home())"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            _logger.debug("hermes_constants lookup failed", exc_info=True)
        else:
            if result.returncode == 0:
                home = Path(result.stdout.strip())
                install = home / "hermes-agent"
                if install.exists():
                    return install
    # 2. 兜底: _code_roots 里含 gateway/run.py 的那个
    rel = _module_to_path("gateway.run")
    for root in _code_roots():
        if (root / rel).exists():
            return root
    return None


def _resolve_module_path(module_name: str, roots: list[Path]) -> Path:
    """定位 Hermes 模块文件，候选代码根优先，importlib 兜底."""
    rel = _module_to_path(module_name)
    for root in roots:
        if candidate := _valid_source(root / rel):
            return candidate

    package = module_name.partition(".")[0]
    try:
        spec = importlib.util.find_spec(package)
        locations = spec.submodule_search_locations if spec else None
        # submodule_search_locations 指向包目录（如 .../gateway），
        # 因此需剥掉包名前缀，得到包内子路径。
        in_pkg = rel.relative_to(Path(package)) if rel.parts[0] == package else rel
        for location in locations or []:
            if candidate := _valid_source(Path(location) / in_pkg):
                return candidate
    except Exception:
        _logger.debug("Failed to resolve Hermes module %s", module_name, exc_info=True)
    return roots[0] / rel if roots else rel


def _default_run_path() -> Path:
    return _resolve_module_path("gateway.run", _code_roots())


def _default_cron_path() -> Path:
    return _resolve_module_path("cron.scheduler", _code_roots())


MK_CRON_DELIVER = f"# {PREFIX}_CRON_DELIVER_BEGIN"
MK_CRON_DELIVER_END = f"# {PREFIX}_CRON_DELIVER_END"

_ANCHOR_CHECKS: list[tuple[str, tuple[str, ...], str]] = [
    ("Restart typing indicator so the user sees activity", (), "interrupt"),
    ('was_interrupted = result.get("interrupted")', (), "queued follow-up boundary"),
    ("return _preserve_queued_followup_history_offset(result, followup_result)", (), "queued follow-up return"),
    ("agent.reasoning_config = reasoning_config", (), "reasoning_config"),
    ("agent.background_review_callback = _bg_review_send", (), "background_review_callback"),
    ("images, text_content = adapter.extract_images(response)", (), "background deliver"),
    ("_already_sent = bool(", (), "complete"),
    ("Discarding stale agent result", (), "abort"),
]


def _make_hook(indent: str, begin: str, end: str, body_lines: list[str]) -> str:
    return f"{indent}{begin}\n" + "".join(f"{indent}{line}\n" for line in body_lines) + f"{indent}{end}\n"


def _feishu_normalize_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_NORMALIZE,
        MK_NORMALIZE_END,
        [
            "try:",
            "    from hermes_lark_streaming.patch import on_feishu_normalize",
            "    on_feishu_normalize(",
            "        message_id=event.message_id,",
            "        source=source,",
            "        event=event,",
            "        reply_anchor_id=self._reply_anchor_for_event(event),",
            "    )",
            "except Exception:",
            "    pass",
        ],
    )


def _start_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_START,
        MK_START_END,
        [
            "try:",
            "    _lark_p = str(getattr(getattr(source, 'platform', None), 'value', None) or getattr(source, 'platform', '') or getattr(event, 'platform', '')).lower()",
            "    if 'feishu' in _lark_p or 'lark' in _lark_p:",
            "        from hermes_lark_streaming.patch import on_message_started",
            "        import uuid as _lark_uuid",
            "        _lark_anchor_id = self._reply_anchor_for_event(event)",
            "        _lark_inbound_id = event.message_id or _lark_anchor_id or f'proactive_{_lark_uuid.uuid4().hex}'",
            "        on_message_started(",
            "            message_id=_lark_inbound_id,",
            "            chat_id=source.chat_id,",
            "            anchor_id=_lark_anchor_id,",
            "            platform=_lark_p,",
            "        )",
            "        if not getattr(event, 'message_id', None):",
            "            event.message_id = _lark_inbound_id",
            "except Exception:",
            "    pass",
        ],
    )


def _complete_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_COMPLETE,
        MK_COMPLETE_END,
        [
            "try:",
            "    _lark_p = str(getattr(getattr(source, 'platform', None), 'value', None) or getattr(source, 'platform', '') or getattr(event, 'platform', '')).lower()",
            "    if 'feishu' in _lark_p or 'lark' in _lark_p:",
            "        from hermes_lark_streaming.patch import on_message_completed_wait, on_message_needs_text_fallback",
            "        _lark_completion_id = agent_result.get('_hermes_lark_completion_id') or event.message_id",
            "        _lark_card_sent = await on_message_completed_wait(",
            "            message_id=_lark_completion_id,",
            "            answer=response,",
            "            duration=_response_time,",
            "            model=agent_result.get('model', ''),",
            "            tokens={",
            "                'input_tokens': agent_result.get('input_tokens', 0),",
            "                'output_tokens': agent_result.get('output_tokens', 0),",
            "            },",
            "            context={",
            "                'used_tokens': agent_result.get('last_prompt_tokens', 0),",
            "                'max_tokens': agent_result.get('context_length', 0),",
            "            },",
            "        )",
            "        if _lark_card_sent:",
            "            agent_result['already_sent'] = True",
            "        elif on_message_needs_text_fallback(message_id=_lark_completion_id):",
            "            agent_result.pop('already_sent', None)",
            "except Exception:",
            "    pass",
        ],
    )


def _modular_complete_hook(indent: str) -> str:
    """Finalize the CardKit card before run_turn's normal final delivery."""
    return _make_hook(
        indent,
        MK_COMPLETE,
        MK_COMPLETE_END,
        [
            "try:",
            "    _lark_p = str(getattr(getattr(source, 'platform', None), 'value', None) or getattr(source, 'platform', '') or getattr(event, 'platform', '')).lower()",
            "    if 'feishu' in _lark_p or 'lark' in _lark_p:",
            "        from hermes_lark_streaming.patch import on_message_completed_wait, on_message_needs_text_fallback",
            "        _lark_completion_id = agent_result.get('_hermes_lark_completion_id') or event.message_id",
            "        _lark_card_sent = await on_message_completed_wait(",
            "            message_id=_lark_completion_id,",
            "            answer=response,",
            "            duration=_turn_seconds,",
            "            model=agent_result.get('model', ''),",
            "            tokens={",
            "                'input_tokens': agent_result.get('input_tokens', 0),",
            "                'output_tokens': agent_result.get('output_tokens', 0),",
            "            },",
            "            context={",
            "                'used_tokens': agent_result.get('last_prompt_tokens', 0),",
            "                'max_tokens': agent_result.get('context_length', 0),",
            "            },",
            "        )",
            "        if _lark_card_sent:",
            "            agent_result['already_sent'] = True",
            "        elif on_message_needs_text_fallback(message_id=_lark_completion_id):",
            "            agent_result.pop('already_sent', None)",
            "except Exception:",
            "    pass",
        ],
    )


def _followup_complete_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_FOLLOWUP_COMPLETE,
        MK_FOLLOWUP_COMPLETE_END,
        [
            "try:",
            "    from hermes_lark_streaming.patch import on_queued_followup_boundary",
            "    await on_queued_followup_boundary(message_id=event_message_id, result=result)",
            "except Exception:",
            "    pass",
        ],
    )


def _followup_result_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_FOLLOWUP_RESULT,
        MK_FOLLOWUP_RESULT_END,
        [
            "try:",
            "    from hermes_lark_streaming.patch import on_queued_followup_result",
            "    _lark_followup_completion_id = next_message_id or getattr(pending_event, 'message_id', None)",
            "    if _lark_followup_completion_id:",
            "        on_queued_followup_result(",
            "            message_id=_lark_followup_completion_id,",
            "            followup_result=followup_result,",
            "        )",
            "except Exception:",
            "    pass",
        ],
    )


def _modular_followup_complete_hook(indent: str) -> str:
    """Close the prior CardKit card before run_turn drains a non-interrupt queue item."""
    return _make_hook(
        indent,
        MK_FOLLOWUP_COMPLETE,
        MK_FOLLOWUP_COMPLETE_END,
        [
            "try:",
            "    _lark_p = str(getattr(getattr(turn_ctx.source, 'platform', None), 'value', None) or getattr(turn_ctx.source, 'platform', '')).lower()",
            "    if 'feishu' in _lark_p or 'lark' in _lark_p:",
            "        from hermes_lark_streaming.patch import on_queued_followup_boundary",
            "        await on_queued_followup_boundary(",
            "            message_id=turn_ctx.inbound_message_id or turn_ctx.event_message_id or '',",
            "            result=result,",
            "        )",
            "except Exception:",
            "    pass",
        ],
    )


def _modular_followup_result_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_FOLLOWUP_RESULT,
        MK_FOLLOWUP_RESULT_END,
        [
            "try:",
            "    from hermes_lark_streaming.patch import on_queued_followup_result",
            "    if next_inbound_id:",
            "        on_queued_followup_result(",
            "            message_id=next_inbound_id,",
            "            followup_result=followup_result,",
            "        )",
            "except Exception:",
            "    pass",
        ],
    )


def _modular_followup_delivery_hook(indent: str) -> str:
    """Do not let the native queue sender duplicate a card already finalized above."""
    return _make_hook(
        indent,
        MK_FOLLOWUP_DELIVERY,
        MK_FOLLOWUP_DELIVERY_END,
        [
            "if isinstance(result, dict) and result.get('already_sent') and not result.get('failed'):",
            "    _bg_cb = self._pop_post_delivery_callback(adapter, session_key, turn_ctx.run_generation)",
            "    if callable(_bg_cb):",
            "        with suppress(Exception):",
            "            _bg_result = _bg_cb()",
            "            if inspect.isawaitable(_bg_result):",
            "                await _bg_result",
            "    return",
        ],
    )


def _legacy_followup_delivery_hook(indent: str) -> str:
    """Marker retained for a uniform install state; legacy drain honors already_sent itself."""
    return _make_hook(
        indent,
        MK_FOLLOWUP_DELIVERY,
        MK_FOLLOWUP_DELIVERY_END,
        ["# Legacy queue delivery observes result['already_sent'] after FOLLOWUP_COMPLETE."],
    )


def _tool_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_TOOL,
        MK_TOOL_END,
        [
            "ctx = self._ctx",
            "try:",
            "    from hermes_lark_streaming.patch import on_tool_updated",
            "    if ctx._run_still_current() and event_type in ('tool.started', 'tool.completed'):",
            "        if on_tool_updated(",
            "            message_id=ctx.event_message_id or '',",
            "            tool_name=tool_name or '',",
            "            status='started' if event_type == 'tool.started' else 'completed',",
            "            detail=preview or '',",
            "        ):",
            "            return",
            "except Exception as _lark_tool_exc:",
            "    logger.warning('Hermes Lark tool hook failed: %s', _lark_tool_exc, exc_info=True)",
        ],
    )


def _answer_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_ANSWER,
        MK_ANSWER_END,
        [
            "try:",
            "    from hermes_lark_streaming.patch import on_answer_delta",
            "    if (text and ctx._run_still_current()",
            "            and on_answer_delta(message_id=ctx.event_message_id or '', text=text)):",
            "        return",
            "except Exception:",
            "    pass",
        ],
    )


def _thinking_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_THINKING,
        MK_THINKING_END,
        [
            "try:",
            "    from hermes_lark_streaming.patch import on_thinking_delta",
            "    if (text and not already_streamed and ctx._run_still_current()",
            "            and on_thinking_delta(message_id=ctx.event_message_id or '', text=text)):",
            "        return",
            "except Exception:",
            "    pass",
        ],
    )


def _reasoning_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_REASONING,
        MK_REASONING_END,
        [
            "def _reasoning_cb(text):",
            "    if text and ctx._run_still_current():",
            "        try:",
            "            from hermes_lark_streaming.patch import on_reasoning_delta",
            "            on_reasoning_delta(message_id=ctx.event_message_id or '', text=text)",
            "        except Exception:",
            "            pass",
            "agent.reasoning_callback = _reasoning_cb",
        ],
    )


def _background_review_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_BACKGROUND_REVIEW,
        MK_BACKGROUND_REVIEW_END,
        [
            "try:",
            "    from hermes_lark_streaming.patch import on_background_review_message",
            "    _lark_bg_review_sender = agent.background_review_callback",
            "    def _lark_bg_review_callback(message):",
            "        _lark_bg_review_deferred = on_background_review_message(",
            "            message_id=ctx.event_message_id or '',",
            "            text=message,",
            "            sender=_lark_bg_review_sender,",
            "        )",
            "        if not _lark_bg_review_deferred:",
            "            _lark_bg_review_sender(message)",
            "    agent.background_review_callback = _lark_bg_review_callback",
            "except Exception:",
            "    pass",
        ],
    )


def _abort_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_ABORT,
        MK_ABORT_END,
        [
            "try:",
            "    _lark_p = str(getattr(getattr(source, 'platform', None), 'value', None) or getattr(source, 'platform', '') or getattr(event, 'platform', '')).lower()",
            "    if 'feishu' in _lark_p or 'lark' in _lark_p:",
            "        from hermes_lark_streaming.patch import on_message_aborted",
            "        on_message_aborted(message_id=event.message_id)",
            "except Exception:",
            "    pass",
        ],
    )


def _interrupt_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_INTERRUPT,
        MK_INTERRUPT_END,
        [
            "try:",
            "    _lark_p = str(getattr(getattr(source, 'platform', None), 'value', None) or getattr(source, 'platform', '')).lower()",
            "    if 'feishu' in _lark_p or 'lark' in _lark_p:",
            "        from hermes_lark_streaming.patch import (",
            "            on_message_aborted, on_message_interrupted, on_message_started,",
            "        )",
            "        _lark_next_message_id = getattr(pending_event, 'message_id', None) or next_message_id",
            "        _lark_next_anchor_id = next_message_id",
            "        if was_interrupted and _lark_next_message_id:",
            "            on_message_interrupted(",
            "                message_id=event_message_id,",
            "                new_message_id=_lark_next_message_id,",
            "                chat_id=source.chat_id,",
            "                anchor_id=_lark_next_anchor_id,",
            "            )",
            "        elif was_interrupted:",
            "            on_message_aborted(message_id=event_message_id)",
            "        elif pending_event is not None and _lark_next_message_id:",
            "            on_message_started(",
            "                message_id=_lark_next_message_id,",
            "                chat_id=getattr(next_source, 'chat_id', source.chat_id),",
            "                anchor_id=_lark_next_anchor_id,",
            "                platform=_lark_p,",
            "            )",
            "except Exception:",
            "    pass",
        ],
    )


def _modular_interrupt_hook(indent: str) -> str:
    """Map run_turn's raw inbound IDs onto CardKit sessions during recursive drains."""
    return _make_hook(
        indent,
        MK_INTERRUPT,
        MK_INTERRUPT_END,
        [
            "try:",
            "    _lark_p = str(getattr(getattr(next_source, 'platform', None), 'value', None) or getattr(next_source, 'platform', '')).lower()",
            "    if 'feishu' in _lark_p or 'lark' in _lark_p:",
            "        from hermes_lark_streaming.patch import (",
            "            on_message_aborted, on_message_interrupted, on_message_started,",
            "        )",
            "        _lark_old_message_id = turn_ctx.inbound_message_id or turn_ctx.event_message_id or ''",
            "        if result.get('interrupted') and next_inbound_id:",
            "            on_message_interrupted(",
            "                message_id=_lark_old_message_id,",
            "                new_message_id=next_inbound_id,",
            "                chat_id=next_source.chat_id,",
            "                anchor_id=next_message_id,",
            "            )",
            "        elif result.get('interrupted'):",
            "            on_message_aborted(message_id=_lark_old_message_id)",
            "        elif pending_event is not None and next_inbound_id:",
            "            on_message_started(",
            "                message_id=next_inbound_id,",
            "                chat_id=next_source.chat_id,",
            "                anchor_id=next_message_id,",
            "                platform=_lark_p,",
            "            )",
            "except Exception:",
            "    pass",
        ],
    )


def _cron_deliver_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_CRON_DELIVER,
        MK_CRON_DELIVER_END,
        [
            "try:",
            "    if platform_name.lower() in ('feishu', 'lark'):",
            "        if '_hermes_lark_cron_seen' not in locals():",
            "            _hermes_lark_cron_seen = set()",
            "        _hermes_lark_cron_key = (str(chat_id), cleaned_delivery_content.strip())",
            "        if _hermes_lark_cron_key in _hermes_lark_cron_seen:",
            "            delivered = True",
            "            continue",
            "        from hermes_lark_streaming.patch import on_cron_deliver",
            "        if on_cron_deliver(",
            "            chat_id=chat_id,",
            "            content=cleaned_delivery_content.strip(),",
            "            loop=loop,",
            "            task_name=job.get('name', ''),",
            "            run_time=job.get('next_run_at', ''),",
            "        ):",
            "            _hermes_lark_cron_seen.add(_hermes_lark_cron_key)",
            "            delivered = True",
            "            continue",
            "except Exception:",
            "    pass",
        ],
    )


def _modular_cron_deliver_hook(indent: str) -> str:
    """Intercept the scheduler_delivery target loop before either native send lane runs."""
    return _make_hook(
        indent,
        MK_CRON_DELIVER,
        MK_CRON_DELIVER_END,
        [
            "try:",
            "    if t.platform_name.lower() in ('feishu', 'lark'):",
            "        if '_hermes_lark_cron_seen' not in locals():",
            "            _hermes_lark_cron_seen = set()",
            "        _hermes_lark_cron_key = (str(t.chat_id), cleaned_delivery_content.strip())",
            "        if _hermes_lark_cron_key in _hermes_lark_cron_seen:",
            "            continue",
            "        from hermes_lark_streaming.patch import on_cron_deliver",
            "        if on_cron_deliver(",
            "            chat_id=t.chat_id,",
            "            content=cleaned_delivery_content.strip(),",
            "            loop=t.loop,",
            "            task_name=job.get('name', ''),",
            "            run_time=job.get('next_run_at', ''),",
            "        ):",
            "            _hermes_lark_cron_seen.add(_hermes_lark_cron_key)",
            "            continue",
            "except Exception:",
            "    pass",
        ],
    )


def _bg_deliver_hook(indent: str) -> str:
    return _make_hook(
        indent,
        MK_BG_DELIVER,
        MK_BG_DELIVER_END,
        [
            "try:",
            "    if source.platform.value.lower() in ('feishu', 'lark') and response:",
            "        from hermes_lark_streaming.patch import on_background_deliver",
            "        _bg_preview = prompt[:60] + ('...' if len(prompt) > 60 else '')",
            "        if await on_background_deliver(",
            "            chat_id=source.chat_id,",
            "            preview=_bg_preview,",
            "            content=text_content,",
            "            reply_to_message_id=event_message_id,",
            "        ):",
            "            text_content = ''",
            "            if not images and not media_files:",
            "                return",
            "except Exception:",
            "    pass",
        ],
    )


def _remove_block(content: str, begin: str, end: str) -> str:
    lines = content.splitlines(keepends=True)
    begin_idx = end_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == begin:
            begin_idx = i
        if stripped == end:
            end_idx = i
            break
    if begin_idx is not None and end_idx is not None:
        return "".join(lines[:begin_idx] + lines[end_idx + 1 :])
    return content


def _atomic_write(path: Path, content: str) -> None:
    """原子写入：先写临时文件再 rename，防止崩溃时文件损坏."""
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=str(path.parent), prefix=".hermes_lark_", mode="w", encoding="utf-8"
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)
        shutil.copymode(path, tmp_path)
        os.replace(str(tmp_path), str(path))
    except BaseException:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


class PatcherError(RuntimeError):
    pass


class Patcher:
    """管理 AST 注入的安装和移除."""

    MARKERS: list[tuple[str, str]] = MARKERS

    def __init__(self, run_path: Path | None = None) -> None:
        self.run_path = run_path or _default_run_path()
        if not self.run_path.exists():
            tried = ", ".join(str(r) for r in _code_roots())
            raise PatcherError(
                f"gateway/run.py not found: {self.run_path} "
                f"(tried: {tried}). "
                f"Set HERMES_HOME to the dir containing hermes-agent/ and rerun."
            )
        gateway_dir = self.run_path.parent
        modular_paths = {
            "inbound": gateway_dir / "run_inbound.py",
            "turn": gateway_dir / "run_turn.py",
            "runner": gateway_dir / "run_turn_runner.py",
        }
        self._is_modular = all(path.is_file() for path in modular_paths.values())
        self._paths: dict[str, Path] = modular_paths if self._is_modular else {"run": self.run_path}

    def is_patched(self) -> bool:
        return any(
            begin in path.read_text(encoding="utf-8")
            for path in self._paths.values()
            for begin, _ in self.MARKERS
        )

    @property
    def target_paths(self) -> tuple[Path, ...]:
        """Concrete files carrying gateway hooks (one legacy file or split v2026.9 modules)."""
        return tuple(self._paths.values())

    def is_fully_patched(self) -> bool:
        if not self._is_modular:
            content = self.run_path.read_text(encoding="utf-8")
            return all(begin in content and end in content for begin, end in self.MARKERS)
        marker_paths = self._modular_marker_paths()
        return all(
            begin in marker_paths[begin].read_text(encoding="utf-8")
            and end in marker_paths[begin].read_text(encoding="utf-8")
            for begin, end in self.MARKERS
        )

    def verify_target(self) -> None:
        if self._is_modular:
            self._verify_modular_target()
            return
        content = self.run_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        handler = _find_func_body(tree, content.splitlines(keepends=True), "_handle_message_with_agent")
        if handler is None:
            raise PatcherError("Cannot find _handle_message_with_agent in run.py — Hermes version may be incompatible")

        anchor_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "emit":
                    hooks_obj = func.value
                    if (
                        isinstance(hooks_obj, ast.Attribute)
                        and hooks_obj.attr == "hooks"
                        and (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "agent:end")
                    ):
                        anchor_found = True
                        break
        if not anchor_found:
            raise PatcherError(
                "Cannot find hooks.emit('agent:end', ...) anchor in run.py — Hermes version may be incompatible"
            )

        required_callbacks = {
            "progress_callback": False,
            "_stream_delta_cb": False,
            "_interim_assistant_cb": False,
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in required_callbacks:
                required_callbacks[node.name] = True
        missing = [name for name, found in required_callbacks.items() if not found]
        if missing:
            raise PatcherError(
                f"Missing injection targets in run.py: {', '.join(missing)} — Hermes version may be incompatible"
            )

        # 字符串锚点检查：primary 命中或任一 fallback 命中即可。
        for primary, fallbacks, label in _ANCHOR_CHECKS:
            if primary in content or any(fb in content for fb in fallbacks):
                continue
            raise PatcherError(f"Cannot find {label} anchor in run.py — Hermes version may be incompatible")

        normalize_site = _find_handle_message_source_site(tree, content.splitlines(keepends=True))
        if normalize_site is None:
            raise PatcherError(
                "Cannot find _handle_message source anchor in run.py — Hermes version may be incompatible"
            )

    def apply(self) -> None:
        if self.is_fully_patched():
            return

        self.verify_target()
        if self._is_modular:
            if self.is_patched():
                self._remove_all_blocks()
            else:
                self._backup()
            for path, content in self._inject_modular_all().items():
                _atomic_write(path, content)
            return
        content = self.run_path.read_text(encoding="utf-8")
        if self.is_patched():
            for begin, end in self.MARKERS:
                content = _remove_block(content, begin, end)
        else:
            self._backup()
        content = self._inject_all(content)
        _atomic_write(self.run_path, content)

    def remove(self) -> None:
        if self._is_modular:
            self._remove_all_blocks()
            return
        content = self.run_path.read_text(encoding="utf-8")
        if not any(begin in content for begin, _ in self.MARKERS):
            return
        for begin, end in self.MARKERS:
            content = _remove_block(content, begin, end)
        _atomic_write(self.run_path, content)

    def restore(self) -> None:
        for path in self._paths.values():
            backup = path.with_suffix(path.suffix + _BACKUP_SUFFIX)
            if not backup.exists():
                raise PatcherError(f"No backup found: {backup}")
        for path in self._paths.values():
            backup = path.with_suffix(path.suffix + _BACKUP_SUFFIX)
            shutil.copy2(backup, path)

    def _backup(self) -> None:
        for path in self._paths.values():
            backup = path.with_suffix(path.suffix + _BACKUP_SUFFIX)
            if not backup.exists():
                shutil.copy2(path, backup)

    def _remove_all_blocks(self) -> None:
        """Remove every known block from every concrete module (safe for partial installs)."""
        for path in self._paths.values():
            content = path.read_text(encoding="utf-8")
            updated = content
            for begin, end in self.MARKERS:
                updated = _remove_block(updated, begin, end)
            if updated != content:
                _atomic_write(path, updated)

    def _modular_marker_paths(self) -> dict[str, Path]:
        return {
            MK_NORMALIZE: self._paths["inbound"],
            MK_START: self._paths["turn"],
            MK_COMPLETE: self._paths["turn"],
            MK_FOLLOWUP_COMPLETE: self._paths["turn"],
            MK_FOLLOWUP_RESULT: self._paths["turn"],
            MK_FOLLOWUP_DELIVERY: self._paths["turn"],
            MK_ABORT: self._paths["turn"],
            MK_INTERRUPT: self._paths["turn"],
            MK_TOOL: self._paths["runner"],
            MK_ANSWER: self._paths["runner"],
            MK_THINKING: self._paths["runner"],
            MK_REASONING: self._paths["runner"],
            MK_BACKGROUND_REVIEW: self._paths["runner"],
            MK_BG_DELIVER: self._paths["turn"],
        }

    def _modular_sources(self) -> dict[str, tuple[ast.Module, list[str]]]:
        return {
            name: (
                ast.parse(path.read_text(encoding="utf-8")),
                path.read_text(encoding="utf-8").splitlines(keepends=True),
            )
            for name, path in self._paths.items()
        }

    def _modular_sites(self) -> dict[str, tuple[str, tuple[int, str] | None]]:
        """Locate v2026.9 module anchors by behavior, not their old run.py coordinates."""
        sources = self._modular_sources()
        inbound_tree, inbound_lines = sources["inbound"]
        turn_tree, turn_lines = sources["turn"]
        runner_tree, runner_lines = sources["runner"]
        return {
            "normalize": ("inbound", _find_source_site(inbound_tree, inbound_lines, "_hm_admit_event")),
            "start": ("turn", _find_func_body(turn_tree, turn_lines, "_handle_message_with_agent")),
            "complete": (
                "turn", _find_line_site(turn_lines, "return await self._hmwa_deliver_turn_response(")
            ),
            "followup_complete": ("turn", _find_line_site(turn_lines, "if not result.get(\"interrupted\"):")),
            "followup_result": (
                "turn",
                _find_line_site(
                    turn_lines, "merged = _preserve_queued_followup_history_offset(result, followup_result)"
                ),
            ),
            "followup_delivery": (
                "turn",
                _find_line_site(
                    turn_lines, "_delivery_result = response if isinstance(response, dict) else (result or {})"
                ),
            ),
            "abort": (
                "turn",
                _find_line_site(turn_lines, "self._hmwa_discard_stale_result(source, _quick_key, run_generation)"),
            ),
            "interrupt": (
                "turn",
                _find_line_site(turn_lines, "# Restart the typing indicator; the outer typing task may be stale."),
            ),
            "tool": ("runner", _find_func_body(runner_tree, runner_lines, "progress_callback")),
            "answer": ("runner", _find_line_site(runner_lines, "for sink in delta_sinks:")),
            "thinking": ("runner", _find_func_body(runner_tree, runner_lines, "interim_assistant_cb")),
            "reasoning": (
                "runner",
                _find_line_site(
                    runner_lines,
                    "agent.reasoning_config, agent.service_tier = reasoning_config, runner._service_tier",
                ),
            ),
            "background_review": (
                "runner",
                _find_line_site(
                    runner_lines,
                    "agent.background_review_callback, bg_release = self._make_bg_review_callbacks()",
                    after=1,
                ),
            ),
            "bg_deliver": (
                "turn", _find_line_site(turn_lines, "images, text_content = adapter.extract_images(response)", after=1)
            ),
        }

    def _verify_modular_target(self) -> None:
        missing = [str(path) for path in self._paths.values() if not path.is_file()]
        if missing:
            raise PatcherError(f"Missing Hermes modular gateway files: {', '.join(missing)}")
        sites = self._modular_sites()
        for name, (_, location) in sites.items():
            if location is None:
                raise PatcherError(
                    f"Cannot find {name} anchor in Hermes modular gateway files — Hermes version may be incompatible"
                )

    def _inject_modular_all(self) -> dict[Path, str]:
        sources = self._modular_sources()
        lines_by_module = {name: lines for name, (_, lines) in sources.items()}
        sites = self._modular_sites()
        hook_fns = {
            "normalize": _feishu_normalize_hook,
            "start": _start_hook,
            "complete": _modular_complete_hook,
            "followup_complete": _modular_followup_complete_hook,
            "followup_result": _modular_followup_result_hook,
            "followup_delivery": _modular_followup_delivery_hook,
            "abort": _abort_hook,
            "interrupt": _modular_interrupt_hook,
            "tool": _tool_hook,
            "answer": _answer_hook,
            "thinking": _thinking_hook,
            "reasoning": _reasoning_hook,
            "background_review": _background_review_hook,
            "bg_deliver": _bg_deliver_hook,
        }
        pending: dict[str, list[tuple[int, str, str]]] = {name: [] for name in lines_by_module}
        for name, (module, location) in sites.items():
            if location is None:
                raise PatcherError(f"Cannot locate {name} injection site — Hermes version may be incompatible")
            pending[module].append((location[0], location[1], name))
        for module, module_sites in pending.items():
            lines = lines_by_module[module]
            for index, indent, name in sorted(module_sites, reverse=True):
                lines[index:index] = hook_fns[name](indent).splitlines(keepends=True)
        return {self._paths[module]: "".join(lines) for module, lines in lines_by_module.items()}

    def _inject_all(self, content: str) -> str:
        tree = ast.parse(content)
        lines = content.splitlines(keepends=True)

        hook_defs: list[tuple[str, str, tuple[int, str] | None]] = [
            ("normalize", "normalize", _find_handle_message_source_site(tree, lines)),
            ("start", "start", _find_func_body(tree, lines, "_handle_message_with_agent")),
            ("complete", "complete", _find_handler_return(tree, lines)),
            ("followup_complete", "followup_complete", _find_followup_complete_site(tree, lines)),
            ("followup_result", "followup_result", _find_followup_result_site(tree, lines)),
            ("followup_delivery", "followup_delivery", _find_followup_complete_site(tree, lines)),
            ("abort", "abort", _find_handler_abort(tree, lines)),
            ("interrupt", "interrupt", _find_interrupt_site(tree, lines)),
            ("tool", "tool", _find_func_body(tree, lines, "progress_callback")),
            ("answer", "answer", _find_answer_delta_site(lines)),
            ("thinking", "thinking", _find_func_body(tree, lines, "_interim_assistant_cb")),
            ("reasoning", "reasoning", _find_reasoning_site(tree, lines)),
            ("background_review", "background_review", _find_background_review_site(tree, lines)),
            ("bg_deliver", "bg_deliver", _find_bg_deliver_site(tree, lines)),
        ]

        sites: list[tuple[int, str, str]] = []
        for hook_fn_name, name, loc in hook_defs:
            if loc is None:
                # 定位失败硬失败，不静默跳过（防位置漂移致重复消息）
                raise PatcherError(
                    f"Cannot locate {name} injection site — Hermes version may be incompatible"
                )
            sites.append((loc[0], loc[1], hook_fn_name))

        sites.sort(key=lambda x: x[0], reverse=True)
        _HOOK_FNS = {
            "normalize": _feishu_normalize_hook,
            "start": _start_hook,
            "complete": _complete_hook,
            "followup_complete": _followup_complete_hook,
            "followup_result": _followup_result_hook,
            "followup_delivery": _legacy_followup_delivery_hook,
            "abort": _abort_hook,
            "interrupt": _interrupt_hook,
            "tool": _tool_hook,
            "answer": _answer_hook,
            "thinking": _thinking_hook,
            "reasoning": _reasoning_hook,
            "background_review": _background_review_hook,
            "bg_deliver": _bg_deliver_hook,
        }
        for idx, indent, fn_name in sites:
            hook = _HOOK_FNS[fn_name](indent)
            lines[idx:idx] = hook.splitlines(keepends=True)

        return "".join(lines)


def _find_func_body(tree: ast.Module, lines: list[str], name: str) -> tuple[int, str] | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            body = node.body
            start = 0
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                start = 1
            if start < len(body):
                lineno = body[start].lineno - 1
                indent = _safe_indent(lines, lineno)
                return lineno, indent
    return None


def _find_answer_delta_site(lines: list[str]) -> tuple[int, str] | None:
    """Find the real text streaming callback, not the TTS-only fallback.

    Hermes 0.20.x has two nested functions named ``_stream_delta_cb``:
    the primary text stream callback calls ``_stream_consumer.on_delta``;
    the later TTS fallback only exists when no text stream consumer was
    created. Injecting the Lark answer hook into the fallback leaves the
    primary consumer active, so Feishu receives both the native streamed
    final and the CardKit finalization.
    """
    for i, line in enumerate(lines):
        if "_stream_consumer.on_delta(text)" not in line:
            continue
        for j in range(i - 1, max(i - 20, -1), -1):
            if "def _stream_delta_cb" in lines[j]:
                for k in range(j + 1, i + 1):
                    stripped = lines[k].lstrip()
                    if "run_still_current()" in stripped and stripped.startswith("if "):
                        return k, _safe_indent(lines, k)
                return j + 1, _safe_indent(lines, j + 1)
    return None


def _find_line_site(lines: list[str], needle: str, *, after: int = 0) -> tuple[int, str] | None:
    """Return a stable insertion point at (or just after) an exact modular source anchor."""
    for index, line in enumerate(lines):
        if line.strip() == needle:
            target = index + after
            return target, _safe_indent(lines, index)
    return None


def _find_source_site(
    tree: ast.Module, lines: list[str], function_name: str,
) -> tuple[int, str] | None:
    """Find ``source = event.source`` inside a specific inbound function."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "source"
                    and isinstance(stmt.value, ast.Attribute)
                    and stmt.value.attr == "source"
                    and isinstance(stmt.value.value, ast.Name)
                    and stmt.value.value.id == "event"
                ):
                    lineno = stmt.end_lineno or stmt.lineno
                    return lineno, _safe_indent(lines, stmt.lineno - 1)
    return None


def _find_handle_message_source_site(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    return _find_source_site(tree, lines, "_handle_message")


def _find_handler_return(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("_already_sent = bool("):
            indent = _safe_indent(lines, i)
            return i, indent

    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "_handle_message_with_agent":
            returns = [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Return)
                and isinstance(n.value, ast.Name)
                and n.value.id == "response"
                and n.lineno is not None
            ]
            if returns:
                target = max(returns, key=lambda x: x.lineno)
                lineno = target.lineno - 1
                indent = _safe_indent(lines, lineno)
                return lineno, indent
    return None


def _find_handler_abort(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        if "Discarding stale agent result" in line:
            for j in range(i + 1, min(i + 20, len(lines))):
                if lines[j].strip() == "return None":
                    indent = _safe_indent(lines, j)
                    return j, indent
            break
    return None


def _find_interrupt_site(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        if "Restart typing indicator so the user sees activity" in line:
            indent = _safe_indent(lines, i)
            return i, indent
    return None


def _find_followup_complete_site(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        if line.strip() == 'was_interrupted = result.get("interrupted")':
            return i, _safe_indent(lines, i)
    return None


def _find_followup_result_site(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        if line.strip() == "return _preserve_queued_followup_history_offset(result, followup_result)":
            return i, _safe_indent(lines, i)
    return None


def _find_reasoning_site(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        if line.strip() == "agent.reasoning_config = reasoning_config":
            return i + 1, _safe_indent(lines, i)
    return None


def _find_background_review_site(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        if line.strip() == "agent.background_review_callback = _bg_review_send":
            return i + 1, _safe_indent(lines, i)
    return None


def _find_bg_deliver_site(tree: ast.Module, lines: list[str]) -> tuple[int, str] | None:
    for i, line in enumerate(lines):
        if line.strip() == "images, text_content = adapter.extract_images(response)":
            return i + 1, _safe_indent(lines, i)
    return None


def _safe_indent(lines: list[str], lineno: int) -> str:
    """获取缩进，跳过空行."""
    for i in range(lineno, -1, -1):
        if 0 <= i < len(lines) and lines[i].strip():
            return lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    for i in range(lineno + 1, len(lines)):
        if lines[i].strip():
            return lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    return ""


class CronPatcher:
    """Inject CRON_DELIVER into scheduler.py or modular scheduler_delivery.py."""

    def __init__(self, cron_path: Path | None = None) -> None:
        self.cron_path = cron_path or _default_cron_path()
        if not self.cron_path.exists():
            tried = ", ".join(str(r) for r in _code_roots())
            raise PatcherError(
                f"cron/scheduler.py not found: {self.cron_path} "
                f"(tried: {tried}). "
                f"Set HERMES_HOME to the dir containing hermes-agent/ and rerun."
            )
        delivery_path = self.cron_path.parent / "scheduler_delivery.py"
        self._is_modular = (
            delivery_path.is_file()
            and "def _deliver_result(" in delivery_path.read_text(encoding="utf-8")
        )
        self._target_path = delivery_path if self._is_modular else self.cron_path

    def is_patched(self) -> bool:
        return MK_CRON_DELIVER in self._target_path.read_text(encoding="utf-8")

    @property
    def target_path(self) -> Path:
        """Concrete scheduler file carrying the hook (scheduler_delivery.py when split)."""
        return self._target_path

    def verify_target(self) -> None:
        content = self._target_path.read_text(encoding="utf-8")
        if self._is_modular:
            if "def _deliver_result(" not in content:
                raise PatcherError("Cannot find _deliver_result in scheduler_delivery.py")
            for anchor in (
                "t = _prepare_target_delivery(",
                "t.platform_name",
                "cleaned_delivery_content",
                "_deliver_via_live_adapter(",
            ):
                if anchor not in content:
                    raise PatcherError(
                        f"Cannot find {anchor!r} anchor in scheduler_delivery.py — Hermes version may be incompatible"
                    )
            return
        if "delivered = False" not in content:
            raise PatcherError("Cannot find 'delivered = False' anchor in scheduler.py")
        if "cleaned_delivery_content" not in content:
            raise PatcherError("Cannot find 'cleaned_delivery_content' in scheduler.py")

    def apply(self) -> None:
        if self.is_patched():
            return
        self.verify_target()
        self._backup()
        lines = self._target_path.read_text(encoding="utf-8").splitlines(keepends=True)

        inject_idx = None
        if self._is_modular:
            for i, line in enumerate(lines):
                if line.strip() == "delivered = t.live_adapter_ready and _deliver_via_live_adapter(":
                    inject_idx = i
                    break
        else:
            for i, line in enumerate(lines):
                if line.strip() == "delivered = False":
                    inject_idx = i
                    break
        if inject_idx is None:
            raise PatcherError("Cannot find cron delivery injection anchor")

        indent = _safe_indent(lines, inject_idx)
        hook = _modular_cron_deliver_hook(indent) if self._is_modular else _cron_deliver_hook(indent)
        lines[inject_idx:inject_idx] = hook.splitlines(keepends=True)
        _atomic_write(self._target_path, "".join(lines))

    def remove(self) -> None:
        content = self._target_path.read_text(encoding="utf-8")
        if MK_CRON_DELIVER not in content:
            return
        content = _remove_block(content, MK_CRON_DELIVER, MK_CRON_DELIVER_END)
        _atomic_write(self._target_path, content)

    def restore(self) -> None:
        backup = self._target_path.with_suffix(self._target_path.suffix + _BACKUP_SUFFIX)
        if not backup.exists():
            raise PatcherError(f"No backup found: {backup}")
        shutil.copy2(backup, self._target_path)

    def _backup(self) -> None:
        backup = self._target_path.with_suffix(self._target_path.suffix + _BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(self._target_path, backup)
