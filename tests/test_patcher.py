"""Patcher tests — copy real run.py, apply/remove/verify against the copy.

Usage:
    ~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_patcher.py -v
"""

from __future__ import annotations

import ast
import shutil
import textwrap
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_lark_streaming.patcher import (
    MARKERS,
    MK_CRON_DELIVER,
    MK_CRON_DELIVER_END,
    CronPatcher,
    Patcher,
    PatcherError,
    _cron_deliver_hook,
    _find_answer_delta_site,
    _remove_block,
)

RUN_SRC = Path.home() / ".hermes" / "hermes-agent" / "gateway" / "run.py"
RUN_BAK = RUN_SRC.with_suffix(RUN_SRC.suffix + ".hermes_lark.bak")
SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLE_RUN = SAMPLES_DIR / "run.py"

_RUN_URL = "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/gateway/run.py"
_CRON_URL = "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/cron/scheduler.py"

CRON_SRC = Path.home() / ".hermes" / "hermes-agent" / "cron" / "scheduler.py"
CRON_BAK = CRON_SRC.with_suffix(CRON_SRC.suffix + ".hermes_lark.bak")
SAMPLE_CRON = SAMPLES_DIR / "scheduler.py"

def _ensure_sample() -> Path:
    src = RUN_BAK if RUN_BAK.exists() else RUN_SRC
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, SAMPLE_RUN)
        return SAMPLE_RUN
    # CI fallback: download from GitHub
    try:
        urllib.request.urlretrieve(_RUN_URL, SAMPLE_RUN)
    except Exception as exc:
        pytest.skip(f"run.py not found locally and download failed: {exc}")
    if not SAMPLE_RUN.exists() or SAMPLE_RUN.stat().st_size == 0:
        pytest.skip("run.py download returned empty file")
    return SAMPLE_RUN


@pytest.fixture()
def run_copy(tmp_path: Path) -> Path:
    src = _ensure_sample()
    dst = tmp_path / "run.py"
    shutil.copy2(src, dst)
    return dst


def _ensure_cron_sample() -> Path:
    src = CRON_BAK if CRON_BAK.exists() else CRON_SRC
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, SAMPLE_CRON)
        return SAMPLE_CRON
    try:
        urllib.request.urlretrieve(_CRON_URL, SAMPLE_CRON)
    except Exception as exc:
        pytest.skip(f"scheduler.py not found locally and download failed: {exc}")
    if not SAMPLE_CRON.exists() or SAMPLE_CRON.stat().st_size == 0:
        pytest.skip("scheduler.py download returned empty file")
    return SAMPLE_CRON


@pytest.fixture()
def scheduler_copy(tmp_path: Path) -> Path:
    src = _ensure_cron_sample()
    dst = tmp_path / "scheduler.py"
    shutil.copy2(src, dst)
    return dst


def _patcher(path: Path) -> Patcher:
    return Patcher(run_path=path)


def _cron_patcher(path: Path) -> CronPatcher:
    return CronPatcher(cron_path=path)


@pytest.fixture()
def modular_gateway_copy(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """A minimal v2026.9 split layout with every behavior-based patch anchor."""
    gateway = tmp_path / "gateway"
    cron = tmp_path / "cron"
    gateway.mkdir()
    cron.mkdir()
    run = gateway / "run.py"
    inbound = gateway / "run_inbound.py"
    turn = gateway / "run_turn.py"
    runner = gateway / "run_turn_runner.py"
    scheduler = cron / "scheduler.py"
    delivery = cron / "scheduler_delivery.py"
    run.write_text("# Gateway assembly module\n", encoding="utf-8")
    inbound.write_text(
        textwrap.dedent("""\
        class GatewayInboundMixin:
            async def _hm_admit_event(self, event):
                source = event.source
                return event, source, False
        """),
        encoding="utf-8",
    )
    turn.write_text(
        textwrap.dedent("""\
        import inspect
        from contextlib import suppress

        class GatewayTurnMixin:
            async def _handle_message_with_agent(self, event, source, _quick_key, run_generation):
                agent_result = {}
                response = ""
                _turn_seconds = 0.0
                self._hmwa_discard_stale_result(source, _quick_key, run_generation)
                return await self._hmwa_deliver_turn_response(
                    event, source, agent_result, response)

            async def _run_background_task_inner(self, prompt, source, event_message_id):
                response = ""
                adapter = self._adapter_for_source(source)
                images, text_content = adapter.extract_images(response)

            async def _run_agent_deliver_first_response(self, turn_ctx, adapter, response, result, stream_task):
                session_key = turn_ctx.session_key
                _delivery_result = response if isinstance(response, dict) else (result or {})
                return _delivery_result

            async def _run_agent_queued_followup(
                self, turn_ctx, adapter, pending, pending_event, response, result, stream_task,
            ):
                next_source = source = turn_ctx.source
                next_message_id = None
                next_inbound_id = None
                if pending_event is not None:
                    next_source = pending_event.source
                    next_message_id = self._reply_anchor_for_event(pending_event)
                    next_inbound_id = pending_event.message_id
                # Restart the typing indicator; the outer typing task may be stale.
                if not result.get("interrupted"):
                    await self._run_agent_deliver_first_response(turn_ctx, adapter, response, result, stream_task)
                followup_result = {}
                merged = _preserve_queued_followup_history_offset(result, followup_result)
                return merged
        """),
        encoding="utf-8",
    )
    runner.write_text(
        textwrap.dedent("""\
        class TurnRunner:
            def progress_callback(self, event_type, tool_name=None, preview=None):
                ctx = self._ctx
                return ctx

            def _setup_stream_consumer(self):
                ctx = self._ctx
                delta_sinks = []
                def stream_delta_cb(text):
                    if ctx._run_still_current():
                        for sink in delta_sinks:
                            sink.on_delta(text)
                def interim_assistant_cb(text, *, already_streamed=False):
                    if not ctx._run_still_current():
                        return
                return stream_delta_cb, interim_assistant_cb

            def _wire_turn_agent_callbacks(self, agent, reasoning_config, runner):
                ctx = self._ctx
                agent.reasoning_config, agent.service_tier = reasoning_config, runner._service_tier
                agent.background_review_callback, bg_release = self._make_bg_review_callbacks()
        """),
        encoding="utf-8",
    )
    scheduler.write_text("from cron.scheduler_delivery import _deliver_result\n", encoding="utf-8")
    delivery.write_text(
        textwrap.dedent("""\
        def _deliver_result(job, content, adapters=None, loop=None):
            cleaned_delivery_content = content
            targets = []
            for target in targets:
                t = _prepare_target_delivery(
                    job, target, adapters=adapters, loop=loop)
                if t is None:
                    continue
                if t.platform_name == "":
                    continue
                target_errors = []
                delivered = t.live_adapter_ready and _deliver_via_live_adapter(
                    t, cleaned_delivery_content, [], target_errors=target_errors)
        """),
        encoding="utf-8",
    )
    return run, inbound, turn, runner, scheduler


def _build_cron_hook_runner():
    namespace: dict = {
        "job": {
            "name": "test",
            "next_run_at": "2026-06-10T14:30:00+08:00",
        }
    }
    source = (
        "def deliver(targets, cleaned_delivery_content, loop):\n"
        "    fallback = []\n"
        "    for platform_name, chat_id in targets:\n"
        "        delivered = False\n"
        f"{_cron_deliver_hook('        ')}"
        "        fallback.append(chat_id)\n"
        "    return fallback\n"
    )
    exec(compile(source, "<cron-hook-test>", "exec"), namespace)
    return namespace["deliver"]


class TestVerify:
    def test_verify_passes_on_real_run(self, run_copy: Path) -> None:
        _patcher(run_copy).verify_target()

    def test_verify_fails_on_missing_handler(self, tmp_path: Path) -> None:
        p = tmp_path / "run.py"
        p.write_text(
            textwrap.dedent("""\
            async def _stream_delta_cb(text):
                pass
            async def progress_callback(event_type):
                pass
        """)
        )
        with pytest.raises(PatcherError, match="_handle_message_with_agent"):
            _patcher(p).verify_target()

    def test_verify_fails_on_missing_callback(self, tmp_path: Path) -> None:
        p = tmp_path / "run.py"
        p.write_text(
            textwrap.dedent("""\
            async def _handle_message_with_agent(source, event):
                self.hooks.emit("agent:end", {})
        """)
        )
        with pytest.raises(PatcherError, match="Missing injection targets"):
            _patcher(p).verify_target()

    def test_verify_fails_on_missing_reasoning_anchor(self, tmp_path: Path) -> None:
        p = tmp_path / "run.py"
        p.write_text(
            textwrap.dedent("""\
            async def _handle_message_with_agent(source, event):
                self.hooks.emit("agent:end", {})
            async def _stream_delta_cb(text):
                pass
            async def progress_callback(event_type):
                pass
            def _interim_assistant_cb(text):
                pass
            # Restart typing indicator so the user sees activity
            was_interrupted = result.get("interrupted")
            return _preserve_queued_followup_history_offset(result, followup_result)
        """)
        )
        with pytest.raises(PatcherError, match="reasoning_config"):
            _patcher(p).verify_target()

    def test_verify_fails_on_missing_complete_anchor(self, tmp_path: Path) -> None:
        """通过所有其他锚点，但缺 COMPLETE 锚点 `_already_sent = bool(`。"""
        p = tmp_path / "run.py"
        p.write_text(
            textwrap.dedent("""\
            async def _handle_message_with_agent(source, event):
                self.hooks.emit("agent:end", {})
            async def _stream_delta_cb(text):
                pass
            async def progress_callback(event_type):
                pass
            def _interim_assistant_cb(text):
                pass
            # Restart typing indicator so the user sees activity
            was_interrupted = result.get("interrupted")
            return _preserve_queued_followup_history_offset(result, followup_result)
            agent.reasoning_config = reasoning_config
            agent.background_review_callback = _bg_review_send
            images, text_content = adapter.extract_images(response)
        """)
        )
        with pytest.raises(PatcherError, match="complete anchor"):
            _patcher(p).verify_target()

    def test_verify_fails_on_missing_abort_anchor(self, tmp_path: Path) -> None:
        """通过所有其他锚点，但缺 ABORT 锚点 `Discarding stale agent result`。"""
        p = tmp_path / "run.py"
        p.write_text(
            textwrap.dedent("""\
            async def _handle_message_with_agent(source, event):
                self.hooks.emit("agent:end", {})
            async def _stream_delta_cb(text):
                pass
            async def progress_callback(event_type):
                pass
            def _interim_assistant_cb(text):
                pass
            # Restart typing indicator so the user sees activity
            was_interrupted = result.get("interrupted")
            return _preserve_queued_followup_history_offset(result, followup_result)
            agent.reasoning_config = reasoning_config
            agent.background_review_callback = _bg_review_send
            images, text_content = adapter.extract_images(response)
            _already_sent = bool(agent_result.get("already_sent"))
        """)
        )
        with pytest.raises(PatcherError, match="abort anchor"):
            _patcher(p).verify_target()


class TestApplyRemove:
    def test_apply_injects_all_markers(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        content = run_copy.read_text(encoding="utf-8")
        for begin, end in MARKERS:
            assert begin in content, f"Missing marker: {begin}"
            assert end in content, f"Missing marker: {end}"

    def test_apply_produces_valid_python(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        content = run_copy.read_text(encoding="utf-8")
        ast.parse(content)  # should not raise

    def test_find_answer_delta_site_uses_primary_stream_consumer_callback(self) -> None:
        lines = textwrap.dedent("""\
            if _want_stream_deltas:
                def _stream_delta_cb(text: str) -> None:
                    if ctx._run_still_current():
                        _stream_consumer.on_delta(text)
                        if _stts_consumer_ref is not None:
                            _stts_consumer_ref.on_delta(text)
            if _stream_delta_cb is None and _stts_consumer_ref is not None:
                def _stream_delta_cb(text: str) -> None:
                    if ctx._run_still_current():
                        _stts_consumer_ref.on_delta(text)
        """).splitlines(keepends=True)

        loc = _find_answer_delta_site(lines)

        assert loc is not None
        assert "_stream_consumer.on_delta" in lines[loc[0] + 1]

    def test_apply_injects_answer_hook_into_primary_stream_consumer_callback(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        lines = run_copy.read_text(encoding="utf-8").splitlines()
        hook_idx = next(i for i, line in enumerate(lines) if "# HERMES_LARK_ANSWER_BEGIN" in line)
        window = "\n".join(lines[max(0, hook_idx - 8): hook_idx + 16])

        assert "_stream_consumer.on_delta(text)" in window
        assert "if _stream_delta_cb is None and _stts_consumer_ref is not None" not in window

    def test_apply_uses_current_turn_message_id_for_card_session(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        content = run_copy.read_text(encoding="utf-8")

        assert "# HERMES_LARK_NORMALIZE_BEGIN" in content
        assert "source = event.source\n        # HERMES_LARK_NORMALIZE_BEGIN" in content
        assert "on_feishu_normalize(" in content
        assert "on_message_started(" in content
        assert "_lark_anchor_id = self._reply_anchor_for_event(event)" in content
        assert "message_id=_lark_inbound_id" in content
        assert "anchor_id=_lark_anchor_id" in content
        assert "_lark_next_message_id = getattr(pending_event, 'message_id', None) or next_message_id" in content
        assert "new_message_id=_lark_next_message_id" in content
        assert "anchor_id=_lark_next_anchor_id" in content
        assert "# HERMES_LARK_FOLLOWUP_COMPLETE_BEGIN" in content
        assert "message_id=event_message_id" in content
        assert "on_queued_followup_boundary(message_id=event_message_id, result=result)" in content
        assert "# HERMES_LARK_FOLLOWUP_RESULT_BEGIN" in content
        assert "on_queued_followup_result(" in content
        assert "on_message_completed_wait(" in content
        assert "on_message_needs_text_fallback" in content
        assert "_lark_card_sent = await on_message_completed_wait(" in content
        assert "agent_result.pop('already_sent', None)" in content
        assert "_lark_completion_id = agent_result.get('_hermes_lark_completion_id') or event.message_id" in content
        assert "message_id=_lark_completion_id" in content
        assert "ctx = self._ctx" in content
        assert "message_id=ctx.event_message_id or ''" in content
        assert "on_answer_delta(message_id=ctx.event_message_id or ''" in content
        assert "on_thinking_delta(message_id=ctx.event_message_id or ''" in content
        assert "on_reasoning_delta(message_id=ctx.event_message_id or ''" in content
        assert "Hermes Lark tool hook failed" in content
        assert "on_background_deliver(" in content
        assert "_bg_preview = prompt[:60] + ('...' if len(prompt) > 60 else '')" in content
        assert "content=text_content" in content
        assert "reply_to_message_id=event_message_id" in content
        assert "if not images and not media_files:" in content

    def test_apply_idempotent(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        after_first = run_copy.read_text(encoding="utf-8")
        patcher.apply()  # second apply should be no-op
        after_second = run_copy.read_text(encoding="utf-8")
        assert after_first == after_second

    def test_apply_upgrades_partial_patch(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        begin, end = next(pair for pair in MARKERS if "BACKGROUND_REVIEW" in pair[0])
        content = _remove_block(run_copy.read_text(encoding="utf-8"), begin, end)
        run_copy.write_text(content, encoding="utf-8")

        patcher.apply()
        upgraded = run_copy.read_text(encoding="utf-8")

        assert upgraded.count(begin) == 1
        assert upgraded.count(end) == 1

    def test_apply_hard_fails_when_injection_site_missing(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        with (
            patch("hermes_lark_streaming.patcher._find_handler_return", return_value=None),
            pytest.raises(PatcherError, match="locate complete injection site"),
        ):
            patcher.apply()

    def test_remove_restores_markers_free(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        original = run_copy.read_text(encoding="utf-8")
        patcher.apply()
        patcher.remove()
        after_remove = run_copy.read_text(encoding="utf-8")
        assert after_remove == original

    def test_remove_produces_valid_python(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        patcher.remove()
        content = run_copy.read_text(encoding="utf-8")
        ast.parse(content)

    def test_remove_on_unpatched_is_noop(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        original = run_copy.read_text(encoding="utf-8")
        patcher.remove()
        assert run_copy.read_text(encoding="utf-8") == original

    def test_apply_then_remove_repeatedly(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        original = run_copy.read_text(encoding="utf-8")
        for _ in range(3):
            patcher.apply()
            patcher.remove()
        assert run_copy.read_text(encoding="utf-8") == original


class TestBackupRestore:
    def test_backup_created_on_apply(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        patcher.apply()
        backup = run_copy.with_suffix(run_copy.suffix + ".hermes_lark.bak")
        assert backup.exists()

    def test_restore_recovers_original(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        original = run_copy.read_text(encoding="utf-8")
        patcher.apply()
        patcher.restore()
        assert run_copy.read_text(encoding="utf-8") == original

    def test_restore_fails_without_backup(self, run_copy: Path) -> None:
        patcher = _patcher(run_copy)
        with pytest.raises(PatcherError, match="No backup found"):
            patcher.restore()


class TestModularGatewayPatcher:
    def test_verify_apply_compile_and_remove_split_modules(
        self, modular_gateway_copy: tuple[Path, Path, Path, Path, Path]
    ) -> None:
        run, inbound, turn, runner, _ = modular_gateway_copy
        patcher = _patcher(run)
        originals = {path: path.read_text(encoding="utf-8") for path in (inbound, turn, runner)}

        assert patcher._is_modular is True
        patcher.verify_target()
        patcher.apply()

        for path in (inbound, turn, runner):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        assert "# HERMES_LARK_NORMALIZE_BEGIN" in inbound.read_text(encoding="utf-8")
        turn_text = turn.read_text(encoding="utf-8")
        assert "# HERMES_LARK_FOLLOWUP_DELIVERY_BEGIN" in turn_text
        assert "turn_ctx.inbound_message_id" in turn_text
        runner_text = runner.read_text(encoding="utf-8")
        assert "on_answer_delta" in runner_text
        assert "on_reasoning_delta" in runner_text

        first = {path: path.read_text(encoding="utf-8") for path in (inbound, turn, runner)}
        patcher.apply()
        assert first == {path: path.read_text(encoding="utf-8") for path in (inbound, turn, runner)}

        patcher.remove()
        patcher.remove()
        assert originals == {path: path.read_text(encoding="utf-8") for path in (inbound, turn, runner)}


# --- CronPatcher ---


class TestCronVerify:
    def test_verify_passes(self, scheduler_copy: Path) -> None:
        _cron_patcher(scheduler_copy).verify_target()

    def test_verify_fails_missing_delivered_false(self, tmp_path: Path) -> None:
        p = tmp_path / "scheduler.py"
        p.write_text("cleaned_delivery_content = ''\n")
        with pytest.raises(PatcherError, match="delivered = False"):
            _cron_patcher(p).verify_target()

    def test_verify_fails_missing_cleaned_content(self, tmp_path: Path) -> None:
        p = tmp_path / "scheduler.py"
        p.write_text("    delivered = False\n")
        with pytest.raises(PatcherError, match="cleaned_delivery_content"):
            _cron_patcher(p).verify_target()


class TestCronApplyRemove:
    def test_apply_injects_markers(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        cp.apply()
        content = scheduler_copy.read_text(encoding="utf-8")
        assert MK_CRON_DELIVER in content
        assert MK_CRON_DELIVER_END in content

    def test_apply_produces_valid_python(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        cp.apply()
        content = scheduler_copy.read_text(encoding="utf-8")
        compile(content, str(scheduler_copy), "exec")

    def test_apply_idempotent(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        cp.apply()
        first = scheduler_copy.read_text(encoding="utf-8")
        cp.apply()
        assert scheduler_copy.read_text(encoding="utf-8") == first

    def test_remove_restores_original(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        original = scheduler_copy.read_text(encoding="utf-8")
        cp.apply()
        cp.remove()
        assert scheduler_copy.read_text(encoding="utf-8") == original

    def test_remove_on_unpatched_is_noop(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        original = scheduler_copy.read_text(encoding="utf-8")
        cp.remove()
        assert scheduler_copy.read_text(encoding="utf-8") == original

    def test_injected_hook_references_on_cron_deliver(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        cp.apply()
        content = scheduler_copy.read_text(encoding="utf-8")
        assert "on_cron_deliver" in content
        assert "platform_name.lower()" in content
        assert "delivered = True" in content

    def test_injected_hook_skips_duplicate_card_target(self) -> None:
        deliver = _build_cron_hook_runner()
        sent = []

        def fake_on_cron_deliver(
            *, chat_id, content, loop, task_name, run_time
        ):
            sent.append((chat_id, content, task_name, run_time))
            return True

        targets = [("feishu", "oc_same"), ("feishu", "oc_same")]
        with patch(
            "hermes_lark_streaming.patch.on_cron_deliver",
            side_effect=fake_on_cron_deliver,
        ):
            fallback = deliver(targets, " failed ", object())

        assert sent == [
            (
                "oc_same",
                "failed",
                "test",
                "2026-06-10T14:30:00+08:00",
            )
        ]
        assert fallback == []

    def test_injected_hook_retries_duplicate_target_after_failure(self) -> None:
        deliver = _build_cron_hook_runner()
        targets = [("feishu", "oc_same"), ("feishu", "oc_same")]
        with patch(
            "hermes_lark_streaming.patch.on_cron_deliver",
            side_effect=[False, True],
        ) as mock_deliver:
            fallback = deliver(targets, "failed", object())

        assert mock_deliver.call_count == 2
        assert fallback == ["oc_same"]

    def test_modular_scheduler_delivery_target(
        self, modular_gateway_copy: tuple[Path, Path, Path, Path, Path]
    ) -> None:
        _, _, _, _, scheduler = modular_gateway_copy
        delivery = scheduler.parent / "scheduler_delivery.py"
        cp = _cron_patcher(scheduler)
        original = delivery.read_text(encoding="utf-8")

        assert cp._is_modular is True
        assert cp._target_path == delivery
        cp.verify_target()
        cp.apply()
        compile(delivery.read_text(encoding="utf-8"), str(delivery), "exec")
        patched = delivery.read_text(encoding="utf-8")
        assert "# HERMES_LARK_CRON_DELIVER_BEGIN" in patched
        assert "t.platform_name.lower()" in patched

        cp.apply()
        assert delivery.read_text(encoding="utf-8") == patched
        cp.remove()
        cp.remove()
        assert delivery.read_text(encoding="utf-8") == original


class TestCronBackupRestore:
    def test_backup_created_on_apply(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        cp.apply()
        backup = scheduler_copy.with_suffix(scheduler_copy.suffix + ".hermes_lark.bak")
        assert backup.exists()

    def test_restore_recovers_original(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        original = scheduler_copy.read_text(encoding="utf-8")
        cp.apply()
        cp.restore()
        assert scheduler_copy.read_text(encoding="utf-8") == original

    def test_restore_fails_without_backup(self, scheduler_copy: Path) -> None:
        cp = _cron_patcher(scheduler_copy)
        with pytest.raises(PatcherError, match="No backup found"):
            cp.restore()


class TestOnCronDeliverHook:
    def test_returns_false_when_disabled(self) -> None:
        from hermes_lark_streaming.patch import on_cron_deliver

        with patch("hermes_lark_streaming.patch.get_controller") as mock_get:
            ctrl = MagicMock()
            ctrl.enabled = False
            mock_get.return_value = ctrl
            assert on_cron_deliver(chat_id="c1", content="text", loop=MagicMock()) is False

    def test_returns_false_when_no_loop(self) -> None:
        from hermes_lark_streaming.patch import on_cron_deliver

        with patch("hermes_lark_streaming.patch.get_controller") as mock_get:
            ctrl = MagicMock()
            ctrl.enabled = True
            mock_get.return_value = ctrl
            assert on_cron_deliver(chat_id="c1", content="text", loop=None) is False

    def test_delegates_to_controller(self) -> None:
        from hermes_lark_streaming.patch import on_cron_deliver

        loop = MagicMock()
        with patch("hermes_lark_streaming.patch.get_controller") as mock_get:
            ctrl = MagicMock()
            ctrl.enabled = True
            ctrl.on_cron_deliver.return_value = True
            mock_get.return_value = ctrl
            result = on_cron_deliver(chat_id="c1", content="hello", loop=loop)
            assert result is True
            ctrl.on_cron_deliver.assert_called_once_with(
                chat_id="c1", content="hello", loop=loop,
                task_name="", run_time="",
            )


class TestQueuedFollowupHooks:
    @pytest.mark.asyncio
    async def test_boundary_marks_result_when_card_sent(self) -> None:
        from hermes_lark_streaming.patch import on_queued_followup_boundary

        with patch("hermes_lark_streaming.patch.get_controller") as mock_get:
            ctrl = MagicMock()
            ctrl.enabled = True
            ctrl.on_completed_wait = AsyncMock(return_value=True)
            mock_get.return_value = ctrl
            result = {"final_response": "ok", "model": "m"}

            assert await on_queued_followup_boundary(message_id="msg", result=result) is True

            assert result["response_previewed"] is True
            assert result["already_sent"] is True

    @pytest.mark.asyncio
    async def test_boundary_consumes_fallback_when_card_not_sent(self) -> None:
        from hermes_lark_streaming.patch import on_queued_followup_boundary

        with patch("hermes_lark_streaming.patch.get_controller") as mock_get:
            ctrl = MagicMock()
            ctrl.enabled = True
            ctrl.on_completed_wait = AsyncMock(return_value=False)
            mock_get.return_value = ctrl
            result = {"final_response": "plain"}

            assert await on_queued_followup_boundary(message_id="msg", result=result) is False

            ctrl.consume_text_fallback.assert_called_once_with("msg")
            assert "response_previewed" not in result

    def test_result_hook_preserves_deepest_completion_id(self) -> None:
        from hermes_lark_streaming.patch import on_queued_followup_result

        with patch("hermes_lark_streaming.patch.get_controller") as mock_get:
            ctrl = MagicMock()
            ctrl.enabled = True
            mock_get.return_value = ctrl
            result = {"_hermes_lark_completion_id": "deep"}

            on_queued_followup_result(message_id="outer", followup_result=result)

            assert result["_hermes_lark_completion_id"] == "deep"
