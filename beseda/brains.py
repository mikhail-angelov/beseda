"""Pluggable "brains": anything that turns a user utterance into a stream of events.

A brain yields ("text", delta) for words to be spoken and ("tool", description)
for actions it performs. Add a new backend by implementing `ask`/`abort` and
registering it in BRAINS.
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Iterator, Protocol

from openai import OpenAI

log = logging.getLogger("beseda.brain")

Event = tuple[str, str]


class Brain(Protocol):
    def ask(self, text: str) -> Iterator[Event]: ...
    def abort(self) -> None: ...
    def close(self) -> None: ...


class OpenAICompatBrain:
    """Plain chat over any OpenAI-compatible API (DeepSeek by default). Keeps history in memory."""

    def __init__(self, model: str, prompt: str, base_url: str, api_key_env: str):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"{api_key_env} is not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.history = [{"role": "system", "content": prompt}]
        self._aborted = threading.Event()
        log.info("openai-compatible brain model=%s base_url=%s", model, base_url)

    def ask(self, text: str) -> Iterator[Event]:
        self._aborted.clear()
        self.history.append({"role": "user", "content": text})
        answer, finish, usage = "", None, None
        started = time.monotonic()
        log.info("request model=%s history=%d", self.model, len(self.history))
        with self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
            stream_options={"include_usage": True},
        ) as response:
            for chunk in response:
                if self._aborted.is_set():
                    log.info("aborted after %.2fs", time.monotonic() - started)
                    break
                usage = chunk.usage or usage
                if not chunk.choices:
                    continue
                finish = chunk.choices[0].finish_reason or finish
                delta = chunk.choices[0].delta.content
                if delta:
                    if not answer:
                        log.info("first token after %.2fs", time.monotonic() - started)
                    answer += delta
                    yield "text", delta
        log.info(
            "response done in %.2fs finish=%s chars=%d usage=%s",
            time.monotonic() - started, finish, len(answer), usage and usage.model_dump(exclude_none=True),
        )
        self.history.append({"role": "assistant", "content": answer})

    def abort(self) -> None:
        self._aborted.set()

    def close(self) -> None:
        pass


class PiBrain:
    """Coding agent `pi` in RPC mode: full tool access, session and history are managed by pi."""

    def __init__(self, model: str, prompt: str):
        cmd = ["pi", "--mode", "rpc", "--model", model, "--append-system-prompt", prompt]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log.info("pi started pid=%d model=%s cwd=%s", self.proc.pid, model, os.getcwd())
        self.events: queue.Queue[dict | None] = queue.Queue()
        self.tool_started: dict[str, float] = {}
        self.closing = False
        threading.Thread(target=self._read, name="pi-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="pi-stderr", daemon=True).start()

    def _read(self) -> None:
        # Protocol is strict JSONL split on LF only, so split bytes rather than using text-mode readline.
        for raw in self.proc.stdout:
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log.warning("pi non-JSON stdout: %r", line[:500])
                continue
            if log.isEnabledFor(logging.DEBUG) and event.get("type") != "message_update":
                log.debug("pi event %s", line[:2000].decode(errors="replace"))
            self.events.put(event)
        rc = self.proc.wait()
        (log.info if self.closing else log.error)("pi exited rc=%s", rc)
        self.events.put(None)

    def _read_stderr(self) -> None:
        for raw in self.proc.stderr:
            log.warning("pi stderr: %s", raw.decode(errors="replace").rstrip())

    def _send(self, command: dict) -> None:
        log.info("pi <- %s", command["type"])
        self.proc.stdin.write(json.dumps(command, ensure_ascii=False).encode() + b"\n")
        self.proc.stdin.flush()

    def ask(self, text: str) -> Iterator[Event]:
        stale = self.events.qsize()
        if stale:
            log.warning("pi has %d unread events before prompt (leftovers from previous turn)", stale)
        self._send({"type": "prompt", "message": text})
        while (event := self.events.get()) is not None:
            kind = event["type"]
            if kind == "response":
                if not event.get("success", True):
                    log.error("pi rejected %s: %s", event.get("command"), event.get("error"))
                    yield "error", str(event.get("error"))
                    return
            elif kind == "message_update":
                delta = event["assistantMessageEvent"]
                if delta["type"] == "text_delta":
                    yield "text", delta["delta"]
                elif delta["type"] == "error":
                    log.error("pi message error reason=%s", delta.get("reason"))
                    yield "error", delta.get("reason", "error")
            elif kind == "message_end":
                message = event.get("message") or {}
                if message.get("role") == "assistant":
                    log.info(
                        "pi message end stop=%s usage=%s error=%s",
                        message.get("stopReason"), message.get("usage"), message.get("errorMessage"),
                    )
            elif kind == "tool_execution_start":
                self.tool_started[event["toolCallId"]] = time.monotonic()
                description = _describe_tool(event["toolName"], event.get("args") or {})
                log.info("tool start %s", description)
                yield "tool", description
            elif kind == "tool_execution_end":
                started = self.tool_started.pop(event["toolCallId"], time.monotonic())
                log.info(
                    "tool end %s in %.2fs error=%s",
                    event["toolName"], time.monotonic() - started, event.get("isError"),
                )
            elif kind in ("auto_retry_start", "auto_retry_end", "compaction_start", "compaction_end", "extension_error"):
                log.warning("pi %s %s", kind, {k: v for k, v in event.items() if k not in ("type", "result")})
            elif kind == "agent_end":
                return
        yield "error", "pi exited"

    def abort(self) -> None:
        self._send({"type": "abort"})

    def close(self) -> None:
        self.closing = True
        self.proc.terminate()


def _describe_tool(name: str, args: dict) -> str:
    detail = args.get("command") or args.get("path") or args.get("pattern") or ""
    return f"{name} {detail}".strip()


# name -> factory(model or None, voice prompt from the language pack)
BRAINS = {
    "deepseek": lambda model, prompt: OpenAICompatBrain(
        model or "deepseek-v4-flash", prompt, "https://api.deepseek.com", "DEEPSEEK_API_KEY"
    ),
    "pi": lambda model, prompt: PiBrain(model or "deepseek/deepseek-v4-flash", prompt),
}
