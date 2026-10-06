"""Pluggable "brains": anything that turns a user utterance into a stream of events.

A brain yields ("text", delta) for words to be spoken and ("tool", description)
for actions it performs. Add a new backend by implementing `ask`/`abort` and
registering it in BRAINS.
"""

import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
from typing import Iterator, Protocol

from openai import OpenAI, Timeout

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
        # Finite network limits: a stalled connection must not hold the conversation forever.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=Timeout(60.0, connect=10.0))
        self.model = model
        self.history = [{"role": "system", "content": prompt}]
        self._aborted = threading.Event()
        self._response = None
        log.info("openai-compatible brain model=%s base_url=%s", model, base_url)

    def ask(self, text: str) -> Iterator[Event]:
        self._aborted.clear()
        self.history.append({"role": "user", "content": text})
        answer, finish, usage = "", None, None
        started = time.monotonic()
        log.info("request model=%s history=%d", self.model, len(self.history))
        try:
            with self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True,
                stream_options={"include_usage": True},
            ) as response:
                self._response = response
                if self._aborted.is_set():  # Esc while the request was being opened
                    response.close()
                for chunk in response:
                    if self._aborted.is_set():
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
        except Exception:
            if not self._aborted.is_set():
                raise
            # abort() closed the stream under a blocked read; that error is the expected way out.
        finally:
            self._response = None
        if self._aborted.is_set():
            log.info("aborted after %.2fs", time.monotonic() - started)
        log.info(
            "response done in %.2fs finish=%s chars=%d usage=%s",
            time.monotonic() - started, finish, len(answer), usage and usage.model_dump(exclude_none=True),
        )
        self.history.append({"role": "assistant", "content": answer})

    def abort(self) -> None:
        self._aborted.set()
        if response := self._response:
            response.close()  # unblocks a read waiting for the next chunk

    def close(self) -> None:
        self.client.close()


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


class CodexBrain:
    """Codex through `codex app-server`: JSON-RPC over stdio with streamed text, one thread per Beseda session.

    Commands run without approval prompts in the workspace-write sandbox: the agent can edit the current folder
    but not the rest of the disk, and commands have no network."""

    def __init__(self, model: str | None, prompt: str):
        self.proc = subprocess.Popen(
            ["codex", "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        log.info("codex app-server started pid=%d cwd=%s", self.proc.pid, os.getcwd())
        self.notifications: queue.Queue[dict | None] = queue.Queue()
        self.responses: dict[int, queue.Queue[dict | None]] = {}
        self.next_id = 0
        self.send_lock = threading.Lock()
        self.closing = False
        self.turn_id: str | None = None
        self.tool_started: dict[str, float] = {}
        threading.Thread(target=self._read, name="codex-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="codex-stderr", daemon=True).start()

        self._request("initialize", {"clientInfo": {"name": "beseda", "version": "0"}, "capabilities": None})
        self._send({"method": "initialized"})
        thread = self._request("thread/start", {
            "model": model, "cwd": os.getcwd(), "approvalPolicy": "never", "sandbox": "workspace-write",
            "developerInstructions": prompt, "ephemeral": True,
        })
        self.thread_id = thread["thread"]["id"]
        log.info("codex thread=%s model=%s effort=%s", self.thread_id, thread.get("model"), thread.get("reasoningEffort"))

    def _read(self) -> None:
        for raw in self.proc.stdout:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("codex non-JSON stdout: %r", raw[:500])
                continue
            if log.isEnabledFor(logging.DEBUG) and message.get("method") != "item/agentMessage/delta":
                log.debug("codex -> %s", raw[:2000].decode(errors="replace").rstrip())
            if "method" not in message:  # a response to one of our requests
                if waiter := self.responses.pop(message.get("id"), None):
                    waiter.put(message)
            elif "id" in message:  # a request from the server; with approvalPolicy=never none are expected
                log.warning("codex request %s refused", message["method"])
                self._send({"id": message["id"], "error": {"code": -32601, "message": "not supported by beseda"}})
            else:
                self.notifications.put(message)
        rc = self.proc.wait()
        (log.info if self.closing else log.error)("codex exited rc=%s", rc)
        self.notifications.put(None)
        for waiter in list(self.responses.values()):
            waiter.put(None)

    def _read_stderr(self) -> None:
        for raw in self.proc.stderr:
            log.warning("codex stderr: %s", raw.decode(errors="replace").rstrip())

    def _send(self, message: dict) -> None:
        with self.send_lock:
            self.proc.stdin.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
            self.proc.stdin.flush()

    def _request(self, method: str, params: dict, wait: bool = True) -> dict:
        with self.send_lock:
            self.next_id += 1
            request_id = self.next_id
        waiter: queue.Queue[dict | None] = queue.Queue()
        if wait:
            self.responses[request_id] = waiter
        log.info("codex <- %s", method)
        self._send({"id": request_id, "method": method, "params": params})
        if not wait:
            return {}
        try:
            response = waiter.get(timeout=60)
        except queue.Empty:
            raise RuntimeError(f"codex: no response to {method}") from None
        if response is None:
            raise RuntimeError("codex exited, see the log")
        if "error" in response:
            raise RuntimeError(f"codex {method}: {response['error'].get('message')}")
        return response["result"]

    def ask(self, text: str) -> Iterator[Event]:
        while not self.notifications.empty():  # notifications between turns (MCP startup, rate limits)
            if self.notifications.get() is None:
                yield "error", "codex exited"
                return
        try:
            turn = self._request("turn/start", {
                "threadId": self.thread_id, "input": [{"type": "text", "text": text, "text_elements": []}],
            })
        except RuntimeError as error:
            log.error("%s", error)
            yield "error", str(error)
            return
        self.turn_id = turn["turn"]["id"]
        spoken = ""
        try:
            while (message := self.notifications.get()) is not None:
                method, params = message["method"], message.get("params") or {}
                if method == "item/agentMessage/delta":
                    spoken += params["delta"]
                    yield "text", params["delta"]
                elif method == "item/started":
                    item = params["item"]
                    if item["type"] == "agentMessage" and spoken and not spoken[-1].isspace():
                        spoken += "\n"  # keeps the commentary and the answer as separate sentences
                        yield "text", "\n"
                    elif description := _describe_codex_item(item):
                        self.tool_started[item["id"]] = time.monotonic()
                        log.info("tool start %s", description)
                        yield "tool", description
                elif method == "item/completed":
                    item = params["item"]
                    if (started := self.tool_started.pop(item["id"], None)) is not None:
                        log.info(
                            "tool end %s in %.2fs status=%s exit=%s", item["type"], time.monotonic() - started,
                            item.get("status"), item.get("exitCode"),
                        )
                elif method == "error":
                    error = params["error"]
                    log.warning("codex error will_retry=%s %s", params.get("willRetry"), error)
                    if not params.get("willRetry"):
                        yield "error", error.get("message", "error")
                elif method == "thread/tokenUsage/updated":
                    log.info("codex usage %s", params.get("tokenUsage", {}).get("last"))
                elif method == "turn/completed":
                    result = params["turn"]
                    log.info("codex turn %s in %sms error=%s", result["status"], result.get("durationMs"), result.get("error"))
                    if result["status"] == "failed":
                        yield "error", (result.get("error") or {}).get("message", "turn failed")
                    return
            yield "error", "codex exited"
        finally:
            self.turn_id = None

    def abort(self) -> None:
        if turn_id := self.turn_id:
            self._request("turn/interrupt", {"threadId": self.thread_id, "turnId": turn_id}, wait=False)

    def close(self) -> None:
        self.closing = True
        self.proc.terminate()


def _describe_codex_item(item: dict) -> str | None:
    """A one-line description of an agent action, or None for items that aren't actions."""
    kind = item["type"]
    if kind == "commandExecution":
        command = item["command"]
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = []
        if len(parts) == 3 and parts[1] in ("-c", "-lc"):  # codex wraps commands in the user's shell
            command = parts[2]
        return f"bash {command}"
    if kind == "fileChange":
        return "edit " + ", ".join(change["path"] for change in item["changes"])
    if kind == "mcpToolCall":
        return f"{item['server']}.{item['tool']}"
    if kind == "webSearch":
        return f"web search {item.get('query', '')}".strip()
    return None


def _describe_tool(name: str, args: dict) -> str:
    detail = args.get("command") or args.get("path") or args.get("pattern") or ""
    return f"{name} {detail}".strip()


# name -> factory(model or None, voice prompt from the language pack)
BRAINS = {
    "deepseek": lambda model, prompt: OpenAICompatBrain(
        model or "deepseek-v4-flash", prompt, "https://api.deepseek.com", "DEEPSEEK_API_KEY"
    ),
    "pi": lambda model, prompt: PiBrain(model or "deepseek/deepseek-v4-flash", prompt),
    "codex": lambda model, prompt: CodexBrain(model, prompt),  # model None: the default from ~/.codex/config.toml
}
