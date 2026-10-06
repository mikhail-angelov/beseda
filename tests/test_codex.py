import stat
import sys
import textwrap

from beseda.brains import CodexBrain

# A stand-in for `codex app-server`: answers the handshake, then plays one turn with a command and two messages.
FAKE_SERVER = textwrap.dedent("""
    import json, sys
    def send(message):
        print(json.dumps(message), flush=True)
    def item(kind, **fields):
        return {"item": {"type": kind, "id": kind + "-1", **fields}}
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if method == "initialize":
            send({"id": message["id"], "result": {}})
        elif method == "thread/start":
            assert message["params"]["approvalPolicy"] == "never"
            send({"id": message["id"], "result": {"thread": {"id": "t1"}, "model": "gpt"}})
        elif method == "turn/start":
            send({"method": "item/commandExecution/requestApproval", "id": 99, "params": {}})
            send({"id": message["id"], "result": {"turn": {"id": "u1"}}})
            send({"method": "item/started", "params": item("agentMessage")})
            send({"method": "item/agentMessage/delta", "params": {"delta": "Counting."}})
            send({"method": "item/started", "params": item("commandExecution", command="/bin/zsh -lc 'ls | wc -l'")})
            send({"method": "item/completed", "params": item("commandExecution", status="completed", exitCode=0)})
            send({"method": "item/started", "params": item("agentMessage")})
            send({"method": "item/agentMessage/delta", "params": {"delta": "Seven files."}})
            send({"method": "turn/completed", "params": {"turn": {"id": "u1", "status": "completed"}}})
""")


def test_turn_streams_text_and_tools(tmp_path, monkeypatch):
    codex = tmp_path / "codex"
    codex.write_text(f"#!{sys.executable}\n{FAKE_SERVER}")
    codex.chmod(codex.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")

    brain = CodexBrain(None, "prompt")
    try:
        assert list(brain.ask("how many files?")) == [
            ("text", "Counting."), ("tool", "bash ls | wc -l"), ("text", "\n"), ("text", "Seven files."),
        ]
    finally:
        brain.close()
