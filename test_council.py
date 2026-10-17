#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for council.py — no external API needed.

Starts two fake OpenAI-compatible servers on 127.0.0.1 random ports, plus one
"dead" member pointing at an unused port. Expected: the dead member is skipped,
the two survivors still get peer-reviewed, and the chair (which we set to the
dead member on purpose) falls back to a survivor and produces the final answer.
"""
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import council


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeHandler(BaseHTTPRequestHandler):
    """Answers whichever phase the system prompt asks for."""

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n))
        sysmsg = next((m.get("content", "") for m in req.get("messages", []) if m.get("role") == "system"), "")
        user = "\n".join(m.get("content", "") for m in req.get("messages", []) if m.get("role") == "user")
        if "impartial judge" in sysmsg:            # phase 2: peer review
            ids = re.findall(r"\[Answer ([A-Z])\]", user)
            items = [{"id": i, "score": 10 - 2 * (ord(i) - ord("A")), "reason": f"reason {i}"}
                     for i in ids]
            content = json.dumps({"rankings": items})
        elif sysmsg.startswith("You chair"):       # phase 3: synthesis
            content = "SYNTHESIZED-FINAL by " + req["model"]
        else:                                       # phase 1: first opinion
            content = f"OPINION from {req['model']}"
        body = json.dumps({"choices": [{"message": {"content": content},
                                        "finish_reason": "stop"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):   # silence request logging
        pass


def start_fake(port):
    srv = HTTPServer(("127.0.0.1", port), FakeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def main():
    port_a, port_b, dead = free_port(), free_port(), free_port()
    servers = [start_fake(port_a), start_fake(port_b)]
    cfg = {
        "members": [
            {"name": "alpha", "base_url": f"http://127.0.0.1:{port_a}/v1", "model": "alpha-model"},
            {"name": "beta",  "base_url": f"http://127.0.0.1:{port_b}/v1", "model": "beta-model"},
            {"name": "dead",  "base_url": f"http://127.0.0.1:{dead}/v1",   "model": "dead-model"},
        ],
        "chairman": "dead",   # intentional: exercise the chair failover
        "timeout": 10,
        "max_tokens": 512,
    }

    # -- 1) library-level run: 2 alive + 1 dead member still reaches the final answer
    buf = io.StringIO()
    with redirect_stdout(buf):
        final = council.run("What is the plan?", cfg, verbose=False, chairman_name="dead")
    out = buf.getvalue()

    assert final and "SYNTHESIZED-FINAL" in final, f"no synthesized final: {final!r}"
    assert "✗" in out and "no response" in out, "dead member not reported as down"
    assert "peer review" in out.lower(), "peer review phase missing:\n" + out
    assert "unavailable" in out, "chair failover note missing:\n" + out
    print("[1/2] library run OK: dead member skipped, chair failover, final answer produced")

    # -- 2) end-to-end CLI run with a real config file
    with tempfile.TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "council.config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "council.py"),
             "What is the plan?", "--config", cfg_path, "--chairman", "dead"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=240, cwd=tempfile.gettempdir())
    assert r.returncode == 0, f"CLI exited {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "SYNTHESIZED-FINAL" in r.stdout, "CLI output missing final answer:\n" + r.stdout
    assert r.stdout.count("✓") == 2 and r.stdout.count("✗") == 1, \
        f"expected 2 up / 1 down, got:\n{r.stdout}"
    print("[2/2] CLI run OK (exit 0, 2 members up, 1 member down, final answer produced)")

    for srv in servers:
        srv.shutdown()
    print("PASS")


if __name__ == "__main__":
    main()
