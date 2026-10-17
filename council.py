#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Council — a multi-model panel for harder questions.

Every member answers the same question in parallel, then members score each
other's answers anonymously, then a chair merges everything into one final
answer. A member that times out or errors is skipped and run continues; if the
configured chair is unreachable, the top-scoring surviving member takes over.

Pure Python standard library. No dependencies, no install step.

Members live in council.config.json (see council.config.example.json). Any
OpenAI-compatible /v1/chat/completions backend works: hosted APIs, gateway
projects, or a local server.

Usage
----
    python council.py "your question"
    python council.py                       # interactive prompt
    python council.py -v "question"         # show every member's raw answer
    python council.py --chairman NAME "q"   # pick the chair member
    python council.py --review --git HEAD "q"  # code-review mode: all members review a git diff
    python council.py --do CMD "question"   # run CMD afterwards, plan piped to stdin
    python council.py --config other.json "question"

Three phases
------
1. First opinions : all members answer the same question in parallel
2. Peer review    : answers are anonymized as Answer A/B/C, each member scores
                    the others (accuracy + insight) and ranks them
3. Synthesis      : the chair reads all answers + scores -> final answer
"""
import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Windows cp950 console / redirected pipes raise UnicodeEncodeError on ✓ or CJK.
# Force utf-8 at import so this is also safe when used as a module.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

USE_COLOR = sys.stdout.isatty()

# ── colors (Windows Terminal / modern cmd support ANSI) ─────────────────────
C = {"dim": "\033[2m", "b": "\033[1m", "cyan": "\033[36m", "yel": "\033[33m",
     "grn": "\033[32m", "red": "\033[31m", "mag": "\033[35m", "rst": "\033[0m"}


def color(s, c):
    return f"{C[c]}{s}{C['rst']}" if USE_COLOR else s


# ── config ──────────────────────────────────────────────────────────────────
CONFIG_NAME = "council.config.json"
DEFAULT_TIMEOUT = 120       # seconds per HTTP request
DEFAULT_MAX_TOKENS = 4096


def load_config(path=None):
    """Load the council config: --config path, else council.config.json next to
    this file, else one in the current directory."""
    if path:
        candidates = [path]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [os.path.join(here, CONFIG_NAME), os.path.join(os.getcwd(), CONFIG_NAME)]
    for cand in candidates:
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["_config_path"] = cand
            return cfg
    raise SystemExit(
        "No council config found. Copy council.config.example.json to council.config.json "
        "next to council.py (or pass --config path.json) and fill in your backends.")


def resolve_members(cfg):
    """Validate config entries and build normalized member dicts."""
    members = []
    for m in cfg.get("members", []):
        base = (m.get("base_url") or "").rstrip("/")
        if not base or not m.get("model"):
            raise SystemExit(f"Each member needs at least base_url and model; got: {m}")
        members.append({
            "name": m.get("name") or m["model"],
            "model": m["model"],
            "base_url": base,
            "endpoint": base + "/chat/completions",
            "api_key_env": m.get("api_key_env") or "",
        })
    if not members:
        raise SystemExit("No members in the config.")
    return members


def _api_key(member):
    # Keys only ever come from environment variables, never from files or the config.
    return os.environ.get(member["api_key_env"], "") if member["api_key_env"] else ""


def _lang_clause(cfg, default=" Respond in the same language as the question."):
    lang = (cfg.get("language") or "").strip()
    return f" Respond in {lang}." if lang else default


# ── OpenAI-compatible chat call ─────────────────────────────────────────────
def chat(member, messages, cfg, max_tokens=None, temperature=0.3):
    """POST {base_url}/chat/completions and return plain text.
    Raises RuntimeError when the backend is down; 429/502/503 are retried."""
    timeout = int(cfg.get("timeout") or DEFAULT_TIMEOUT)
    mt = max_tokens or int(cfg.get("max_tokens") or DEFAULT_MAX_TOKENS)
    body = {
        "model": member["model"],
        "messages": messages,
        "max_tokens": mt,
        "temperature": temperature,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    key = _api_key(member)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    elif member["api_key_env"]:
        print(f"[council] env var {member['api_key_env']} is not set; "
              f"calling {member['name']} without auth", file=sys.stderr)

    req = urllib.request.Request(member["endpoint"], data=payload, headers=headers)
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            choice = data["choices"][0]
            msg = choice["message"]
            content = (msg.get("content") or "").strip()
            if choice.get("finish_reason") == "length":
                # Say it out loud, otherwise half a sentence looks like a full answer.
                print(color(f"  [warn] {member['model']} output truncated at max_tokens={mt}", "yel"),
                      file=sys.stderr)
                if not content:
                    return ""   # nothing but reasoning tokens = no answer
            if content:
                return content
            # Some reasoning models return the answer only in reasoning_content;
            # take the last paragraph as a fallback.
            rc = (msg.get("reasoning_content") or "").strip()
            return (re.split(r"\n\n+", rc)[-1].strip() if rc else "")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (429, 502, 503):   # transient -> retry
                time.sleep(1.5)
                continue
            break                            # 4xx/other -> backend rejected us
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5)                  # connection error -> retry
    raise RuntimeError(last_err or "unknown error")


def _ask_member(member, messages, cfg, **kw):
    try:
        txt = chat(member, messages, cfg, **kw)
        return {"member": member, "text": txt, "ok": bool(txt)}
    except RuntimeError as e:
        return {"member": member, "text": "", "ok": False, "err": str(e)}


# ── Phase 1: first opinions ─────────────────────────────────────────────────
ANSWER_SYS = ("You are a rigorous expert. Give an accurate, insightful, well-structured answer. "
              "Answer directly, no pleasantries.")


def first_opinions(question, members, cfg):
    """Ask every member in parallel. Answers from dead members are marked down."""
    def ask(m):
        return _ask_member(m, [
            {"role": "system", "content": ANSWER_SYS + _lang_clause(cfg)},
            {"role": "user", "content": question},
        ], cfg)

    with ThreadPoolExecutor(max_workers=len(members)) as ex:
        return list(ex.map(ask, members))


# ── Phase 2: anonymous peer review ──────────────────────────────────────────
RANK_SYS = (
    "You are an impartial judge. Below are several anonymous answers to the same question. "
    "Score each 1-10 on accuracy and depth of insight, and rank them. "
    "Output exactly one JSON object, no ``` fences, no text before or after. Required format: "
    '{"rankings":[{"id":"A","score":8,"reason":"short reason"},{"id":"B","score":6,"reason":"..."}]} '
    "Use the key id (A/B/C...), not answer; include one entry per answer; you may score your own (you do not know which it is). "
    "reason: one sentence in English."
)
REVIEW_ANSWER_CAP = 3000   # truncate long answers so reviewers emit clean JSON


def _parse_rankings(raw):
    """Squeeze rankings out of a reviewer response. Tolerates id/answer/label keys,
    rankings/ranking/results envelopes, missing JSON fences, or prose around JSON."""
    if not raw:
        return []
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    def _norm(items):
        out = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            rid = it.get("id") or it.get("answer") or it.get("label")
            sc = it.get("score")
            if rid is not None and isinstance(sc, (int, float)):
                out.append({"id": str(rid).strip().upper()[:1],
                            "score": float(sc), "reason": it.get("reason", "")})
        return out

    def _pull(d):
        if isinstance(d, dict):
            return _norm(d.get("rankings") or d.get("ranking") or d.get("results") or [])
        if isinstance(d, list):
            return _norm(d)
        return []

    try:
        n = _pull(json.loads(cleaned))
        if n:
            return n
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            n = _pull(json.loads(m.group(0)))
            if n:
                return n
        except Exception:
            pass
    out = []
    for mm in re.finditer(
        r'"?(?:id|answer|label)"?\s*[:=]\s*"?([A-Za-z])"?.*?"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
        cleaned, re.IGNORECASE | re.DOTALL):
        out.append({"id": mm.group(1).upper(), "score": float(mm.group(2))})
    return out


def peer_review(question, alive, cfg):
    """alive: [{member, text}]. Returns {label: {member, text, avg, votes, reasons}}."""
    labels = [chr(ord("A") + i) for i in range(len(alive))]
    order = list(range(len(alive)))
    random.shuffle(order)
    label_map = {}
    panel_text = []
    for lab, idx in zip(labels, order):
        label_map[lab] = idx
        ans = alive[idx]["text"]
        if len(ans) > REVIEW_ANSWER_CAP:
            ans = ans[:REVIEW_ANSWER_CAP] + " ...(truncated)"
        panel_text.append(f"[Answer {lab}]\n{ans}")
    panel = "\n\n".join(panel_text)
    user_msg = f"Question: {question}\n\n{panel}\n\nScore and rank. Output JSON only."

    def review(member):
        try:
            raw = chat(member, [
                {"role": "system", "content": RANK_SYS},
                {"role": "user", "content": user_msg},
            ], cfg, temperature=0.0)
            rankings = _parse_rankings(raw)
            if not rankings:   # retry once with a firmer instruction
                raw2 = chat(member, [
                    {"role": "system", "content": RANK_SYS},
                    {"role": "user", "content": user_msg + "\n\nReminder: output JSON only; the first character must be {."},
                ], cfg, temperature=0.0)
                rankings = _parse_rankings(raw2)
            return rankings
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=len(alive)) as ex:
        all_rankings = list(ex.map(review, [a["member"] for a in alive]))

    scores = {lab: [] for lab in labels}
    reasons = {lab: [] for lab in labels}
    for rk in all_rankings:
        for item in rk:
            lab = str(item.get("id", "")).strip().upper()[:1]
            if lab in scores and isinstance(item.get("score"), (int, float)):
                scores[lab].append(float(item["score"]))
                if item.get("reason"):
                    reasons[lab].append(str(item["reason"]))

    out = {}
    for lab in labels:
        idx = label_map[lab]
        vs = scores[lab]
        out[lab] = {
            "member": alive[idx]["member"],
            "text": alive[idx]["text"],
            "avg": (sum(vs) / len(vs)) if vs else 0.0,
            "n": len(vs),
            "reasons": reasons[lab],
        }
    return out


# ── Phase 3: chair synthesis ────────────────────────────────────────────────
CHAIR_SYS = (
    "You chair a council of LLMs. Below are several experts' answers to one question, "
    "with the panel's anonymous scores. "
    "Synthesize all views (lean on high-scoring answers, but keep unique points from low-scoring ones), "
    "resolve contradictions, and produce a final answer more complete and accurate than any single one. "
    "Give the final answer directly."
)


def synthesize(question, scored, chairman, cfg):
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["avg"], reverse=True)
    blocks = []
    for lab, d in ranked:
        blocks.append(f"[Answer {lab} - avg {d['avg']:.1f}/10]\n{d['text']}")
    body = "\n\n".join(blocks)
    user_msg = f"Question: {question}\n\n{body}\n\nSynthesize the final answer."
    return chat(chairman, [
        {"role": "system", "content": CHAIR_SYS + _lang_clause(cfg)},
        {"role": "user", "content": user_msg},
    ], cfg, temperature=0.4)


# ── main flow ───────────────────────────────────────────────────────────────
def _pick_chair(members, alive, scored, chairman_name):
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["avg"], reverse=True)
    chair = next((m for m in members if m["name"] == chairman_name), None) if chairman_name else None
    alive_names = {a["member"]["name"] for a in alive}
    if not chair or chair["name"] not in alive_names:
        chair = ranked[0][1]["member"]   # fall back to the top-scoring survivor
        fallback = True
    else:
        fallback = False
    return chair, fallback


def run(question, cfg, verbose=False, chairman_name=None):
    members = resolve_members(cfg)
    print(color("\n════════ LLM COUNCIL ════════", "b"))
    print(color(f"Question: {question}\n", "cyan"))

    print(color(f"[1/3] Collecting first opinions ({len(members)} members in parallel)…", "dim"))
    results = first_opinions(question, members, cfg)
    alive, down = [], []
    for r in results:
        if r["ok"]:
            alive.append(r)
            tag = color("✓", "grn")
        else:
            down.append(r)
            tag = color("✗ no response", "red")
        print(f"  {tag} {r['member']['name']:<20} ({r['member']['model']})")
        if not r["ok"]:
            print(color(f"      └─ {r.get('err', '')[:80]}", "dim"))

    if not alive:
        print(color("\nNo member responded; cannot convene. Check URLs, API keys, network.", "red"))
        return None
    if len(alive) == 1:
        only = alive[0]
        print(color(f"\nOnly {only['member']['name']} responded → returning its answer directly:\n", "yel"))
        print(only["text"])
        return only["text"]

    if verbose:
        for r in alive:
            print(color(f"\n── first opinion: {r['member']['name']} ──", "mag"))
            print(r["text"])

    print(color("\n[2/3] Anonymous peer review…", "dim"))
    scored = peer_review(question, alive, cfg)
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["avg"], reverse=True)
    print(color("\n  Scores (anonymized labels revealed):", "b"))
    for lab, d in ranked:
        name = color(d["member"]["name"], "cyan")
        score = color(f"{d['avg']:.1f}/10", "yel")
        bar = color("█" * int(round(d["avg"])), "grn")
        print(f"   Answer {lab} = {name:<24} {score} {bar} ({d['n']} votes)")

    chair, fallback = _pick_chair(members, alive, scored, chairman_name or cfg.get("chairman"))
    if fallback:
        print(color(f"\n  (chair {chairman_name or cfg.get('chairman') or '(none)'} unavailable → "
                    f"falling back to {chair['name']})", "yel"))
    print(color(f"\n[3/3] Chair \"{chair['name']}\" synthesizing…", "dim"))
    final = synthesize(question, scored, chair, cfg)

    print(color("\n════════ FINAL ANSWER ════════", "b"))
    print(color(f"(chair: {chair['name']})\n", "dim"))
    print(final)
    print(color("\n═════════════════════════\n", "b"))
    return final


# ── --do: hand the synthesized plan to a shell command ──────────────────────
def execute_cmd(cmd, plan):
    """Run cmd in a shell with the final answer piped to stdin and set as
    the COUNCIL_PLAN environment variable."""
    print(color(f"\n[execute] running: {cmd}", "yel"))
    env = os.environ.copy()
    env["COUNCIL_PLAN"] = plan
    try:
        r = subprocess.run(cmd, shell=True, input=plan, text=True,
                           encoding="utf-8", errors="replace", env=env)
        return r.returncode
    except Exception as e:
        print(color(f"  command failed: {e}", "red"))
        return 1


# ── --review: all members review the same git diff, the chair merges ────────
REVIEW_SYS = (
    "You are a senior code reviewer. Review the git diff below for correctness bugs, security issues, "
    "edge cases, and performance/maintainability concerns. Tag each point with severity [BLOCKER/MAJOR/MINOR] and file location. "
    "Terse bullet list, no pleasantries. If nothing is wrong, say 'No issues found'. Write in English."
)
CONSOLIDATE_SYS = (
    "You are the review editor. Below are independent reviews of the same diff by several senior engineers. "
    "Merge them into one final review: dedupe, sort by severity, mark each point as majority consensus or a single opinion, "
    "drop obvious false positives, and end with one overall verdict."
)
DIFF_CAP = int(os.environ.get("COUNCIL_DIFF_CAP", "16000"))


def get_diff(git_target):
    # Accept "HEAD", "main", "HEAD -- path/to/file", "--staged", etc.
    parts = shlex.split(git_target) if git_target.strip() else []
    try:
        r = subprocess.run(["git", "diff", *parts], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=30)
        return r.stdout or ""
    except Exception as e:
        print(color(f"  git diff failed: {e}", "red"))
        return ""


def run_review(git_target, cfg, chairman_name=None):
    members = resolve_members(cfg)
    print(color("\n════════ COUNCIL CODE REVIEW ════════", "b"))
    print(color(f"Target: git diff {git_target}\n", "cyan"))
    diff = get_diff(git_target)
    if not diff.strip():
        print(color("No changes to review (git diff is empty).", "yel"))
        return
    if len(diff) > DIFF_CAP:
        print(color(f"  [warn] diff over {DIFF_CAP} chars, truncated before review", "yel"))
        diff = diff[:DIFF_CAP] + "\n...(diff truncated)"

    print(color(f"[1/2] {len(members)} independent reviews…", "dim"))
    user_msg = f"Review this diff:\n\n```diff\n{diff}\n```"

    def rev(m):
        return _ask_member(m, [
            {"role": "system", "content": REVIEW_SYS},
            {"role": "user", "content": user_msg},
        ], cfg, temperature=0.0)

    with ThreadPoolExecutor(max_workers=len(members)) as ex:
        results = list(ex.map(rev, members))
    alive = [r for r in results if r["ok"] and r["text"]]
    for r in results:
        tag = color("✓", "grn") if (r["ok"] and r["text"]) else color("✗", "red")
        print(f"  {tag} {r['member']['name']}")
    if not alive:
        print(color("\nNo member responded; cannot review.", "red"))
        return

    blocks = [f"[Review by {r['member']['name']}]\n{r['text']}" for r in alive]
    editor = next((m for m in members if m["name"] == (chairman_name or cfg.get("chairman"))), None)
    alive_names = {a["member"]["name"] for a in alive}
    if not editor or editor["name"] not in alive_names:
        editor = alive[0]["member"]
    print(color(f"\n[2/2] Editor \"{editor['name']}\" merging…", "dim"))
    final = chat(editor, [
        {"role": "system", "content": CONSOLIDATE_SYS + _lang_clause(cfg, default=" Write the final review in English.")},
        {"role": "user", "content": "\n\n".join(blocks)},
    ], cfg, temperature=0.2)

    print(color("\n════════ CONSOLIDATED REVIEW ════════", "b"))
    print(color(f"(editor: {editor['name']}; {len(alive)} reviewers)\n", "dim"))
    print(final)
    print(color("\n══════════════════════════════\n", "b"))


def main():
    ap = argparse.ArgumentParser(
        description="LLM Council — ask your configured LLMs, score anonymously, synthesize a final answer.")
    ap.add_argument("question", nargs="*", help="the question to ask")
    ap.add_argument("-v", "--verbose", action="store_true", help="show every member's raw answer")
    ap.add_argument("--chairman", metavar="NAME", help="member name to use as chair "
                    "(default: config 'chairman', else the top-scoring member)")
    ap.add_argument("--config", metavar="PATH", help="path to council.config.json "
                    "(default: next to council.py or in the current directory)")
    ap.add_argument("--review", action="store_true",
                    help="code review mode: every member reviews a git diff, the chair merges one review")
    ap.add_argument("--git", default="HEAD", help="git diff target for --review (default HEAD)")
    ap.add_argument("--do", metavar="CMD",
                    help="after the council, run CMD in a shell; the final answer is piped to its stdin "
                         "and also set as the COUNCIL_PLAN environment variable")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.review:
        try:
            run_review(args.git, cfg, args.chairman)
        except KeyboardInterrupt:
            print(color("\nInterrupted.", "yel"))
        return

    question = " ".join(args.question).strip()
    if not question:
        try:
            question = input(color("Question> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            return
    if not question:
        return
    try:
        final = run(question, cfg, verbose=args.verbose, chairman_name=args.chairman)
        if args.do and final:
            execute_cmd(args.do, final)
    except KeyboardInterrupt:
        print(color("\nInterrupted.", "yel"))


if __name__ == "__main__":
    main()
