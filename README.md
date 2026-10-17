# llm-council

Ask a panel of LLMs one question, let them score each other anonymously, and get one synthesized answer.

- Single file, pure Python standard library — zero dependencies, no install step.
- Members are defined in a JSON config; any OpenAI-compatible `/v1/chat/completions` backend works (hosted APIs, a local Ollama, gateway servers).
- An unreachable member is skipped, so one down backend never kills the run; the configured chair fails over to the best surviving member.
- Peer review is anonymized: answers are shuffled and relabeled Answer A/B/C, so scores reflect content, not brand.
- API keys are only ever read from environment variables named in the config.

## Quick start (60 seconds)

```bash
# 1. copy the example config and fill in your endpoints, model ids and key names
cp council.config.example.json council.config.json

# 2. make sure the env vars your config names are set
export OPENAI_API_KEY=sk-...

# 3. ask
python council.py "Is battery swapping better than fast charging for a taxi fleet?"

# show every member's raw answer
python council.py -v "same question"
```

## Config

`council.config.json` (gitignored; see `council.config.example.json`):

| field | meaning |
|---|---|
| `members[].name` | display name, also used by `--chairman` |
| `members[].base_url` | API root, e.g. `https://api.openai.com/v1`; the script POSTs to `<base_url>/chat/completions` (the official Anthropic API is Anthropic-format `/v1/messages`, not this shape — put Claude behind an OpenAI-compatible gateway, see the example config) |
| `members[].model` | model id sent to the backend |
| `members[].api_key_env` | env var holding the key; omit for keyless servers (e.g. Ollama) |
| `chairman` | member that writes the final answer; falls back to the top survivor |
| `timeout` | seconds per request (default 120) |
| `max_tokens` | per-request completion budget (default 4096) |
| `language` | optional: force reply language, e.g. `"Traditional Chinese"` (default: reply in the question's language) |

## Modes

```bash
python council.py "question"                           # standard council
python council.py -v "question"                        # also print every member's raw answer
python council.py --chairman local "question"
python council.py --review --git HEAD                  # every member reviews a git diff, the chair merges one review
python council.py --review --git=--staged              # review the staged changes instead
python council.py --do "pytest -x" "how do we fix the flaky test?"  # run CMD afterwards, plan on stdin (+ $COUNCIL_PLAN)
```

## How it works

1. **First opinions** — all members are queried in parallel.
2. **Peer review** — answers are truncated, shuffled and relabeled A/B/C; every member scores each answer 1-10 (accuracy + insight). The JSON is parsed tolerantly, and a reviewer that fails to answer its turns is simply dropped from the tally.
3. **Synthesis** — the chair reads all answers plus scores and writes the final answer. Transient HTTP errors (429/502/503) are retried.

## Differences from karpathy/llm-council

- **Zero dependencies**: standard library only, one file.
- **Resilient**: an unreachable member is skipped instead of aborting; the chair fails over to the best surviving member.
- **Config-driven**: swap backends by editing JSON, not code.
- **Extra modes**: `--review` (multi-model code review consolidated by an editor) and `--do` (hand the synthesized plan to a shell command).

## License

MIT — see [LICENSE](LICENSE).

---

## 繁中簡版

把同一個問題同時問多個**你設定的模型** → 答案匿名化（洗牌成 A/B/C）互評打分 → 主席彙整成一份總結。
單檔、純 Python 標準庫、零依賴；成員寫在 `council.config.json`，key 一律讀環境變數，不寫進檔案。
某個成員掛掉會自動跳過、主席自動改派存活者，不影響產出。

```bash
cp council.config.example.json council.config.json   # 填好 base_url / model / api_key_env
python council.py "你的問題"
python council.py --review --git HEAD   # 多模型 code review，主席彙整成一份
```
