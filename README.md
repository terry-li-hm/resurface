# Resurface

Search across AI coding assistant chat history — Claude Code, Codex, and OpenCode — from one place.

## Usage

```bash
# Scan today's sessions
resurface

# Scan a specific date
resurface 2026-02-21

# Search prompts from the last 7 days
resurface search "auth middleware"

# Deep search (includes assistant responses, not just prompts)
resurface search "debounce" --deep --days 30

# Filter by tool
resurface search "migration" --tool Claude
```

## Features

- **Multi-tool** — indexes Claude Code (`history.jsonl` + project transcripts), Codex, and OpenCode in one search
- **Two search modes** — prompt-only (fast) or full transcript (deep, parallelised with Rayon)
- **Regex patterns** — case-insensitive by default
- **HKT timestamps** — all output in Hong Kong Time
- **JSON output** — `--json` flag for piping to other tools
- **Session grouping** — scan mode shows prompts grouped by session with metadata

## Install

```bash
cargo build --release
# Binary at ./target/release/resurface
```

## License

MIT
