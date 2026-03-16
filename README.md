# dev-memory-mcp

A Model Context Protocol (MCP) server that gives Claude persistent memory across conversations — specifically built for developers who want Claude to remember architectural decisions, failed approaches, and project context between sessions.

---

## The problem

Claude is an exceptional thought partner for software development. But every conversation starts from zero. You spend the first few minutes re-explaining your stack, your constraints, what you already tried. Decisions you made two weeks ago are gone. Dead ends you hit are forgotten. The reasoning behind your architecture exists nowhere.

`dev-memory-mcp` solves this by giving Claude a persistent, searchable memory store that lives on your machine. At the start of a session, Claude can brief itself on your project. During a session, it logs decisions and flags dead ends automatically. The next session, that context is waiting.

---

## How it works

The server exposes 8 tools to Claude via MCP. Claude calls these tools naturally during conversation — you don't need to invoke them manually.

Four tools **write** to memory:

| Tool | What it stores |
|---|---|
| `remember_decision` | A design choice with its reasoning and alternatives considered |
| `log_dead_end` | A failed approach, why it failed, and when it might be worth retrying |
| `save_context` | A snapshot of project state — what's working, in progress, blocked |
| `add_question` | A deferred question or open issue to revisit later |

Four tools **read** from memory:

| Tool | What it returns |
|---|---|
| `get_session_brief` | A full structured summary to re-orient Claude at the start of a session |
| `recall` | Semantically relevant records for a natural language query |
| `resolve_question` | Marks an open question as answered |
| `list_projects` | All projects that have memory records |

All data is stored locally in a SQLite database. Nothing leaves your machine.

---

## Architecture

```
Claude Desktop
      │
      ▼
  server.py          ← FastMCP entry point
      │
      ├── tools.py   ← Tool definitions and input validation (Pydantic)
      ├── db.py      ← SQLite schema, CRUD, session brief aggregation
      └── embeddings.py  ← Local embedding generation + semantic search
            │
            └── all-MiniLM-L6-v2 (sentence-transformers, runs locally)
                  │
                  └── memory.db (SQLite — stores records + float32 vectors)
```

### Key design decisions

**Local-first.** `sentence-transformers` generates embeddings on your machine using `all-MiniLM-L6-v2` (~80MB, downloaded once). No OpenAI API key. No cloud. No cost per query. Your dev notes never leave your computer.

**No vector database.** Embeddings are stored as raw `float32` bytes directly in SQLite. Similarity search loads project vectors into memory and ranks with a matrix dot product (cosine similarity, since vectors are pre-normalized). This is correct and fast at personal scale — no `sqlite-vec`, FAISS, or Chroma dependency needed.

**Structured record types over freeform logs.** Four specific record types capture the information that's actually useful and actually gets forgotten: decisions (with reasoning), dead ends (with failure reasons), context snapshots, and open questions.

---

## Setup

### Prerequisites

- Python 3.10+
- [Claude Desktop](https://claude.ai/download)

### Install

```bash
git clone https://github.com/Kurious-George/dev-memory-mcp
cd dev-memory-mcp

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt  # Windows
# or
.venv/bin/pip install -r requirements.txt                    # macOS/Linux
```

Dependencies are intentionally minimal:

```
mcp
sentence-transformers
numpy
pydantic
```

### Configure Claude Desktop

Open your Claude Desktop config file (Settings → Developer → Edit Config) and add:

```json
{
  "mcpServers": {
    "dev-memory": {
      "command": "C:/path/to/dev-memory-mcp/.venv/Scripts/python.exe",
      "args": ["C:/path/to/dev-memory-mcp/server.py"]
    }
  }
}
```

Use the full absolute path to the venv's Python executable. Restart Claude Desktop after saving.

### Verify it's working

On first use, Claude will download `all-MiniLM-L6-v2` (~80MB) and cache it locally. This only happens once.

To confirm the server connected, start a new conversation and ask:

> "List all projects in dev memory"

Claude should call `list_projects` and respond (with an empty list if you haven't stored anything yet).

---

## Usage

You don't need to ask Claude to use specific tools. Just talk naturally — Claude will call the appropriate tool when relevant.

**Start a session:**
> "Give me a session brief for the FastRecov project"

**Log a decision mid-session:**
> "Remember that we chose Firecracker over QEMU for FastRecov because of the minimal attack surface and sub-second boot times"

**Flag a dead end:**
> "Log that we tried using sqlite-vec for vector storage but dropped it because the Windows DLL loading was fragile and we didn't need the extra dependency"

**Defer a question:**
> "Add an open question: how should we handle eBPF program lifecycle when a VM exits unexpectedly?"

**Search across memory:**
> "What do we know about our database decisions?"

**End a session:**
> "Save context for FastRecov before I close out"

---

## File structure

```
dev-memory-mcp/
├── server.py        ← Entry point, FastMCP initialization
├── db.py            ← SQLite schema and all database operations
├── embeddings.py    ← Embedding generation and semantic search
├── tools.py         ← All MCP tool definitions
├── requirements.txt
├── memory.db        ← Created on first run (git-ignored)
└── README.md
```

---

## .gitignore

```
.venv/
memory.db
__pycache__/
*.pyc
```

---

## Why MCP for this?

Claude can reason, write code, and analyze complex problems within a single conversation. What it fundamentally cannot do is persist state between conversations or access data that wasn't pasted into the context window.

MCP provides the bridge. This project is specifically designed around what MCP uniquely enables — not as a thin API wrapper, but as a structural solution to Claude's statelessness. The test: could you get 80% of this value by pasting data into a Claude conversation? For a personal dev memory store that accumulates across months of sessions, no. The data is too large, too dynamic, and too private to paste in.