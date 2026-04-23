# ai-mem

Semantic memory for AI agents. Store, search, and retrieve information across sessions using natural language — backed by [ChromaDB](https://www.trychroma.com/).

Works with **Claude Code**, **Gemini CLI**, and **Cursor**.

## Install

```bash
git clone https://github.com/avataristvan/ai-mem.git
cd ai-mem
python3 install.py
```

Restart your AI tool after installation.

## Usage

Once installed, your AI agent has four tools:

| Tool | What it does |
|------|-------------|
| `mem_add` | Store or update information |
| `mem_query` | Search memory semantically |
| `mem_list` | List all collections and their sizes |
| `mem_delete` | Delete entries or an entire collection |

### Examples

**Store something:**
> "Remember that the BLE reconnect delay is 3 seconds."

**Retrieve it later:**
> "What do we know about BLE reconnect timing?"

**For beginners:** just use the default — no configuration needed. Everything goes into a shared `workspace` collection.

**For teams:** use project-slug collections (`exodeck`, `aide`, …) to keep things organized.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_MEM_PATH` | `~/.local/share/ai-mem` | Where ChromaDB stores data |

Example:
```bash
AI_MEM_PATH=/team/shared/memory python3 install.py
```

## Update

```bash
cd ai-mem
git pull
```

No reinstall needed — the package is installed in editable mode.

## Requirements

- Python 3.10+
- pip
