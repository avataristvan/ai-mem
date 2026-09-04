# Collections and Memory Scoping

## Auto-detection

`repo_context.py` detects the correct collection from the working directory:

| Condition | Collection |
|-----------|------------|
| `CLAUDE.md` at git root | `repo.<repo-name>` |
| `CLAUDE.md` in monorepo subdir | `repo.<repo-name>.<subdir>` |
| No `CLAUDE.md` found | `workspace` |
| Cross-session general knowledge | `global` |

The collection name is derived from the git remote URL slug or directory name.

## Entry Types

The optional `type` metadata field on `mem_add` tags an entry for filtering (`mem_query(type=...)`, `mem_topics`) and for `mem-dream`'s consolidation rules. Canonical types, each with a fixed structure enforced by convention (not schema validation):

| Type | Structure | Use for |
|---|---|---|
| `pattern` | `Rule: / When: / Why:` | A reusable technique or best practice |
| `anti-pattern` | `Tried: / Failed because: / Instead:` | A documented failure mode |
| `dilemma` | `Tension: / Context A: / Context B: / Affected A: / Affected B: / Questions:` | A genuine value conflict with no single correct answer |
| `domain` | `Decision: / Why: / Scope:` | Business/domain knowledge — why a rule exists, a domain constraint — as opposed to a coding technique |
| `feedback`, `project`, `reference`, `user` | free-form | Descriptive entries, no fixed structure |

`domain` entries should be written as soon as the knowledge surfaces during coding, not deferred to an end-of-session reflection step — that kind of insight is easy to lose if it isn't captured immediately. There's no hook that forces this (recognizing a domain decision requires semantic judgment a deterministic hook can't reliably make); it's agent discipline, reinforced by the `type` description on `mem_add` itself and by the `knowledge_architecture_framework` entry in the `global` collection.

`mem-dream` never merges a `pattern`/`anti-pattern` entry with a `domain` entry — technique and business rationale are different axes even when they discuss the same code.

## Special Entries

- **`current_focus`** (id `"current_focus"`) — primary context entry per collection; the SessionStart hook injects it as `[scope focus]` text
- The Stop hook reminds the agent to update `current_focus` whenever files changed during the session

## Collection Lifecycle

### First Session (auto-seed)
When a `repo.*` collection is empty on SessionStart and the global `injection_rate >= 0.60`:
1. `repo_seeder.py` reads CLAUDE.md from the git root
2. Splits it into H2 sections (`## ` headings only; H1 intro excluded)
3. Stores each section as `seed_<slug>` entry with `seeded_from: "CLAUDE.md"` metadata
4. Gate ensures seeding only activates once memories are actually being used (not on day 1)

### TTL Expiry
Set `ttl_days` on `mem_add` to auto-expire entries. `mem_cleanup` removes expired entries.

### Forgetting Curve
`mem_cleanup stale_after_days=N` removes entries not accessed within N days — useful for pruning outdated knowledge.

## mem_delete Semantics

- `mem_delete(collection="foo", ids=["a","b"])` — deletes specific entries, returns count
- `mem_delete(collection="foo")` — drops the entire collection, returns `-1`

## Routing at Query Time

Always pass the collection explicitly to `mem_add` and `mem_query`. The SessionStart hook prints the active collection:

```
Active collection: "repo.ai-mem". Pass collection="repo.ai-mem" to mem_add and mem_query in this session.
```

Use `global` for knowledge that should persist across all repos (user preferences, cross-project patterns).
