# Changelog

## Unreleased

### Breaking: boost_count gate for [always-present] (commit 2088853)

Entries injected proactively at every session start (`[always-present]`) now require:

- `confidence > 0.9`
- `access_count ≥ 3`
- **`boost_count ≥ 1`** ← new

Existing entries (created before this change) have no `boost_count` key and will
no longer appear in `[always-present]` until they receive an explicit `/reflect` boost.

**Migration:** run once after `git pull`:

```bash
python3 scripts/migrate_boost_count.py --dry-run   # preview
python3 scripts/migrate_boost_count.py             # apply
```

The script sets `boost_count=1` for entries with `confidence ≥ 1.0` and
`access_count ≥ 3` — entries that were heavily used before the boost_count signal
existed. New entries (confidence = 0.7) are not affected.

---

### Breaking: confidence default changed from 1.0 to 0.7 (commit 7308f82)

New entries now start at `confidence = 0.7`. Upserts preserve the stored value.

**Effect:** New entries require 2 explicit `/reflect` boosts (`+0.1` each) to reach
`confidence > 0.9` and qualify for `[always-present]`.

No migration needed — existing entries are unaffected.

---

### Removed: ML ranker (commit a7110da)

`TorchMicroRanker`, `RankerRegistry`, `BuildFeaturesUseCase`, `TrainRankerUseCase`,
and the `[ml]` pip extra have been removed. The `mem_train` MCP tool no longer exists.

`QueryMemoryUseCase` now returns ChromaDB cosine results directly. Confidence lifecycle
(`/reflect` → `mem_boost`) is the replacement signal.

**Migration:** remove `pip install -e ".[ml]"` from any automation scripts. If you have
`.pt` ranker weight files in `~/.local/share/ai-mem/rankers/`, they can be deleted.

---

## Update procedure

```bash
cd ai-mem && git pull
python3 scripts/migrate_boost_count.py   # run after any update that lists a migration above
```

No reinstall needed — installed in editable mode.
