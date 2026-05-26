# skillopt-rs

Rust port of Microsoft's SkillOpt. Text-space optimizer for agent skills:
trains a single `skill.md` against a frozen LLM through scored rollouts,
reflection-driven structured edits, an edit-budget "textual learning rate",
a rejected-edit buffer, and a held-out validation gate.

This port targets **any OpenAI-compatible endpoint** via `OPENAI_BASE_URL` —
9router, OpenAI, vLLM, Together, Ollama, etc. No Azure required.

## Install

```
cargo build --release
cp .env.example .env && $EDITOR .env
source .env
```

## Train

```
./target/release/skillopt-train \
  --config configs/searchqa.yaml \
  --split-dir data/searchqa \
  --out-root outputs/run1
```

Resume = re-run the same command. State lives in `outputs/<run>/runtime_state.json`.

## Eval

```
./target/release/skillopt-eval \
  --config configs/searchqa.yaml \
  --skill outputs/run1/best_skill.md \
  --split valid_unseen \
  --split-dir data/searchqa
```

## Layout

```
outputs/<run>/
├── best_skill.md
├── history.jsonl
├── runtime_state.json
├── skills/skill_v0001.md ...
└── steps/step_0001/{patch.json,eval.json,record.json}
```

## Method (mirrors arxiv:2605.23904)

1. **Rollout** — frozen target runs batch with current skill. Score each item.
2. **Reflect** — optimizer model splits success/failure minibatches, proposes add/delete/replace edits.
3. **Aggregate** — merge edits across minibatches.
4. **Select** — rank by utility, clip to top `Lₜ` (cosine schedule by default).
5. **Update** — apply patch → candidate skill.
6. **Gate** — evaluate on `valid_seen`. Accept iff strictly improves. Rejected edits + score drop go into the rejected-edit buffer for the rest of the epoch.
7. **Slow update / meta skill** — epoch boundary; longitudinal comparison + optimizer-side memory.

Deployed artifact = `best_skill.md` (300–2000 tokens). Zero added inference-time calls.
