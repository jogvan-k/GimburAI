# Self-Play Training Pipeline

The pipeline is current-only and trains the complete `catan_policy_value_v1`
full-state policy/value model described in
[complete-policy-value-model.md](complete-policy-value-model.md). Placement-only,
combined dual-model, and placement-and-state pipeline modes are not supported.

## Generation Flow

Each generation simulates complete full-state roots, trains one model, then runs
the configured benchmarks. Generation N resumes from generation N-1 and uses it
for neural priors during simulation.

```text
pipeline/data/genN/*.json
pipeline/models/genN.pt
pipeline/results/genN/{benchmark}.json
```

Run from the repository root:

```bash
python -m gimbur_nn.pipeline --config pipeline.json
```

The JSON config accepts `//` comments. See
[`pipeline.example.json`](../pipeline.example.json) for all commonly used fields.
The top-level model fields are `mapConfig`, `gameConfig`, and `modelConfig`; there
are no model-type, architecture, placement-model, or training-mode selectors.

## Training

The `train` section controls epochs, patience, batch size, learning rate, data
splits, replay generations, value blending, victory-point sampling, and value and
policy loss weights. Training always consumes full-state samples and always
optimizes both heads. Checkpoints require version 5 and architecture
`catan_policy_value_v1`.

## Serving

The pipeline starts `gimbur_nn.serve` with one model. Serving exposes full-state
parent policy/value prediction and state leaf-value batching under `/state/...`.
There are no placement endpoints or placement model arguments.

## Promotion And Benchmarks

Promotion remains optional. Generation 0 bootstraps the champion; later
challengers may be tested against greedy and the current champion using direct
and MCTS benchmark gates. Draws count as half a win. Failed gates may append
additional self-play games and retrain up to `maxRetries`.

Normal benchmarks run after training and write one result per configured entry.
`summary.json` and `progress.png` track results across generations. Baseline
benchmarks, chart-only regeneration, resume detection, and per-generation
checkpoint recovery remain supported.

## Resume

Without `--start-gen`, the pipeline resumes at the first generation missing any
of these markers:

1. `data/genN/` contains the configured number of accepted JSON games.
2. `models/genN.pt` exists.
3. Every configured benchmark result exists.

Use `--start-gen N` to override detection or `--chart-only` to regenerate the
progress chart from existing results.
