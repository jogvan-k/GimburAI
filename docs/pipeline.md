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

### Gen-0 Milestone Sweep

Set `gen0Milestones` to cumulative counts such as
`[200, 400, 600, 800, 1000]`. One shared `data/gen0` directory is extended only
by the games missing from the next target. Each milestone trains a separate
model from scratch on all accumulated games and runs the full benchmark suite.

Models and results are stored under `models/bootstrap/{games}` and
`results/bootstrap/{games}`. The pipeline writes `bootstrap-summary.json` and
`bootstrap-progress.png`. The final milestone becomes the Gen-0 champion, then
normal processing continues at Gen 1. Existing milestone artifacts are reused
after interruption.

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

The optional `inference` section can export an inference-only FP16 checkpoint
beside each FP32 training checkpoint (`model.fp16.pt`). FP32 remains authoritative
for training and resume. `simulationPrecision` selects the model used for neural
self-play, `benchmarkPrecisions` runs separately named benchmark variants, and
`promotionPrecision` selects the model used by promotion gates. FP16 serving
requires CUDA.

`serve.batchWindowMs` optionally waits after the first queued prior or leaf
request arrives, allowing nearby requests to share one GPU batch. `0` disables
the window; fractional millisecond values are supported.

`serve.compileModel` compiles the CUDA inference model with `torch.compile` and
dynamic batch shapes, then performs a one-state warmup before the health endpoint
becomes available. Startup can take several minutes on the first compilation.

The optional `monitoring` section starts a pipeline-managed resource monitor for
each simulation, training, and benchmark subprocess. It writes append-only JSONL
records for utilization over each configured interval, including system/process
CPU and memory, NVIDIA GPU utilization/power/temperature, and inference
`/diagnostics` deltas when available. `intervalSeconds: 120` records two-minute
intervals and flushes a final partial interval when a step stops.

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
