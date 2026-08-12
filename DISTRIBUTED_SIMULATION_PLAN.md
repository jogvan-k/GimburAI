# Distributed Simulation Plan

## Status

Proposed implementation plan for review. This document does not imply that the
architecture or Azure costs have been approved.

## Objective

Extend the existing self-play pipeline so its simulation step can run across the
local machine and temporary Azure GPU workers. The master remains the only process
that owns pipeline state, trains models, promotes checkpoints, runs benchmarks in
the first implementation, and decides when enough accepted games exist.

The first distributed workload is `gimbur simulate`. The job protocol should be
extensible to benchmark shards later, but distributed benchmarking is out of scope
for the initial implementation.

## Design Principles

- Preserve the current local-only pipeline as the default.
- Keep one authoritative master for models, generation state, summaries, and
  promotion decisions.
- Treat workers as disposable and interruption-safe.
- Use immutable, versioned container images. Workers never build source code.
- Use Bicep for all persistent Azure resources and role assignments.
- Use managed identities rather than storage keys or registry passwords.
- Store each game independently and atomically in Blob Storage.
- Make duplicate queue delivery, worker eviction, and master restart safe.
- Scale Azure compute to zero immediately after simulation reaches its target.
- Continue running a local simulation worker while cloud workers are active.
- Measure accepted games per hour and cost per accepted game before scaling out.

## Existing Behavior To Preserve

The current pipeline has several contracts that distributed execution must retain:

- Accepted games are top-level `data/genN/*.json` files.
- Discard diagnostics are separate from accepted games.
- Training reads complete per-game JSON objects and embedded symmetry variants.
- Gen-0 milestones extend one cumulative `data/gen0` dataset.
- Promotion retries append games to the same generation.
- Generation N uses the current champion or generation N-1 model for neural priors.
- Simulation completes when the configured number of accepted games exists, not
  when a fixed number of attempts has run.
- Existing local artifacts remain valid and resume-safe.

The current exporter writes GUID-named files directly and simulation seeds restart
after an interrupted command. Those behaviors are acceptable locally but must be
strengthened before multiple workers write to shared cloud storage.

## Proposed Architecture

```text
                         Azure subscription

  Local or Azure master                         Worker VM scale set
  +-------------------------+                  +-------------------------+
  | gimbur_nn.pipeline      |  scale/start     | node agent              |
  | distributed coordinator|----------------->| gimbur_nn.serve container|
  | local simulation worker|                  | gimbur CLI container     |
  | trainer                 |                  +------------+------------+
  | benchmark runner        |                               |
  +------------+------------+                               |
               |                                            |
               | jobs/status                  games/discards/models
               v                                            v
  +-------------------------------------------------------------------+
  | Azure Storage                                                     |
  | Queue Storage: simulation jobs and completion receipts            |
  | Blob Storage: model checkpoints, games, manifests, worker logs    |
  +-------------------------------------------------------------------+

  +----------------------+       +------------------------------------+
  | Azure Container      |------>| versioned CLI, inference, and      |
  | Registry             | pull  | worker-agent images                |
  +----------------------+       +------------------------------------+
```

### Master

The master is the existing Python pipeline with a distributed simulation backend.
It may run on a developer workstation or an Azure VM. It is responsible for:

1. Reloading and validating pipeline configuration.
2. Publishing the exact model checkpoint required by a simulation generation.
3. Creating a run manifest and deterministic simulation work shards.
4. Scaling the Azure worker pool from zero to the requested capacity.
5. Running one local simulation shard concurrently.
6. Monitoring unique accepted games in Blob Storage.
7. Validating and importing games into the local `data/genN` directory.
8. Cancelling outstanding work after the accepted target is reached.
9. Scaling the worker pool back to zero in a `finally` block.
10. Continuing with local training and benchmarks.

The master must not expose its local inference server to Azure workers. Each GPU
worker runs inference locally to avoid WAN latency and a central GPU bottleneck.

### Azure Workers

Each Azure VM hosts one independent simulation unit:

- One `gimbur-nn-serve` container loads the assigned checkpoint on its GPU.
- One `gimbur-cli` container runs one or more C# simulation processes against the
  inference container over a private Docker network.
- One lightweight `gimbur-worker-agent` claims jobs, downloads checkpoints,
  creates generated CLI configs, supervises both containers, uploads results, and
  renews queue visibility leases.

The initial worker should use one inference server per physical GPU. Simulation
parallelism is tuned per SKU rather than inferred from vCPU count.

### Azure Services

Use the minimum persistent control-plane services:

- Azure Container Registry, Basic tier initially.
- General-purpose v2 Storage Account.
- Blob containers for models, accepted games, discards, manifests, and logs.
- Queue Storage queues for simulation jobs, completion receipts, and poison jobs.
- Spot GPU Virtual Machine Scale Set in Flexible orchestration mode, capacity zero
  when idle.
- User-assigned managed identity for worker access.
- Optional Log Analytics workspace, disabled by default to control cost.

AKS is intentionally excluded from the first version. A single-purpose VM scale set
has lower operational complexity and maps directly to one GPU worker per VM. Azure
Batch can be reconsidered if queue and VM lifecycle code becomes burdensome.

## Repository Layout

Add the following version-controlled structure:

```text
infra/
  main.bicep
  modules/
    container-registry.bicep
    storage.bicep
    identity-rbac.bicep
    network.bicep
    simulation-vmss.bicep
  environments/
    dev.bicepparam
    prod.bicepparam.example
  cloud-init/
    simulation-worker.yaml

containers/
  cli/Dockerfile
  serve/Dockerfile
  worker/Dockerfile
  compose.worker.yaml

python/gimbur_nn/distributed/
  models.py
  coordinator.py
  azure_client.py
  artifact_store.py
  worker.py
  validation.py

scripts/
  build-images.sh
  deploy-azure.sh
  destroy-azure.sh
  query-spot-capacity.sh
```

Production parameter files containing subscription IDs or environment-specific
names remain outside source control.

## Bicep Infrastructure

### Deployment Scope

Use a subscription-scope `infra/main.bicep` so it can create or target a dedicated
resource group. Every resource receives common tags:

- `project=gimbur-ai`
- `environment`
- `managed-by=bicep`
- `workload=distributed-simulation`
- optional cost-center and owner tags

### Storage

Provision one StorageV2 account with:

- HTTPS-only traffic and TLS 1.2 or later.
- Public blob listing disabled.
- Shared-key access disabled after managed-identity flows are verified.
- Soft delete and blob versioning for model and manifest recovery.
- Lifecycle rules that expire worker logs and unimported scratch artifacts.
- Containers: `models`, `games`, `discards`, `manifests`, and `logs`.
- Queues: `simulation-jobs`, `simulation-receipts`, and
  `simulation-jobs-poison`.

Blob Storage is the source of truth for distributed output. Azure Files is not used
because concurrent filesystem semantics and partial-file visibility are unnecessary
risks.

### Container Registry

Provision an ACR with immutable image tags for releases. Deployments use image
digests, not mutable tags such as `latest`. Enable retention for untagged manifests.

### Identity And RBAC

Create a user-assigned worker identity with narrowly scoped roles:

- `AcrPull` on the registry.
- `Storage Blob Data Contributor` on required blob containers.
- `Storage Queue Data Contributor` on the storage account or individual queues.

The master identity requires:

- Blob and queue data access.
- Permission to read and update VMSS capacity and inspect instances.
- No permission to build or push images during normal pipeline execution.

Local masters authenticate with `DefaultAzureCredential`, normally through Azure
CLI or workload identity. No credentials are written into pipeline JSON.

### Network

Workers require outbound access to ACR, Storage, and Azure Instance Metadata. They
need no public inbound ports. Begin with service endpoints or public service
endpoints protected by identity, then add private endpoints if required by the
deployment environment.

### Compute

Provision a Spot GPU VMSS with configurable parameters:

- SKU, region, zones, maximum capacity, and initial capacity zero.
- Spot priority and explicit eviction policy.
- Configurable maximum hourly price, with `-1` supported.
- Ubuntu GPU-compatible image.
- NVIDIA driver extension or a documented CUDA image strategy.
- System disk delete-on-eviction.
- Health reporting from the worker agent.
- Cloud-init that installs Docker, authenticates through managed identity, pulls
  image digests, and starts the worker agent.

Start SKU evaluation with `Standard_NC4as_T4_v3` Spot. Compare accepted games per
dollar with `Standard_NV18ads_A10_v5`. Do not select the production SKU based only
on nominal GPU performance; benchmark the actual model and queue workload.

### Cost Controls

- Default VMSS capacity to zero.
- Define a hard `maxWorkers` configuration value.
- Add an Azure budget and alert outside the runtime scaling loop.
- Log worker allocation time, accepted games, and estimated compute cost.
- Query Spot Placement Score before scale-out and allow an ordered region/SKU
  fallback list in a later phase.

## Container Images

### `gimbur-cli`

Use a multi-stage Dockerfile:

1. Restore and publish `Gimbur.Cli` in Release mode.
2. Copy only published output into a matching .NET runtime image.
3. Run as a non-root user.
4. Expose no ports.
5. Accept the existing generated simulation config path as the command argument.

Pin the .NET SDK/runtime versions and all base image digests.

### `gimbur-nn-serve`

Build a CUDA-compatible Python inference image containing:

- The `gimbur_nn` package and serving dependencies.
- A pinned CUDA/PyTorch runtime compatible with Azure NVIDIA drivers.
- A non-root runtime user.
- A health check against `/health`.
- A mounted read-only checkpoint path.

The checkpoint is not baked into the image because every generation changes it.

### `gimbur-worker-agent`

The worker agent image contains Azure Identity, Blob, Queue, and Compute client
dependencies. It receives account, queue, container, and image information through
environment variables supplied by cloud-init.

For the initial implementation the agent may mount the Docker socket to launch the
CLI and inference images. If that security tradeoff is unacceptable, install the
agent as a systemd service on the host instead.

### Image Build And Publication

Add CI or scripts that:

1. Run .NET and Python tests.
2. Build images once on a build machine or ACR Tasks.
3. Smoke-test CLI and inference containers together.
4. Push version and Git-SHA tags.
5. Resolve and record image digests.
6. Update deployment parameters only through reviewable changes.

Workers must never execute `dotnet build`, `pip install`, or source checkout.

## Job And Artifact Protocol

### Run Manifest

Before dispatch, the master writes an immutable manifest:

```json
{
  "schemaVersion": 1,
  "runId": "...",
  "generation": 1,
  "attempt": 0,
  "targetAcceptedGames": 200,
  "simulationConfigHash": "sha256:...",
  "modelBlob": "models/.../model.pt",
  "modelSha256": "...",
  "cliImageDigest": "...",
  "serveImageDigest": "...",
  "createdAtUtc": "..."
}
```

Changing simulation settings during an active distributed step creates a new run
manifest. It must not silently mix differently configured games in one logical run.
The master can either finish the active run or cancel it and start a replacement.

### Simulation Job

Queue messages are small references to immutable job manifests. A job includes:

- Schema version, run ID, generation, attempt, and job ID.
- Requested accepted-game count.
- Reserved seed interval.
- Full simulation configuration and its hash.
- Model blob path and SHA-256 hash, or no model for Gen 0.
- Output blob prefix.
- Lease timeout and maximum attempts.

Use small shards, initially two to five accepted games. Near the global target, the
master stops issuing new jobs or reduces shard size to one game.

### Seed Allocation

Implement a central seed allocator. Every job receives a disjoint seed interval,
and the worker must stop when that interval is exhausted. Add a C# simulation option
for maximum attempts or an explicit seed range so discarded games cannot spill into
another job's allocation.

Do not derive distributed seeds solely from worker number or current accepted-file
count. Queue retries and Spot eviction would create duplicates.

### Blob Naming

Use deterministic names rather than GUID-only names:

```text
games/{runId}/gen{N}/accepted/{seed}-{contentSha256}.json
discards/{runId}/gen{N}/{seed}-{attempt}.json
logs/{runId}/{workerId}/{jobId}.json
```

Upload with an atomic single-blob commit and `If-None-Match: *`. Duplicate queue
delivery then converges on one artifact rather than creating duplicate games.

Add blob metadata for run ID, generation, job ID, worker ID, seed, config hash,
model hash, schema version, and image digest.

### Receipts And Leases

Azure Queue Storage is at-least-once delivery. The worker must:

1. Claim a message and periodically renew its visibility timeout.
2. Check whether the deterministic job receipt already exists.
3. Process the job idempotently.
4. Upload accepted and discarded artifacts.
5. Write an immutable completion receipt containing counts and hashes.
6. Delete the queue message only after the receipt is committed.

After repeated failures, move the job to the poison queue and retain logs for master
diagnostics.

## Master Pipeline Integration

### Configuration

Add an optional section to `pipeline.json`:

```json
{
  "distributedSimulation": {
    "enabled": false,
    "provider": "azure",
    "localWorker": true,
    "cloudWorkers": 2,
    "maxWorkers": 4,
    "acceptedGamesPerJob": 3,
    "pollIntervalSeconds": 5,
    "shutdownTimeoutSeconds": 120,
    "resourceGroup": "...",
    "vmScaleSet": "...",
    "storageAccount": "...",
    "jobQueue": "simulation-jobs",
    "receiptQueue": "simulation-receipts",
    "modelContainer": "models",
    "gameContainer": "games",
    "cliImage": "registry/...@sha256:...",
    "serveImage": "registry/...@sha256:...",
    "workerImage": "registry/...@sha256:..."
  }
}
```

Keep secrets out of this section. Validate that distributed configuration is
complete before provisioning or uploading anything.

### Backend Interface

Refactor `_step_simulate` behind a small backend abstraction:

```text
SimulationBackend.run(request) -> SimulationSummary
  LocalSimulationBackend
  AzureDistributedSimulationBackend
```

`SimulationRequest` contains generation, target accepted count, model, seed policy,
simulation settings, and local output path. This isolates cloud orchestration from
training, promotion, and benchmark logic.

### Distributed Step Sequence

1. Reload pipeline configuration.
2. Count and validate existing local accepted games.
3. Return immediately if the target is already met.
4. Resolve the exact model and image digests.
5. Upload the model if its content hash is absent in Blob Storage.
6. Write a run manifest.
7. Clear only stale queue messages belonging to the same cancelled run.
8. Scale VMSS to configured cloud-worker count.
9. Start the local worker through the same logical job contract.
10. Enqueue bounded seed shards.
11. Monitor receipts and accepted blobs.
12. Validate each new game before local import.
13. Import through `data/genN/.incoming`, then atomically rename into `data/genN`.
14. Stop scheduling when enough unique accepted games exist.
15. Publish cancellation for outstanding jobs.
16. Stop the local worker and wait briefly for workers to acknowledge cancellation.
17. Scale VMSS to zero in `finally`, including exceptions and Ctrl+C.
18. Verify the final accepted count and return to training.

### Master Restart

On restart, the master reads the run manifest, receipts, and blobs rather than
assuming the previous process ended cleanly. It may adopt an active compatible run
or cancel it. It must scale down orphaned worker capacity before starting an
incompatible run.

### Local Worker

The local master should contribute simulations without writing through Azure first:

- Assign it seed shards from the same allocator.
- Run the existing CLI/inference processes locally.
- Import local output using the same validator and deduplication rules.
- Optionally upload accepted local games to Blob Storage so cloud storage remains a
  complete distributed run archive.

The local worker is disabled automatically when no compatible GPU/inference model is
available, unless the generation uses the Gen-0 greedy prior.

## Data Validation And Exact Completion

Before counting a blob as accepted, validate:

- JSON parses completely.
- Export schema version is supported.
- Generation/run metadata matches.
- Map, player count, and simulation config hash match the run manifest.
- Model hash matches for neural-prior generations.
- Seed is inside the assigned interval and has not already been accepted.
- Required board, state, action, and diagnostic fields exist.
- Optional content hash matches the downloaded bytes.

Do not count blob listings alone. The master should maintain a durable accepted-game
index in the run manifest area or reconstruct it from validated deterministic blob
names.

Concurrent workers can finish after the target is reached. Keep extra valid games in
Blob Storage, but import only a deterministic subset unless the configured policy
explicitly permits overshoot. For cumulative Gen-0 milestones, preserved extras may
be imported at the next milestone after validation.

## Cancellation And Shutdown

The master writes a cancellation marker under the run prefix. Workers check it:

- Before claiming a new job.
- Before starting each game or small simulation chunk.
- Before uploading nonessential logs.

Shutdown order:

1. Stop enqueueing jobs.
2. Publish cancellation.
3. Stop local simulation.
4. Wait for active workers up to the configured timeout.
5. Set VMSS capacity to zero.
6. Record unfinished jobs as reclaimable.

Spot eviction is treated like an unacknowledged queue message. Another worker may
claim it after visibility timeout and deterministic uploads prevent duplication.

## Observability

Write structured worker/job summaries containing:

- Worker and VM instance IDs.
- SKU and region.
- Image and model digests.
- Job start/end timestamps.
- Attempted, accepted, discarded, and uploaded counts.
- Accepted games/hour.
- Prior and leaf model invocation diagnostics.
- Inference errors, fallbacks, queue latency, and GPU information.
- Exit reason, including eviction or cancellation where observable.

The master logs aggregate progress periodically:

```text
accepted=143/200 local=31 azure=112 activeJobs=4 queuedJobs=2 workers=2
```

Log Analytics integration is optional. Blob logs are sufficient for the first
version and avoid an always-on ingestion cost.

## Security

- No public inference endpoints.
- No SSH requirement for normal operation; use Azure Run Command for diagnostics.
- No account keys, SAS tokens, or registry passwords in source or queue messages.
- Managed identities scoped to required resources.
- Containers run non-root with read-only model mounts.
- Pin all images by digest and verify checkpoint hashes.
- Treat queue messages and blobs as untrusted input during parsing.
- Never let workers update model, summary, promotion, or benchmark artifacts.

## Testing Strategy

### Unit Tests

- Distributed config parsing and validation.
- Seed interval allocation and exhaustion.
- Job serialization and schema-version rejection.
- Config/model hash computation.
- Blob naming and idempotent duplicate handling.
- Accepted-game validation and deterministic import selection.
- Target and overshoot calculations.
- Cancellation and scale-to-zero `finally` behavior.

### Integration Tests With Azurite

- Queue claim, visibility renewal, retry, and poison handling.
- Atomic blob uploads and duplicate `If-None-Match` behavior.
- Master restart with jobs in queued, active, and completed states.
- Worker failure after upload but before queue acknowledgement.
- Partial/corrupt blob rejection.

### Container Tests

- CLI container runs one deterministic no-prior smoke game.
- Inference container loads a small test checkpoint and passes `/health`.
- CLI reaches inference by Docker service name.
- Worker agent downloads a model, starts services, uploads a game, and exits.
- SIGTERM and cancellation leave no partially counted artifact.

### Azure Dev Environment Tests

- Deploy Bicep from an empty resource group.
- Verify RBAC without storage keys.
- Scale VMSS from zero to one and back to zero.
- Run one Gen-0 job and one neural-prior job.
- Simulate a Spot eviction or force-delete an instance mid-job.
- Resume and verify no duplicate accepted seed.
- Confirm the master scales to zero after success, error, and Ctrl+C.

### Performance Tests

For every candidate SKU, capture:

- Boot-to-ready time.
- Accepted games/hour.
- GPU utilization and memory.
- Mean inference latency and queue depth.
- Error/fallback/discard rates.
- Spot price and estimated cost per accepted game.

Use these measurements to choose worker parallelism and SKU. The fastest GPU is not
necessarily the best cost-per-game option.

## Implementation Phases

### Phase 0: Correctness Prerequisites

1. Add atomic local per-game export using temporary file plus rename.
2. Add explicit distributed run/job/config/model metadata to exports or blob
   metadata.
3. Add bounded seed intervals and maximum attempts to the CLI.
4. Add a reusable accepted-game validator.
5. Add deterministic deduplication by run and seed.

Exit criterion: interrupted and repeated local shards cannot create duplicate or
partially counted accepted games.

### Phase 1: Containerization

1. Add CLI, inference, and worker-agent images.
2. Add local Compose topology and smoke tests.
3. Pin toolchains and image digests.
4. Add build/push scripts or CI.

Exit criterion: a clean GPU host can run a neural-prior simulation using only image
pulls, a checkpoint, and configuration.

### Phase 2: Azure Infrastructure

1. Implement Bicep modules and environment parameters.
2. Deploy ACR, Storage, queues, identity/RBAC, network, and capacity-zero VMSS.
3. Add deployment and teardown scripts.
4. Document quota and Spot Placement Score checks.

Exit criterion: repeatable Bicep deployment produces an idle, zero-compute-cost
worker environment with validated identity access.

### Phase 3: Worker Protocol

1. Implement queue job models and worker agent.
2. Implement model download/cache and hash verification.
3. Supervise inference and CLI containers.
4. Upload artifacts and receipts idempotently.
5. Implement cancellation, lease renewal, retries, and poison handling.

Exit criterion: one Azure worker can survive duplicate delivery and master restart
without duplicate accepted games.

### Phase 4: Master Integration

1. Introduce simulation backend abstraction.
2. Implement Azure coordinator and local concurrent worker.
3. Add VMSS scaling, monitoring, import, cancellation, and guaranteed scale-down.
4. Integrate Gen-0 milestones, normal generations, and promotion retries.
5. Preserve local-only default behavior.

Exit criterion: an existing pipeline configuration can opt into distributed
simulation and continue automatically into local training.

### Phase 5: Hardening And Cost Tuning

1. Run eviction, timeout, corruption, and quota-failure tests.
2. Tune shard size and simulation parallelism.
3. Compare T4 and A10 accepted games per dollar.
4. Add budget alerts and orphan-resource cleanup.
5. Document operations and incident recovery.

Exit criterion: repeated production-like runs leave no orphaned compute and produce
the same validated training-data contract as local simulation.

### Future Phase: Distributed Benchmarks

Extend the job protocol with a benchmark job type only after simulation is stable.
Benchmark shards require deterministic seat/seed assignment and a purpose-built
result merger for wins, draws, labels, confidence intervals, per-game records, and
NN diagnostics. The master remains the sole writer of final benchmark JSON and
pipeline summaries.

## Operational Runbook

The initial operational workflow should be:

1. Deploy or update infrastructure with Bicep.
2. Build and publish immutable images.
3. Check regional quota, Spot Placement Score, current price, and eviction history.
4. Set image digests and Azure resource names in pipeline configuration.
5. Start the normal pipeline on the master.
6. Observe accepted games/hour and discard/error thresholds.
7. Confirm VMSS capacity returns to zero before training begins.
8. Deallocate or delete any manually created diagnostic resources.

Provide a separate cleanup command that detects active Gimbur VMSS instances and
requires explicit confirmation before scaling them to zero.

## Decisions Required Before Implementation

1. Should Bicep create the resource group at subscription scope, or target an
   existing resource group?
2. Which Azure region and fallback regions are allowed?
3. What Spot maximum price and eviction policy should be used?
4. Is mounting the Docker socket into the worker agent acceptable?
5. Should valid overshoot games be imported immediately, saved for the next
   cumulative target, or retained only as cloud archive?
6. Should Blob Storage be a complete archive of local-worker games as well as cloud
   games?
7. Is optional Log Analytics worth the ingestion cost?
8. Should the first deployment support only one VMSS/SKU, or an ordered fallback
   list from the beginning?
9. Is distributed benchmarking explicitly deferred, as assumed by this plan?

## Definition Of Done

The distributed simulation feature is complete when:

- Infrastructure deploys reproducibly from Bicep.
- Workers start from immutable images without source builds.
- The master and Azure workers simulate concurrently.
- All workers use the correct checkpoint and simulation configuration.
- Accepted-game targets use validated unique games, not attempt or blob counts.
- Worker eviction and duplicate queue delivery do not duplicate training samples.
- Pipeline restart resumes active or completed distributed work safely.
- The master always scales Azure compute to zero after simulation.
- Training and existing benchmark/promotion flows continue unchanged.
- Local-only pipeline behavior remains fully supported.
- Cost and throughput metrics are available per Azure SKU and run.
