# Caller-controlled inference URL enables SSRF and resource exhaustion

**Severity:** Critical

`src/Gimbur.Server/Program.cs:25-26,88-109` accepts `nnUrl` from an unauthenticated request and passes it to permanent static pools. `PriorClientPool.cs:17-26` and `CatanStateLeafEvaluatorPool.cs:7-10` create long-lived HTTP clients and background workers for every distinct URL, with no allowlist, cardinality bound, eviction, or shutdown disposal.

An external caller can make the server contact loopback/private/metadata endpoints and permanently consume threads, sockets, and memory by submitting unique URLs.

## Reproduction

Send valid `mcts-nn-ai` requests with unique `nnUrl` values pointing at a controlled or internal host. Observe requests to inference paths and a growing number of pooled workers.

## Recommended fix

Configure inference endpoints at startup instead of accepting them in request bodies. Add authentication, URI/address allowlisting, rate and concurrency limits, bounded lifecycle-managed pools, and shutdown disposal.
