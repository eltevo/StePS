# Perf note: host-side octree build in the PDS GPU Barnes-Hut force

Observed on the 6.3M-particle matched-resolution run (`test256`, 4× H200) at z=31:

```
PDS Barnes-Hut force ... Wall-clock 1.89s  [host tree build 0.55s (29%), GPU section 1.02s]
```

The host-side flatten is ~30% of the force eval at this N and will grow as structure
clusters (deeper trees, more nodes), so it is the next scaling target after the GPU walk.

## What is already parallel

In `forces_pds_bh_cuda` (`src/forces_cuda.cu`, ~lines 1202–1231):
- `Max_radius` reduction — `#pragma omp parallel for reduction(max:)` ✓
- Morton key computation — `#pragma omp parallel for` ✓
- key sort — `__gnu_parallel::sort` ✓
- `keys`/`tree` are `static std::vector` → capacity is retained across force evals, so
  there are no per-step reallocations after the first call. ✓

## The remaining serial bottleneck

`pds_morton_build` (`src/forces_cuda.cu`, ~lines 1084–1120) is a **serial recursive DFS**
that emits the flattened nodes (~7.5M for this run) into `out` in preorder with per-node
`escape` (skip) pointers.  It is serial because the DFS append order and the escape
offsets are inherently sequential.  At ~0.55 s for ~7.5M nodes this is ~14M nodes/s on one
core — the whole 16-thread budget is idle during this phase.

## Proposed optimization — task-parallel subtree flatten

The keys are already Morton-sorted, so the ≤8 children of the root (and their children)
occupy **contiguous, disjoint** key ranges that can be flattened independently:

1. Serially build the root and descend the first `L` Morton levels (e.g. L=2 → up to 64
   independent subtree ranges).  Cheap (≤ a few hundred nodes).
2. For each level-`L` range, `#pragma omp task` → flatten into a **thread-local**
   `std::vector<PDSNodeGPU>` via the existing recursion.
3. Concatenate the thread-local buffers in order; while copying buffer *b* at global
   offset `O_b`, add `O_b` to every `escape` field (and any child indices) of its nodes.
   Patch the stub parent nodes' `escape`/child pointers to the concatenated offsets.

This keeps the exact same flattened layout (so the GPU kernel is unchanged) and parallelizes
the dominant cost.  Expected ~`min(nthreads, #subtrees)`× on the build → ~0.55 s → ~0.1 s
with 8–16 threads, cutting the host fraction from ~30% to ~5–8%.

### Lower-risk partial step

Even just parallelizing the **leaf-aggregate COM sums** is not the win (they are cheap);
the node *emission* is the cost, so the task-parallel flatten above is what matters.  A
`tree.reserve(2*N+16)` before the build is harmless insurance but largely redundant given
the static vector.

## Notes / caveats for implementation

- Validate against the current serial build: the flattened arrays must be **bit-identical**
  in node count, COM, mass, `nodesize`, `is_leaf`, and `escape` (the GPU walk depends on
  the escape pointers).  Add a debug compare on a small run.
- The CPU direct/CPU-BH paths (`forces.cc`) use the recursive `OctreeNode` build and are a
  separate code path; this note is only about the GPU-BH host flatten.
- Implement and test **after** the current `test256` run frees the GPUs (the kernel output
  must be regression-checked, e.g. via `examples/pds_tests/run_tests.py` and a GPU-vs-CPU
  force diff), so it does not perturb in-flight results.
