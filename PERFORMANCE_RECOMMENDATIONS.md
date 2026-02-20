# Performance Recommendations

This document captures high-value next steps beyond the optimizations already implemented in the local pipeline.

## Priority 1: Platform Throughput

1. Move processing to queued asynchronous jobs with worker autoscaling.
2. Separate analysis workers from rendering workers.
3. Persist intermediate artifacts in object storage for retry-safe stages.

Expected outcome:

- Better throughput under concurrent workloads
- More reliable reruns and lower restart cost

## Priority 2: Model and Pipeline Efficiency

1. Add optional frame-skipping and interpolation profiles.
2. Introduce caching for repeated runs with unchanged inputs/config.
3. Evaluate detector upgrades only when quality gain justifies latency increase.

Expected outcome:

- Lower GPU minutes per match
- Faster turnaround for iterative workflows

## Priority 3: Media Processing

1. Keep NVENC as preferred encoder on supported hardware.
2. Add profile-based encoding settings (`fast`, `balanced`, `quality`).
3. Segment very long matches into resumable chunks.

Expected outcome:

- Reduced encoding bottlenecks
- Better failure recovery on long jobs

## Priority 4: Quality Operations and Observability

1. Capture stage-level metrics (`track`, `detect`, `render`, `export`).
2. Store event confidence plus reviewer correction feedback.
3. Run regression benchmarks against labeled datasets on each major change.

Expected outcome:

- Controlled quality evolution with measurable performance impact

## Recommended Benchmark Matrix

Benchmark dimensions:

1. Resolution and frame rate: 1080p30, 1080p60, 4K30
2. Match length: 20, 45, 90 minutes
3. Compute mode: CPU-only and GPU-enabled
4. Feature mode: overlay on/off, audio on/off

Track metrics:

1. End-to-end wall time
2. GPU utilization and peak memory
3. Event precision/recall on labeled set
4. Cost per processed match