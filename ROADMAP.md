# Roadmap

This roadmap aligns with `PRD.md` and targets a web/cloud platform with Dockerized CPU/GPU workers.

## Phase 0: Planning and Architecture (1-2 weeks)

Deliverables:

1. Final PRD sign-off and requirement IDs lock
2. Architecture decision records (API, queue, storage, worker model)
3. Baseline labeled dataset and evaluation metrics

Exit criteria:

1. MVP scope approved
2. Build/deploy strategy approved for local and cloud

## Phase 1: Pipeline Refactor and Container Baseline (2-3 weeks)

Deliverables:

1. Extract processing pipeline into worker-friendly modules
2. Introduce job schema for input, config, status, and artifacts
3. Build container strategy for `cpu` and `gpu` images

Exit criteria:

1. Pipeline can run independently from UI
2. Repeatable Docker builds for CPU/GPU targets

## Phase 2: Cloud MVP Platform (4-6 weeks)

Scope focus:

- `FR-INGEST-*`
- `FR-CAM-01` to `FR-CAM-05`
- `FR-EVENT-01` to `FR-EVENT-07`
- `FR-EVENT-09`
- `NFR-DEPLOY-*`

Deliverables:

1. API for uploads, jobs, and output retrieval
2. Queue + worker orchestration with retries
3. Object storage and relational database integration
4. Web UI for upload, status, and download

Exit criteria:

1. Full-match upload and async processing works end-to-end
2. Follow-cam, timeline, and base highlight outputs are produced

## Phase 3: Editor and Analytics (3-5 weeks)

Scope focus:

- `FR-EDITOR-*`
- `FR-TEAM-*`
- `FR-PLAYER-01`
- `FR-PLAYER-04`
- `FR-LEARN-01` to `FR-LEARN-03`
- `FR-AGENT-01`

Deliverables:

1. Timeline editor with event navigation and clip adjustments
2. Annotation tools and clip tagging
3. Momentum graph, heatmaps, and position summaries
4. Export presets (clip, montage, player package)
5. Structured reviewer feedback capture in editor workflows
6. Basic natural-language event query assistant

Exit criteria:

1. Coach review workflow is complete in web UI
2. Team analytics and editable highlights are shippable

## Phase 4: Advanced Player Intelligence and Live (4-8 weeks)

Scope focus:

- `FR-EVENT-08`
- `FR-PLAYER-02`
- `FR-PLAYER-03`
- `FR-LIVE-*`
- `FR-QA-01` to `FR-QA-04`
- `FR-LONG-01` to `FR-LONG-03`
- `FR-SCOUT-01` to `FR-SCOUT-03`
- `FR-AGENT-02` to `FR-AGENT-03`
- `FR-LEARN-04`

Deliverables:

1. Jersey-assisted player identification pipeline with manual correction UX
2. Foul candidate detection with confidence markers
3. Live stream ingestion and near-real-time marker pipeline
4. Instant replay bookmarks
5. Confidence calibration and human-review queue workflow
6. Season trend dashboards and opponent scouting report generation
7. Agent explainability summaries and review-priority suggestions
8. Automated error trend reporting

Exit criteria:

1. Advanced features reach target quality thresholds
2. Live feature set validated for supported deployment profiles

## Phase 5: Hardening and Scale (ongoing)

Scope focus:

- `NFR-OPS-01`
- `NFR-SEC-01` to `NFR-SEC-03`
- `NFR-DATA-01` to `NFR-DATA-02`
- `NFR-PLAT-01`
- `NFR-COMP-01`
- `NFR-COST-01`
- `FR-INT-01` to `FR-INT-04`
- `FR-FLEET-01` to `FR-FLEET-04`
- `FR-AGENT-04`
- `FR-LEARN-05` to `FR-LEARN-06`
- `NFR-AGENT-01` to `NFR-AGENT-03`

Deliverables:

1. Autoscaling and queue priority controls
2. Production observability and alerting
3. Retention/deletion policy enforcement
4. Multi-tenant hardening and cost controls
5. Open API and webhook ecosystem
6. Edge/camera fleet diagnostics and remote operations controls
7. LLM provider abstraction and governance controls
8. Feedback-driven retraining pipeline with staged rollout gates

Exit criteria:

1. Stable operations with measurable SLOs
2. Predictable cost and reliability under load

## Phase 6: Coaching, Recruiting, and Commercial Expansion (ongoing)

Scope focus:

- `FR-COACH-01` to `FR-COACH-03`
- `FR-RECRUIT-01` to `FR-RECRUIT-03`
- `FR-DIST-01` to `FR-DIST-04`
- `FR-COLLAB-01` to `FR-COLLAB-04`
- `NFR-API-01`

Deliverables:

1. Drill recommendation and coaching action-plan engine
2. Recruiting profile, comparison, and verified sharing workflows
3. Social distribution templates, sponsor overlays, and rights controls
4. Collaboration threads, approval workflow, and audit history

Exit criteria:

1. Coaching and recruiting workflows produce measurable user retention gains
2. Distribution and monetization features are production-ready for target customer segments

## Milestones

### M1 (End of Phase 2)

1. Full-match upload
2. Async jobs with retries
3. Follow-cam + base event timeline + highlight export
4. Docker deployment with CPU/GPU worker support

### M2 (End of Phase 3)

1. Full editor workflow
2. Team analytics in UI
3. Player spotlight export improvements
4. Reviewer feedback capture and basic AI query assistant available

### M3 (End of Phase 4)

1. Jersey-assisted identity workflow
2. Foul candidate markers
3. Live and instant replay capabilities for supported environments
4. Confidence review workflows and scouting reports live

### M4 (End of Phase 5)

1. Open API/webhook integrations
2. Fleet and edge operations tooling
3. Security/compliance and data residency controls enforced
4. LLM provider abstraction and retraining governance operational

### M5 (End of Phase 6)

1. Coaching action engine and recruiting suite released
2. Collaboration governance and commercial distribution toolkit launched

## Suggested Team Shape

1. Backend/platform engineer
2. CV/ML engineer
3. Frontend engineer
4. DevOps/platform engineer
5. QA + data labeling analyst
