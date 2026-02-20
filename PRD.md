# Product Requirements Document (PRD)

## 1. Purpose

Define the target product requirements for rebuilding Video Highlights as a web/cloud-enabled platform while preserving local and Docker execution options.

## 2. Product Vision

Enable coaches and families to upload full soccer match recordings and receive searchable events, player spotlights, analytics, and editable highlight reels with minimal manual effort.

## 3. Goals

1. Support full-match video ingest and asynchronous processing.
2. Generate follow-cam output from panoramic or wide-field recordings.
3. Detect key match events and produce highlight suggestions.
4. Provide player and team analytics for review workflows.
5. Deliver a Docker-deployable architecture with optional GPU acceleration on local hosts and cloud/edge nodes.
6. Build a decision-intelligence platform for coaching, recruiting, and club operations.
7. Add an AI copilot layer for natural-language analysis and structured feedback-driven model improvement.

## 4. Scope and Constraints

### 4.1 In Scope

1. Web application and API platform
2. Background worker architecture (CPU and GPU)
3. Timeline editor and export workflows
4. Batch/offline processing and phase-gated live features

### 4.2 Out of Scope for MVP

1. Referee-grade foul adjudication accuracy guarantees
2. Fully autonomous player identity without manual correction path
3. Hard real-time SLA guarantees across all deployment profiles

## 5. Personas

1. Coach: reviews events, creates team clips, shares feedback.
2. Parent: generates player-specific reels and downloads highlights.
3. Club admin: manages teams, users, quotas, and reliability.

## 6. Functional Requirements

### 6.1 Ingest and Job Management

- `FR-INGEST-01`: Upload full-match videos, including long-duration games.
- `FR-INGEST-02`: Support resumable uploads and durable storage.
- `FR-INGEST-03`: Run processing asynchronously with stage-level progress.
- `FR-INGEST-04`: Persist job metadata, artifacts, and retry history.

### 6.2 Camera/Tracking Intelligence

- `FR-CAM-01`: Ingest panoramic or wide-field video (including 180-degree style captures).
- `FR-CAM-02`: Generate automated follow-cam output with AI pan/zoom behavior from fixed-camera recordings.
- `FR-CAM-03`: Track ball and players across the full match timeline.
- `FR-CAM-04`: Support 4K-class source input pipelines, including dual-lens panoramic camera output formats.
- `FR-CAM-05`: Support unattended fixed-camera capture workflows (no dedicated camera operator needed for pan/zoom actions).

### 6.3 Event Detection

- `FR-EVENT-01`: Detect and timestamp goals.
- `FR-EVENT-02`: Detect and timestamp shots.
- `FR-EVENT-03`: Detect and timestamp corner kicks.
- `FR-EVENT-04`: Detect and timestamp penalty kicks.
- `FR-EVENT-05`: Detect and timestamp free kicks.
- `FR-EVENT-06`: Detect and timestamp goal kicks.
- `FR-EVENT-07`: Detect and timestamp kickoffs.
- `FR-EVENT-08`: Detect and timestamp likely fouls with confidence scoring.
- `FR-EVENT-09`: Provide confidence score per event.

### 6.4 Player-Centric Features

- `FR-PLAYER-01`: Generate player spotlight clips.
- `FR-PLAYER-02`: Support jersey-number assisted player identification.
- `FR-PLAYER-03`: Provide manual correction workflow for identity assignment.
- `FR-PLAYER-04`: Track player movement paths and per-player involvement metrics.

### 6.5 Team Analytics

- `FR-TEAM-01`: Provide match momentum visualization over timeline windows.
- `FR-TEAM-02`: Generate team heatmaps.
- `FR-TEAM-03`: Provide position and shape summaries.

### 6.6 Review and Editor

- `FR-EDITOR-01`: Allow switching between panoramic and follow-cam views.
- `FR-EDITOR-02`: Support zoom/pan inspection in review mode.
- `FR-EDITOR-03`: Provide timeline navigation with event jump points.
- `FR-EDITOR-04`: Provide drawing/annotation tools (arrows, shapes, notes).
- `FR-EDITOR-05`: Enable custom highlight creation and trimming.
- `FR-EDITOR-06`: Support tagging players and events on clips.
- `FR-EDITOR-07`: Support export and share flows for clips and montages.

### 6.7 Live and Instant Playback (Phase-Gated)

- `FR-LIVE-01`: Support live streaming input.
- `FR-LIVE-02`: Surface near-real-time event markers.
- `FR-LIVE-03`: Support instant replay bookmarks.

### 6.8 Data Trust and Quality Operations

- `FR-QA-01`: Calibrate event and tracking confidence by competition level, camera angle, and field conditions.
- `FR-QA-02`: Route low-confidence detections to human-review queues.
- `FR-QA-03`: Maintain benchmark datasets and recurring evaluation reports.
- `FR-QA-04`: Expose event evidence frames and rationale metadata for analyst review.

### 6.9 Longitudinal Player and Team Intelligence

- `FR-LONG-01`: Track season and career trends for players and teams.
- `FR-LONG-02`: Provide role-based benchmarks against peer cohorts.
- `FR-LONG-03`: Generate improvement and decline alerts from longitudinal metrics.

### 6.10 Opponent Scouting Automation

- `FR-SCOUT-01`: Generate opponent pattern reports (build-up, transition, and defensive shape).
- `FR-SCOUT-02`: Auto-detect and summarize set-piece tendencies.
- `FR-SCOUT-03`: Produce opponent threat maps and danger-zone summaries.

### 6.11 Coaching Action Engine

- `FR-COACH-01`: Recommend training drills from detected weaknesses with linked video evidence.
- `FR-COACH-02`: Generate weekly coaching plans with clip bundles and priorities.
- `FR-COACH-03`: Track completion and outcomes for coaching action items.

### 6.12 Recruiting and Exposure Workflows

- `FR-RECRUIT-01`: Build verified player profiles with tagged evidence clips.
- `FR-RECRUIT-02`: Provide recruiter-facing share links and controlled access packages.
- `FR-RECRUIT-03`: Support player-to-cohort comparison views for scouting decisions.

### 6.13 Distribution and Monetization

- `FR-DIST-01`: Provide social-first export profiles for major platforms.
- `FR-DIST-02`: Support sponsor overlays and branded templates.
- `FR-DIST-03`: Support fan distribution models (subscription or pay-per-view where applicable).
- `FR-DIST-04`: Apply watermarking and rights controls on distributed assets.

### 6.14 Integrations and Open Platform

- `FR-INT-01`: Provide public APIs and webhooks for downstream systems.
- `FR-INT-02`: Sync roster, schedule, and match metadata with external club tools.
- `FR-INT-03`: Support enterprise authentication integrations (SSO).
- `FR-INT-04`: Provide standards-based data export for analytics interoperability.

### 6.15 Camera Fleet and Edge Operations

- `FR-FLEET-01`: Monitor camera and edge node health status remotely.
- `FR-FLEET-02`: Provide calibration and setup validation diagnostics.
- `FR-FLEET-03`: Support offline capture with deferred upload/sync.
- `FR-FLEET-04`: Provide deployment diagnostics for site-level troubleshooting.

### 6.16 Collaboration and Governance

- `FR-COLLAB-01`: Enforce role-based permissions for coaches, analysts, players, and admins.
- `FR-COLLAB-02`: Support clip approval workflows before publication/sharing.
- `FR-COLLAB-03`: Provide threaded collaboration on annotations and clips.
- `FR-COLLAB-04`: Keep audit trails for edits, approvals, and shares.

### 6.17 AI Agent Copilot

- `FR-AGENT-01`: Provide natural-language match query workflows over detected events and tracked entities.
- `FR-AGENT-02`: Generate explainability summaries for detected events using timestamped evidence artifacts.
- `FR-AGENT-03`: Suggest candidate missed events for analyst review using low-confidence and temporal-context signals.
- `FR-AGENT-04`: Support configurable LLM API provider integration for agent capabilities.
- `FR-AGENT-05`: Keep CV/event models as authoritative event source while the agent acts as reasoning and workflow assistant.

### 6.18 Feedback Capture and Continuous Learning

- `FR-LEARN-01`: Capture structured reviewer feedback for false positives, missed events, wrong timestamps, and misattributions.
- `FR-LEARN-02`: Store feedback with event evidence, reviewer metadata, and decision status for curation.
- `FR-LEARN-03`: Route feedback into labeling/review queues with approval controls.
- `FR-LEARN-04`: Produce periodic quality reports (top misses, top false positives, class-specific error trends).
- `FR-LEARN-05`: Support dataset versioning and retraining triggers from approved feedback cohorts.
- `FR-LEARN-06`: Validate retrained models against benchmark datasets before staged rollout.

### 6.19 Multi-Tenant and Administration

- `FR-TENANT-01`: Support multi-tenant isolation where each tenant has independent matches, jobs, events, feedback, and training records.
- `FR-TENANT-02`: Provide a global admin control plane for tenant lifecycle management (create/update tenants, users, memberships).
- `FR-TENANT-03`: Provide a tenant-admin control plane for tenant-scoped user and role management.
- `FR-TENANT-04`: Enforce tenant membership checks for API access to tenant-scoped resources.
- `FR-TENANT-05`: Support separate admin portal experiences for global admins and tenant admins.

## 7. Non-Functional Requirements

- `NFR-DEPLOY-01`: Must run in Docker.
- `NFR-DEPLOY-02`: Must support GPU acceleration for analysis workers.
- `NFR-DEPLOY-03`: Must support CPU-only fallback mode.
- `NFR-PLAT-01`: Deployment options include local PC host, NVIDIA-capable edge host, and cloud infrastructure.
- `NFR-OPS-01`: Provide logs, metrics, and traceable job stages.
- `NFR-SEC-01`: Multi-user access control and project-level isolation.
- `NFR-SEC-01A`: Tenant-level data isolation must prevent cross-tenant read/write access.
- `NFR-SEC-02`: Support MFA and enterprise SSO policy enforcement.
- `NFR-SEC-03`: Provide tamper-evident audit logging for sensitive actions.
- `NFR-DATA-01`: Configurable retention and deletion policy for video assets.
- `NFR-DATA-02`: Support data residency controls for regional deployments.
- `NFR-COMP-01`: Support youth privacy and consent management requirements.
- `NFR-API-01`: Provide versioned API contracts and backward-compatibility policy.
- `NFR-COST-01`: Provide quota and budget guardrails by organization/project.
- `NFR-AGENT-01`: Maintain prompt/version traceability for agent outputs used in operational workflows.
- `NFR-AGENT-02`: Enforce guardrails to prevent the agent from mutating authoritative event labels without explicit reviewer approval.
- `NFR-AGENT-03`: Support provider abstraction to allow LLM API vendor switching without major product rewrites.

## 8. Acceptance Criteria (MVP)

1. A user can upload a full match and receive asynchronous job completion without blocking the UI.
2. Completed jobs produce follow-cam output, event timeline, and highlight clips.
3. Event timeline includes timestamp, type, and confidence fields.
4. Users can edit highlights in a timeline UI and export clips/montage.
5. Docker deployment can run CPU workers and GPU workers (`--gpus all`) from the same codebase.
6. Tenant A users cannot read or mutate Tenant B resources through API or portals.
7. Global admins and tenant admins have separate administration surfaces with role-appropriate access.

## 9. Suggested Architecture

1. Web frontend
2. API backend
3. Queue/scheduler
4. GPU analysis workers
5. Rendering workers
6. Object storage
7. Relational database
8. Monitoring/alerting stack

## 10. Success Metrics

1. Time to first playable highlights per match
2. Event precision/recall on labeled validation set
3. Share/export completion rate
4. User edit rate on auto-generated clips
5. Compute cost per processed match

## 11. Risks and Mitigations

1. Camera angle and quality variance
   - Mitigation: calibration profiles, confidence gating, model tiers
2. Long-running job failures
   - Mitigation: checkpointing, retries, resumable stages
3. Player identity ambiguity
   - Mitigation: assisted jersey pipeline and manual assignment
4. Compute cost pressure
   - Mitigation: autoscaling, queue priority classes, fast mode profiles
