# Requirements Traceability

This file confirms coverage of requirements provided in the planning context.

## Context-Window Requirement Coverage

| Context requirement | Coverage in PRD | Roadmap phase | Status |
|---|---|---|---|
| Analyze full-game soccer recordings and generate highlights | `FR-INGEST-01`, `FR-INGEST-03`, `FR-EDITOR-05` | Phase 2, Phase 3 | Captured |
| Re-architecture from desktop/local to web/cloud | PRD Sections 2, 3, 9 | Phase 0, Phase 2 | Captured |
| Dockerized deployment | `NFR-DEPLOY-01` | Phase 1, Phase 2 | Captured |
| GPU acceleration on local host and cloud/edge | `NFR-DEPLOY-02`, `NFR-PLAT-01` | Phase 1, Phase 2, Phase 5 | Captured |
| Multi-client (multi-tenant) isolation with per-tenant users | `FR-TENANT-01`, `FR-TENANT-04`, `NFR-SEC-01A` | Phase 1, Phase 2 | Captured |
| Robust overall admin portal (platform/global admin) | `FR-TENANT-02`, `FR-TENANT-05` | Phase 1, Phase 2 | Captured |
| Separate tenant admin portal per tenant | `FR-TENANT-03`, `FR-TENANT-05` | Phase 1, Phase 2 | Captured |
| Automatic ball tracking | `FR-CAM-03` | Phase 2 | Captured |
| Broadcast-style AI follow-cam from static recording | `FR-CAM-02` | Phase 2 | Captured |
| Panoramic/wide input support (180-degree style) | `FR-CAM-01` | Phase 2 | Captured |
| Dual-lens/4K-class camera input support | `FR-CAM-04` | Phase 2 | Captured |
| Unattended fixed-camera workflow (no operator pan/zoom) | `FR-CAM-05` | Phase 2 | Captured |
| Event detection: goals | `FR-EVENT-01` | Phase 2 | Captured |
| Event detection: shots | `FR-EVENT-02` | Phase 2 | Captured |
| Event detection: corner kicks | `FR-EVENT-03` | Phase 2 | Captured |
| Event detection: penalty kicks | `FR-EVENT-04` | Phase 2 | Captured |
| Event detection: free kicks | `FR-EVENT-05` | Phase 2 | Captured |
| Event detection: goal kicks | `FR-EVENT-06` | Phase 2 | Captured |
| Event detection: kick-offs | `FR-EVENT-07` | Phase 2 | Captured |
| Event detection: fouls | `FR-EVENT-08` | Phase 4 | Captured (phase-gated) |
| Player spotlight clips | `FR-PLAYER-01` | Phase 3 | Captured |
| Player tracking by shirt number | `FR-PLAYER-02` | Phase 4 | Captured |
| Player movement and performance metrics | `FR-PLAYER-04` | Phase 3 | Captured |
| Match momentum graph | `FR-TEAM-01` | Phase 3 | Captured |
| Heatmaps and position analysis | `FR-TEAM-02`, `FR-TEAM-03` | Phase 3 | Captured |
| Interactive editor with panoramic/follow-cam switching | `FR-EDITOR-01` | Phase 3 | Captured |
| Zoom in editor | `FR-EDITOR-02` | Phase 3 | Captured |
| Drawing/annotation tools | `FR-EDITOR-04` | Phase 3 | Captured |
| Custom highlight creation and player tagging | `FR-EDITOR-05`, `FR-EDITOR-06` | Phase 3 | Captured |
| Highlight sharing/export | `FR-EDITOR-07` | Phase 3 | Captured |
| Live streaming support | `FR-LIVE-01` | Phase 4 | Captured (phase-gated) |
| Instant playback/replay bookmarks | `FR-LIVE-03` | Phase 4 | Captured (phase-gated) |
| AI agent API for natural-language analysis support | `FR-AGENT-01`, `FR-AGENT-04` | Phase 3, Phase 5 | Captured |
| Explainable event reasoning from AI assistant | `FR-AGENT-02` | Phase 4 | Captured |
| AI-assisted missed-event review suggestions | `FR-AGENT-03` | Phase 4 | Captured |
| Capture feedback on missed/incorrect events | `FR-LEARN-01` to `FR-LEARN-03` | Phase 3 | Captured |
| Use approved feedback to improve model training | `FR-LEARN-05`, `FR-LEARN-06` | Phase 5 | Captured |

## Notes

1. Fouls, jersey-number workflows, and live features are explicitly planned but phase-gated after MVP.
2. MVP focuses on stable asynchronous full-match processing, follow-cam, base event timeline, and editable exports.

## Market-Leadership Scope Coverage

| Added scope area | Coverage in PRD | Roadmap phase | Status |
|---|---|---|---|
| Data trust layer (calibration, confidence QA, human review) | `FR-QA-01` to `FR-QA-04` | Phase 4 | Captured |
| Season and career intelligence | `FR-LONG-01` to `FR-LONG-03` | Phase 4 | Captured |
| Opponent scouting automation | `FR-SCOUT-01` to `FR-SCOUT-03` | Phase 4 | Captured |
| Coach action engine (drills + plans) | `FR-COACH-01` to `FR-COACH-03` | Phase 6 | Captured |
| Recruiting workflows | `FR-RECRUIT-01` to `FR-RECRUIT-03` | Phase 6 | Captured |
| Distribution and monetization | `FR-DIST-01` to `FR-DIST-04` | Phase 6 | Captured |
| Open platform and integrations | `FR-INT-01` to `FR-INT-04` | Phase 5 | Captured |
| Camera fleet and edge operations | `FR-FLEET-01` to `FR-FLEET-04` | Phase 5 | Captured |
| Collaboration and governance | `FR-COLLAB-01` to `FR-COLLAB-04` | Phase 6 | Captured |
| Enterprise-grade security and compliance | `NFR-SEC-02`, `NFR-SEC-03`, `NFR-COMP-01`, `NFR-DATA-02` | Phase 5 | Captured |
| LLM governance and provider flexibility | `NFR-AGENT-01` to `NFR-AGENT-03` | Phase 5 | Captured |
