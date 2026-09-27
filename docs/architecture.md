# Architecture

## Purpose

Log vehicle entry and exit at Gate 1 of Global College of Management (Baneshwor,
Kathmandu) from two UNV CCTV cameras. The next step is an agentic AI layer on top that
can answer questions about the traffic, notice when something is off, and keep itself
running with no one watching. That layer is designed but not built yet.

Constraints that shape everything: **CPU only** (no GPU at GCM), and it must run for long
stretches unsupervised on a server.

## Two layers

```
                         ┌────────────── agentic AI layer (Claude, planned) ───────────────┐
                         │  analyst agent      monitor agent        watchdog agent          │
                         │  (NL questions)     (anomaly detection)  (self-healing)          │
                         │        └──────────────── shared tools ──────────┘                │
                         └───────────────────────────┬────────────────────────────────────┘
                                                      │ reads/acts via tools
  camera ─▶ capture ─▶ perception ─▶ storage ─────────┘
           (RTSP,      (YOLO detect,   (crossings +
            reconnect)  ByteTrack,      visits in
                        LineZone)       SQLite)
```

### Perception pipeline (the boring, reliable part)

1. **capture** — RTSP frames per camera, with automatic reconnect. Every Nth frame only
   (`frame_stride`) to stay light on CPU.
2. **perception** — `ultralytics` YOLO detects vehicles; `supervision` ByteTrack gives
   each a stable ID; `supervision` LineZone turns a gate-line crossing into an
   entry or exit. This replaces v1's two hand-rolled trackers and its buggy, duplicated
   crossing logic.
3. **storage** — each crossing is recorded and paired into a **visit** (an entry and its
   later exit) in SQLite, with real timestamps and indices.

### Agentic AI layer (planned, not built yet)

The capture, perception, and storage layers and the HTTP API run today. The agents below
are only designed so far. `agents/` holds their interfaces, every one of them raises
`NotImplementedError`, and the pipeline does not start them. `POST /ask` returns 503
until the analyst exists. Until the watchdog exists, what keeps GateWatch running
unattended is systemd's `Restart=always` plus the capture layer reconnecting on its own
when a camera drops.

The plan is to build it on the Anthropic Claude API with the SDK's tool runner. The agent loop runs on our
own server — no GPU and no external sandbox — which fits an unsupervised on-prem box.

**Split of responsibilities: Claude is the brain, Google Cloud Vision is the eyes.**
Claude decides which tools to call and reasons over structured results and the log; every
image-understanding call (OCR + frame analysis) goes through Google Cloud Vision, which
returns labels, objects, and detected text (including Nepali/Devanagari script). No
vision-capable LLM is used for the pixels.

- **Analyst agent** (Opus 4.8) will answer natural-language questions over the log,
  exposed via `POST /ask` and optionally over MCP.
- **Monitor agent** (Haiku 4.5, escalating to Opus 4.8) will review recent activity on an
  interval, along with a Google Cloud Vision analysis of the current frame (the
  `inspect_frame` tool), and raise an alert only when a human should look. That means an
  exit with no matching entry, an after-hours entry, loitering, or a spike.
- **Watchdog agent** (Haiku 4.5) will supervise the pipeline's own health and self-heal
  or alert, using the signals `/healthz` already reports.

All three will draw from one shared tool surface (`agents/tools.py`) so behaviour is
consistent and testable.

## Why not just v1 plus fixes

v1 was a proof of concept: two scripts, hardcoded credentials, a display-bound Dash UI,
an English-only OCR path that never actually read a Nepali plate, and no supervision of
its own health. v2 keeps v1's good instincts (YOLO detection, line-crossing direction,
SQLite logging) and rebuilds around modern, tested components, with room for the agentic layer.

## On plates

Deliberately de-emphasized. `docs/feasibility.md` measured the current cameras and showed
they cannot resolve Devanagari plate characters — a camera-placement problem, not a model
problem. `docs/camera_placement.md` and `docs/resolution_budget.py` cover what a third,
plate-dedicated camera would need. When plates are enabled, OCR runs through **Google
Cloud Vision** (`plate/ocr.py`, strong Devanagari support), feeding the plate
parser/validator in `plate/nepali_plate.py`. The `plate/` package holds all of this so
it's ready if a suitable camera is added, but the core system does not depend on it.
