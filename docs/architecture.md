# Architecture

```mermaid
flowchart TD
    U[User / Judge] -->|upload image| API[FastAPI :8000\nPOST /inspect]
    U -->|browser| UI[Gradio :7860\nannotated + trace viewer]
    API --> AG[Agent loop\nzoomin/agent.py]
    UI --> AG
    AG -->|scan_overview| SC[Scorer\nCoarseScorer or NormalityScorer\n+ ONNX hook via OpenCV 5 DNN]
    AG -->|zoom_crop| ZC[zoom_crop tool]
    AG -->|inspect_crop| IC[inspect_crop tool\npeak heat + coverage\nfrom aligned heatmap slice]
    AG -->|ambiguous| RP{re-plan?\nre-zoom tighter}
    RP -->|coverage confirms| DEF[DEFECT]
    RP -->|still unsure| ESC[escalate to human]
    AG -->|finalize| RPT[report.py\nJSON + annotated PNG]
    AG -.->|trace| TR[(trace log\nAgentic Vision evidence)]

    subgraph AWS [AWS deployment]
        EC2[EC2 c8g.xlarge\nCOOL AMI\nDocker Compose]
        EC2 --> API
        EC2 --> UI
        S3[(S3\nbenchmark CSVs + logs)]
        EC2 -->|aws s3 sync| S3
    end
    subgraph BENCH [Benchmark harness]
        B[bench/benchmark.py\nsame script × 3 configs]
        B -->|A| COOL[COOL on Graviton]
        B -->|B| VG[vanilla pip on Graviton]
        B -->|C| VX[vanilla pip on x86]
    end
```

## Components

| Component | Runs where | OpenCV 5 usage | AWS usage |
|---|---|---|---|
| Agent loop + tools | EC2 container | morphology, contours, connected components | EC2 Graviton (COOL AMI) |
| CoarseScorer | EC2 container | medianBlur, boxFilter, GaussianBlur | — |
| NormalityScorer | EC2 container | resize, abs-diff, GaussianBlur | — |
| inspect_crop | EC2 container | threshold, findContours | — |
| FastAPI / Gradio | EC2 container | imdecode/imencode | served from EC2 |
| Benchmark | EC2 (3 configs) | whole pipeline | Graviton vs x86, $/10K images |
| Artifacts | — | — | S3 bucket (`aws s3 sync` in the runbook, not in app code) |

The UI runs the agent in-process; it does not call the API over HTTP. Both
entry points share `load_default_scorer()` and `InspectorConfig.for_scorer()`,
so they cannot disagree about which scorer or thresholds are active.

## Data flow

1. Image lands via API upload or the UI (`POST /inspect` reads the upload as a
   plain `def`, so the CPU-bound loop runs in a worker thread rather than on
   the event loop).
2. `scan_overview` produces a heatmap; boxes are proposed from it.
3. The agent spends its crop budget: `zoom_crop` → `inspect_crop` → decide.
   Ambiguous regions trigger a tighter re-zoom (re-plan); the re-zoom's
   `coverage` decides between DEFECT and escalation.
4. `finalize_report` emits verdicts, the full tool trace, and an annotated image.
5. The trace is the Agentic Vision deliverable: it proves vision outputs changed
   later tool calls (heatmap → crop coordinates; coverage → confirm vs escalate).
