# Architecture

```mermaid
flowchart TD
    U[User / Judge] -->|upload image| API[FastAPI :8000\nPOST /inspect]
    U -->|browser| UI[Gradio :7860\nannotated + trace viewer]
    UI --> API
    API --> AG[Agent loop\nzoomin/agent.py]
    AG -->|scan_overview| SC[CoarseScorer\nclassical OpenCV\n+ ONNX hook via OpenCV 5 DNN]
    AG -->|zoom_crop| ZC[zoom_crop tool]
    AG -->|inspect_crop| IC[inspect_crop tool\nOtsu residual + morphology\n+ optional PatchCore]
    AG -->|ambiguous| RP{re-plan?\nre-zoom tighter}
    RP -->|still unsure| ESC[escalate to human]
    AG -->|finalize| RPT[report.py\nJSON + annotated PNG]
    AG --> S3[(S3\nimages + artifacts)]
    AG -.->|trace| TR[(trace log\nAgentic Vision evidence)]

    subgraph AWS [AWS deployment]
        EC2[EC2 c7g.xlarge\nCOOL AMI\nDocker Compose]
        EC2 --> API
        EC2 --> UI
        EC2 --> S3
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
| Agent loop + tools | EC2 container | DNN engine (ONNX hook), morphology, contours, connected components | EC2 Graviton (COOL AMI) |
| CoarseScorer | EC2 container | medianBlur, boxFilter, GaussianBlur | — |
| inspect_crop | EC2 container | Otsu threshold, morphologyEx, findContours | — |
| FastAPI / Gradio | EC2 container | imdecode/imencode | served from EC2, images in S3 |
| Benchmark | EC2 (3 configs) | whole pipeline | Graviton vs x86, $/10K images |
| Dataset | — | — | S3 bucket |

## Data flow

1. Image lands via API/UI (or S3 for batch benchmark).
2. `scan_overview` produces a heatmap; boxes are proposed from it.
3. The agent spends its crop budget: `zoom_crop` → `inspect_crop` → decide.
   Ambiguous regions trigger a tighter re-zoom (re-plan) or escalation.
4. `finalize_report` emits verdicts, the full tool trace, and an annotated image.
5. The trace is the Agentic Vision deliverable: it proves vision outputs changed
   later tool calls (heatmap → crop coordinates; score → re-zoom vs escalate).
