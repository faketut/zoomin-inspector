# AWS Deployment Runbook — ZoomIn Inspector

Target: OpenCV AI Competition 2026 — substantive OpenCV 5 workload + meaningful
AWS component, with reproducible COOL-vs-vanilla benchmark evidence.

**Region:** `us-east-1` (safest for Marketplace AMI availability; `ca-central-1`
works if the COOL AMI is listed there).

## 0. Prerequisites (your AWS account)

1. AWS account (new accounts may qualify for $100 free-tier credits — see the
   competition page). Do all of this in `us-east-1`.
2. A key pair: EC2 → Key Pairs → Create key pair (`zoomin-key`, `.pem`) — keep the file safe.
3. Push this repo to GitHub (private is fine — the competition allows private repos):
   ```bash
   gh repo create faketut/zoomin-inspector --private --source=. --push
   ```

## 1. Subscribe to the COOL AMI

1. Open the listing: https://aws.amazon.com/marketplace/pp/prodview-jz6w32bsfv6sg
2. **Continue to Subscribe → Accept Terms.** Note any hourly software charge shown
   (on top of EC2) — factor it into the benchmark's $/10K-images math.
3. **Continue to Configuration → Continue to Launch** (or find the AMI later under
   EC2 → AMIs → Owned by Amazon/marketplace search "COOL").

## 2. S3 bucket (data + artifacts)

```bash
aws s3 mb s3://zoomin-inspector-<yourname> --region us-east-1
```

## 3. Security group

Create `zoomin-sg` with inbound rules:

| Port | Source | Purpose |
|---|---|---|
| 22 | your IP | SSH |
| 8000 | 0.0.0.0/0 (demo window only) | FastAPI for judges |
| 7860 | 0.0.0.0/0 (demo window only) | Gradio UI |

Lock 8000/7860 back down after judging — the competition scores responsible operation.

## 4. Launch the Graviton instance (config A + B)

- **AMI:** the subscribed COOL AMI (Ubuntu 24.04, `/opt/cool`).
- **Type:** `c7g.xlarge` (~$0.145/hr on-demand us-east-1 — confirm at launch).
- **Storage:** 30 GB gp3. **Key pair:** `zoomin-key`. **Security group:** `zoomin-sg`.
- **User data:** paste `scripts/ec2-user-data.sh` (installs docker, compose plugin, awscli, git).

```bash
ssh -i zoomin-key.pem ubuntu@<instance-public-dns>
```

## 5. Verify the COOL runtime (do this first — it's the award evidence)

```bash
ls /opt/cool                          # discover the bundled venvs
for py in /opt/cool/*/bin/python3; do echo "== $py"; $py -c "import cv2; print(cv2.__version__)"; done
curl -s localhost:8000/health || true  # later, after the API is up
```

Record the exact `cv2.__version__` string — the AMI's build identifies itself here.
(Prior reports note the AMI paths may reference an `opencv4` dir name; what matters
is the runtime version string, so pin it in the report.)

## 6. Code + data on the instance

```bash
git clone https://github.com/faketut/zoomin-inspector.git && cd zoomin-inspector
python3 scripts/download_mvtec.py --categories bottle transistor capsule --out data/mvtec
```

## 7. Config A — COOL on Graviton (the money run)

⚠️ **Trap:** `requirements.txt` pins `opencv-python-headless`. Installing it into the
COOL venv would *overwrite* the COOL-optimized cv2 with the stock pip wheel. Exclude it:

```bash
COOLPY=$(ls /opt/cool/*/bin/python3 | head -1)
$COOLPY -m pip install --quiet $(grep -v '^opencv-python-headless' requirements.txt | tr '\n' ' ')
$COOLPY -c "import cv2; assert cv2.__version__.startswith('5'), cv2.__version__; print('COOL cv2:', cv2.__version__)"

$COOLPY bench/benchmark.py --images "data/mvtec/bottle/test" \
    --config cool-graviton --dollars-per-hour 0.145 --out bench/results.csv
```

Then serve the demo endpoint from the COOL runtime:

```bash
nohup $COOLPY -m uvicorn app.api:app --host 0.0.0.0 --port 8000 >/tmp/api.log 2>&1 &
curl -s localhost:8000/health
```

## 8. Config B — vanilla pip OpenCV on the same Graviton instance

```bash
python3 -m venv /opt/venvs/vanilla
/opt/venvs/vanilla/bin/pip install --quiet -r requirements.txt
/opt/venvs/vanilla/bin/python bench/benchmark.py --images "data/mvtec/bottle/test" \
    --config vanilla-graviton --dollars-per-hour 0.145 --out bench/results.csv
```

## 9. Config C — vanilla pip OpenCV on x86

Launch a **c7i.xlarge** (~$0.17/hr) with stock Ubuntu 24.04, same security group:

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv git
git clone https://github.com/faketut/zoomin-inspector.git && cd zoomin-inspector
python3 -m venv .venv && .venv/bin/pip install --quiet -r requirements.txt
python3 scripts/download_mvtec.py --categories bottle --out data/mvtec
.venv/bin/python bench/benchmark.py --images "data/mvtec/bottle/test" \
    --config vanilla-x86 --dollars-per-hour 0.17 --out bench/results.csv
```

**Terminate this instance when the CSV is saved** (it's the most expensive per hour).

## 10. Collect results

```bash
# on each instance:
aws s3 sync bench/ s3://zoomin-inspector-<yourname>/results/<config>/
```

Back home: download the three CSVs, paste the comparison table into
`docs/REPORT.md` §6.3, and fill in the eval numbers (§6.1, §6.2, §6.4).

## 11. Judges' endpoint

Keep the Graviton instance running while you record the demo video:
`http://<instance-public-dns>:8000` (API) and `:7860` if you also run the UI
(`$COOLPY -m app.ui --server-name 0.0.0.0`). The submission also accepts a live
screen-share demo instead of a public endpoint.

## 12. Cleanup (cost control)

Terminate both instances when done: EC2 → Instances → Terminate.
Expected total for a focused session: a few dollars (instances) + cents (S3).
Keep the AMI subscription (no charge while no instance runs) or unsubscribe from
the Marketplace listing if you prefer.

## Troubleshooting

- `cv2` import fails in the COOL venv → re-run step 5 discovery; never `pip install opencv-*` there.
- Gradio import error about httpx port `:1]` → `env -u NO_PROXY -u no_proxy` (sandbox proxy quirk, not AWS).
- Marketplace "not subscribed" at launch → redo step 1; subscriptions are per-account, not per-region, but AMIs are regional — launch in the region you subscribed from.
