# Smart Dairy: Autonomous Barn Vision, Agentic Control & COOL Analytics on AWS

[![OpenCV AI Competition 2026](https://img.shields.io/badge/OpenCV_AI_Competition_2026-Powered_by_AWS-00E5A0?style=for-the-badge)](https://opencv26.devpost.com/)
[![OpenCV 5 + COOL](https://img.shields.io/badge/OpenCV_5-COOL_Graviton4_ARM64-4DA8FF?style=for-the-badge)](./opencv5_cool_engine.py)
[![Agentic Vision](https://img.shields.io/badge/Award_Track-Agentic_Vision_%26_COOL-FFB547?style=for-the-badge)](./agentic_vision.py)

**Smart Dairy** is an end-to-end **Physical AI & Agentic Computer Vision system** built for the **OpenCV AI Competition 2026, powered by AWS**. It combines **OpenCV 5**, the **Cloud-Optimized OpenCV Library (COOL)** on **AWS Graviton**, **YOLO26 + BoT-SORT** multi-object tracking, and an **Amazon Bedrock Agentic Vision Loop** to count cattle entering a barn, recover occluded doorway crossings, monitor locomotion welfare, and autonomously regulate barn entrance bottlenecks.

---

## Hackathon Submission Checklist & Deliverables

| Competition Requirement | Status | File / Endpoint Reference |
| :--- | :---: | :--- |
| **Substantive OpenCV 5 Analysis** | Complete | [`opencv5_cool_engine.py`](./opencv5_cool_engine.py) & [`main.py`](./main.py) (LAB CLAHE, Farneback Dense Optical Flow, Sparse Optical Flow GMC, Planar Homography BEV, Spatial Footprints) |
| **Meaningful AWS Cloud Component** | Complete | [`aws_integration.py`](./aws_integration.py) & [`aws_cloudformation.yaml`](./aws_cloudformation.yaml) (Amazon DynamoDB, S3, CloudWatch, Bedrock + zero-config local mirror) |
| **Agentic Vision Award Track** | Complete | [`agentic_vision.py`](./agentic_vision.py) (Multi-step Perception $\rightarrow$ Decision $\rightarrow$ Action loop + Human-in-the-Loop approval gate) |
| **Best Use of COOL Award Track** | Complete | [`opencv5_cool_engine.py`](./opencv5_cool_engine.py) (`run_cool_benchmark`: **2.27x speedup** & **42.8% cost savings** on AWS Graviton4) |
| **Working Web Endpoint** | Complete | [`web_app.py`](./web_app.py) (`http://localhost:8000` interactive dashboard, video/heatmap viewer, agent controls, REST API) |
| **Technical Report** | Complete | [`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md) |
| **Architecture Diagram** | Complete | [`ARCHITECTURE.md`](./ARCHITECTURE.md) |
| **Pinned Dependencies & Build Docs** | Complete | [`requirements.txt`](./requirements.txt), [`Dockerfile`](./Dockerfile), and [`tests/test_hackathon_suite.py`](./tests/test_hackathon_suite.py) |

---

## Quick Start: Build, Run & Test Instructions

### 1. Launch the Working Web Endpoint Immediately (Zero-Config)
The web application serves the processed 1080p dashboard video, final movement heatmap, live Agentic Vision Control Center, Human-in-the-Loop (HITL) approval queue, AWS DynamoDB/S3 telemetry, and interactive COOL benchmark runner:

```bash
python3 web_app.py --host 0.0.0.0 --port 8000
```
- Open **`http://localhost:8000`** in your browser.
- **REST API Endpoints:**
  - `GET /api/health` — OpenCV 5 + COOL runtime & AWS connection health check
  - `GET /api/overview` — Complete JSON payload (KPIs, cow entries, Agentic Vision trace, AWS ledger, COOL benchmark)
  - `POST /api/cool/benchmark` — Execute a live OpenCV 5 + COOL benchmark run
  - `POST /api/agent/trigger` — Trigger closed-loop Agentic Vision scenarios (`doorway_shadow`, `crowd_bottleneck`, `gait_anomaly`)
  - `POST /api/agent/approve` — Approve or reject Human-in-the-Loop (HITL) veterinary sorting actions

### 2. Install Pinned Dependencies (Full CV + AWS Environment)
```bash
pip install -r requirements.txt
```

### 3. Run the Full Video Analytics Pipeline (`main.py`)
```bash
# Process video/your_video.mp4 with OpenCV 5 + YOLO26l + BoT-SORT + Agentic Loop + AWS Sync
python3 main.py --no-show

# Optional: Interactively pick a new barn entry line on the first video frame
python3 main.py --pick-line

# Run the standalone OpenCV 5 + COOL (AWS Graviton) benchmark suite
python3 main.py --benchmark-cool
```

### 4. Run the Automated Verification Test Suite
```bash
python3 -m unittest discover -s tests -v
```

---

## Deploying on AWS (EC2 + Docker + CloudFormation)

### Option A: One-command cloud deploy (recommended)
Prerequisites: [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with credentials (`aws configure` or `aws sso login`) that can create CloudFormation, EC2, IAM, S3 and DynamoDB resources.

```bash
./deploy.sh us-east-1
```
This script:
1. Deploys [`aws_cloudformation.yaml`](./aws_cloudformation.yaml): S3 bucket, DynamoDB table, IAM role/instance profile (S3, DynamoDB, CloudWatch, Bedrock, SSM), security group (port 80) and an EC2 instance.
2. Bundles the project and uploads it to S3.
3. On first boot the instance installs Docker, downloads the bundle, builds the image and serves the app on port 80.
4. Prints the website URL (`http://ec2-...compute-1.amazonaws.com`). Allow ~5-10 minutes for the build, then check `/api/health` (it should report `aws_mode: LIVE_AWS_BOTO3`).

Notes:
- Default instance is `m7i-flex.large` (x86_64, Free Tier eligible). Graviton types (`c8g.*`, `c7g.xlarge`, `t4g.large`) are listed in the template, but they need the arm64 AMI: change `LatestAmiId` to `...al2023-ami-kernel-default-arm64`.
- Restrict access with `--parameter-overrides AllowedCidr=<your-ip>/32`. The site is plain HTTP.
- Troubleshooting: read `/var/log/smart-dairy-init.log` on the instance.
- Push new demo media to the running server: `./upload_assets.sh` then `./sync_assets.sh`.
- Tear down (empty the S3 bucket first): `aws cloudformation delete-stack --stack-name smart-dairy`.

### Option B: Run the container yourself
```bash
docker build -t smart-dairy-opencv26 .
docker run -p 8000:8000 -e AWS_REGION=us-east-1 smart-dairy-opencv26
```
On EC2 with an instance role, `aws_integration.py` detects the role credentials automatically. Elsewhere, export `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (or `AWS_PROFILE`) and optionally:
```bash
export SMART_DAIRY_S3_BUCKET="smart-dairy-opencv26-<account-id>-us-east-1"
export SMART_DAIRY_DDB_TABLE="SmartDairyCowEvents"
export SMART_DAIRY_CW_NAMESPACE="SmartDairy/OpenCV5"
```
Without credentials the app falls back to `AWS_HYBRID_LOCAL_MIRROR` and records everything locally.

---

## Repository Structure

```text
cow_counter/
├── main.py                      # Core OpenCV 5 + YOLO26 + BoT-SORT + Agentic + AWS video pipeline
├── opencv5_cool_engine.py       # OpenCV 5 CLAHE, Dense Optical Flow, Homography BEV & COOL Benchmark
├── agentic_vision.py            # Closed-loop Perception -> Decision -> Action controller + HITL queue
├── aws_integration.py           # Amazon S3, DynamoDB, CloudWatch & Bedrock integration (+ local mirror)
├── web_app.py                   # Working Web Endpoint (HTTP + FastAPI compatible) & Control Center UI
├── botsort_cows.yaml            # BoT-SORT tracker config with OpenCV sparseOptFlow camera motion comp
├── requirements.txt             # Pinned dependencies for reproducible builds
├── Dockerfile                   # Container definition (builds on arm64 or x86_64 hosts)
├── aws_cloudformation.yaml      # AWS IaC template (S3, DynamoDB, IAM, EC2, security group)
├── deploy.sh                    # One-command AWS deploy (stack + code bundle upload)
├── upload_assets.sh             # Upload demo video/heatmap/CSVs to S3
├── sync_assets.sh               # Copy S3 media into the running container via SSM
├── reboot_instance.sh           # Reboot the EC2 instance (e.g. to register with SSM)
├── TECHNICAL_REPORT.md          # Comprehensive competition Technical Report
├── ARCHITECTURE.md              # System & Agentic Vision Mermaid Architecture Diagrams
├── tests/
│   └── test_hackathon_suite.py  # Automated unit & integration tests
├── video/
│   └── your_video.mp4           # Source barn entrance video (240 frames)
└── output/
    ├── cow_counter_dashboard.mp4  # Rendered 1920x1080 analytics dashboard video
    ├── heatmap_final.png          # Final OpenCV 5 spatial footprint heatmap
    ├── entries.csv                # Verified cow entry crossing log (8 cows entered)
    ├── cow_records.csv            # Per-cow tracking lifecycle & status log (24 unique tracks)
    ├── cool_benchmark_report.json # Reproducible COOL Graviton benchmark report
    ├── agentic_vision_log.json    # Multi-step Agentic Vision reasoning & tool call trace
    └── aws_cloud_telemetry.json   # Mirrored DynamoDB items, S3 artifacts & CloudWatch metrics
```

