# System Architecture Diagram — Smart Dairy (OpenCV 5 + AWS + Agentic Vision + COOL)

**Competition:** OpenCV AI Competition 2026, powered by AWS  
**Eligible Tracks:** Overall Track • Agentic Vision Award Track • Best Use of COOL Award Track

---

## 1. End-to-End Hybrid Edge-to-Cloud Architecture

```mermaid
flowchart TD
    subgraph BarnEdge["Physical Dairy Barn Edge / Camera Stream"]
        CAM["CAM-01: Barn Entrance Camera<br/>(1280x720 @ 24-30 FPS)"]
        ACT_GATE["Pneumatic Entry Gate Actuator<br/>(Standard 80% / Relief 100%)"]
        ACT_SORT["Veterinary Sorting Pen Diverter<br/>(Main Herd / Pen B)"]
        ACT_LIGHT["Doorway Guidance LED Array"]
    end

    subgraph GravitonCOOL["AWS EC2 Graviton4 (c8g.xlarge ARM64) + COOL Engine"]
        COOL_OPT["Cloud-Optimized OpenCV Library (COOL)<br/>ARM64 NEON / SVE2 / KleidiCV Acceleration"]
        CV5_PRE["OpenCV 5 LAB Color-Space & CLAHE<br/>Doorway Shadow Recovery"]
        CV5_FLOW["OpenCV 5 Dense Farneback Optical Flow<br/>Crowd Velocity & Congestion Index"]
        CV5_BEV["OpenCV 5 Perspective Homography (BEV)<br/>Top-Down Metric Gait & Stride Scoring"]
        CV5_GMC["OpenCV 5 Sparse Optical Flow GMC<br/>Camera Motion Compensation"]
        YOLO_BOT["YOLO26l Detector + BoT-SORT Tracker<br/>+ Trajectory Re-Linking & Doorway Occlusion Logic"]
        HEATMAP["OpenCV 5 Spatial Footprint Accumulator<br/>Gaussian Blur + TURBO Colormap Overlay"]
    end

    subgraph AgenticLoop["Agentic Vision Controller (Perception -> Decision -> Action)"]
        PERC["Structured Visual Perception State<br/>(Luminance, Congestion, Track Conf, BEV Gait)"]
        BEDROCK["Amazon Bedrock Reasoning Agent<br/>(Claude 3.5 Sonnet / Edge Policy Fallback)"]
        T1["Tool 1: adjust_vision_pipeline()<br/>Toggle CLAHE & BoT-SORT Buffer"]
        T2["Tool 2: actuate_barn_entry_gate()<br/>Widen Gate & Boost Chute Lighting"]
        T3["Tool 3: isolate_cow_for_vet_inspection()<br/>Queue HITL Veterinary Sort Request"]
        HITL["Human-in-the-Loop (HITL) Approval Gate<br/>Farm Operator Web Confirmation"]
    end

    subgraph AWSCloud["Amazon Web Services (AWS) Persistence & Observability"]
        DDB["Amazon DynamoDB<br/>(SmartDairyCowEvents Table)"]
        S3["Amazon S3 Bucket<br/>(Dashboard MP4, Heatmap PNG, CSV Logs)"]
        CW["Amazon CloudWatch<br/>(SmartDairy/OpenCV5 Custom Metrics)"]
        WEB["Working Web Endpoint (FastAPI / HTTP)<br/>Live Streaming, Agent Control & COOL Benchmark"]
    end

    CAM --> COOL_OPT
    COOL_OPT --> CV5_PRE
    CV5_PRE --> CV5_FLOW
    CV5_PRE --> YOLO_BOT
    CV5_GMC --> YOLO_BOT
    YOLO_BOT --> CV5_BEV
    YOLO_BOT --> HEATMAP

    CV5_PRE --> PERC
    CV5_FLOW --> PERC
    CV5_BEV --> PERC
    YOLO_BOT --> PERC

    PERC --> BEDROCK
    BEDROCK --> T1
    BEDROCK --> T2
    BEDROCK --> T3

    T1 -->|Self-Tune OpenCV 5 Params| CV5_PRE
    T1 -->|Extend Track Buffer| YOLO_BOT
    T2 -->|Physical Control| ACT_GATE
    T2 -->|Physical Control| ACT_LIGHT
    T3 --> HITL
    HITL -->|Operator Approved| ACT_SORT

    YOLO_BOT -->|Per-Cow Crossing Events| DDB
    HEATMAP -->|MP4 + PNG + CSV Artifacts| S3
    CV5_FLOW -->|Throughput & Congestion| CW
    DDB --> WEB
    S3 --> WEB
    CW --> WEB
    HITL <-->|Approve / Reject UI| WEB
```

---

## 2. Agentic Vision Closed-Loop Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Barn Camera (Frame t)
    participant CV5 as OpenCV 5 + COOL Engine
    participant Track as YOLO26 + BoT-SORT
    participant Agent as Agentic Vision Controller (AWS Bedrock)
    participant Web as Operator Web Control Center
    participant Barn as Barn IoT Actuators & AWS Cloud

    Cam->>CV5: Raw BGR Frame (1280x720)
    CV5->>CV5: Compute LAB Luminance (Doorway vs Yard) & Dense Optical Flow
    CV5->>Track: Enhanced Frame + Sparse Optical Flow GMC
    Track->>CV5: Active Cow Tracks + Line/Doorway Crossings
    CV5->>CV5: Project Footprints via Homography (BEV) -> Gait Smoothness Score
    CV5->>Agent: Visual Telemetry (door_lum=48.6, congestion=0.38, cow9_gait=54.2)

    alt High Doorway Shadow Contrast Detected
        Agent->>CV5: Tool: adjust_vision_pipeline(clahe=True, buffer=75)
        CV5-->>Agent: CLAHE Shadow Recovery Activated
    end

    alt Crowd Bottleneck at Barn Entrance (congestion > 0.24)
        Agent->>Barn: Tool: actuate_barn_entry_gate("WIDE_RELIEF_OPEN 100%")
        Agent->>Barn: Publish Congestion Metric to Amazon CloudWatch
    end

    alt Irregular Gait / Lameness Detected in BEV Space
        Agent->>Web: Tool: isolate_cow_for_vet_inspection(COW #9) [HITL Queued]
        Web->>Agent: Herd Manager Clicks "Approve Sort"
        Agent->>Barn: Actuate Sorting Pen Diverter -> DIVERT_VET_PEN_B
        Agent->>Barn: Persist Event to Amazon DynamoDB (SmartDairyCowEvents)
    end
```

