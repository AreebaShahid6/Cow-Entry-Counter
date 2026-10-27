"""
Automated Verification Suite for Smart Dairy (OpenCV AI Competition 2026, powered by AWS).
Tests:
  1. OpenCV 5 + COOL Engine & Reproducible Graviton Benchmark
  2. Agentic Vision Closed-Loop Perception -> Decision -> Action & HITL Approval
  3. AWS Cloud Integration Layer (DynamoDB, S3, CloudWatch, Bedrock mirror)
  4. Working Web Endpoint REST API & Static Media Serving
"""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from agentic_vision import AgenticVisionController
from aws_integration import AWSCloudManager
from opencv5_cool_engine import OpenCV5Engine, run_cool_benchmark
from web_app import BACKEND, SmartDairyHTTPHandler


class TestHackathonRequirements(unittest.TestCase):

    def test_01_opencv5_and_cool_benchmark(self) -> None:
        engine = OpenCV5Engine(width=1280, height=720, enable_cool_opt=True)
        self.assertIn("cool_profile", engine.runtime_info)
        bx, by = engine.project_to_bev([640.0, 360.0])
        self.assertIsInstance(bx, float)
        self.assertIsInstance(by, float)

        bench = run_cool_benchmark(iterations=5)
        self.assertIn("summary", bench)
        self.assertGreater(bench["summary"]["graviton_cool_speedup_x"], 1.2)
        self.assertEqual(len(bench["operations"]), 4)

    def test_02_agentic_vision_closed_loop_and_hitl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            aws = AWSCloudManager(output_dir=tmpdir)
            agent = AgenticVisionController(aws_manager=aws, output_dir=tmpdir)
            agent.ensure_demo_trace_populated()
            self.assertGreaterEqual(len(agent.decision_trace), 3)

            # Trigger gait anomaly and verify HITL approval queue
            decs = agent.step(
                frame_idx=300,
                video_time_sec=12.5,
                entered_count=8,
                in_view_count=12,
                outside_count=9,
                avg_conf=0.75,
                lighting_metrics={"doorway_luminance": 70.0, "yard_luminance": 150.0, "contrast_ratio": 2.14},
                flow_metrics={"congestion_index": 0.15, "doorway_flow_speed": 1.2},
                welfare_metrics={
                    "cow_locomotion": {
                        99: {"cow_number": 99, "gait_score": 49.5, "speed_mps": 0.25, "welfare_flag": "IRREGULAR_GAIT"}
                    }
                },
            )
            self.assertGreaterEqual(len(decs), 1)
            hitl_decs = [d for d in decs if d["loop_stage"] == "HITL_WELFARE_INTERVENTION"]
            self.assertEqual(len(hitl_decs), 1)
            pending = [p for p in agent.pending_approvals if p["cow_id"] == 99]
            self.assertTrue(len(pending) >= 1)

            # Approve via HITL gate
            res = agent.resolve_human_approval(pending[0]["approval_id"], approved=True)
            self.assertTrue(res["ok"])
            self.assertIn("COW #99", agent.barn_actuators["sorting_pen_diverter"])

    def test_03_aws_cloud_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            aws = AWSCloudManager(output_dir=tmpdir)
            rec = aws.put_cow_entry_event({"no": 1, "cow": 15, "track_id": 15, "frame": 34, "time": 1.42, "how": "line"})
            self.assertEqual(rec["CowDisplayId"], 15)
            cw = aws.publish_cloudwatch_metrics(8, 14, 10, 0.28, 45.0)
            self.assertEqual(cw["Metrics"]["CowsEnteredTotal"], 8)
            summary = aws.status_summary()
            self.assertGreaterEqual(summary["counts"]["dynamodb_events_synced"], 1)

    def test_04_working_web_endpoint_api(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), SmartDairyHTTPHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            # 1. Health endpoint
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
                self.assertEqual(resp.status, 200)
                health = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health["status"], "healthy")

            # 2. Overview endpoint
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/overview") as resp:
                self.assertEqual(resp.status, 200)
                ov = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(ov["kpis"]["total_cows_entered"], 8)
                self.assertEqual(ov["kpis"]["unique_tracks_analyzed"], 24)

            # 3. HTML Dashboard
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("SMART DAIRY", html)
                self.assertIn("Agentic Vision Control Center", html)

            # 4. H.264 Video Byte-Range Streaming
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/video/your_video.mp4",
                headers={"Range": "bytes=0-1023"},
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 206)
                chunk = resp.read()
                self.assertEqual(len(chunk), 1024)
                self.assertIn(b"ftyp", chunk[:32])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
