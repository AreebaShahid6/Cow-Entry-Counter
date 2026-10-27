"""
=====================================================================
  SMART DAIRY  |  AI Cow Entry Counter & Agentic Vision System
  OpenCV AI Competition 2026, powered by AWS
  OpenCV 5 + COOL (Graviton) + YOLO26 + BoT-SORT + AWS Bedrock Agent
---------------------------------------------------------------------
  Folder layout
      main.py
      opencv5_cool_engine.py      <- OpenCV 5 & COOL benchmark engine
      agentic_vision.py           <- Perception-Decision-Action loop
      aws_integration.py          <- AWS S3, DynamoDB, CloudWatch, Bedrock
      web_app.py                  <- Interactive Web Endpoint & API
      requirements.txt
      video/your_video.mp4        <- put your video here
      output/                     <- created automatically

  Run
      pip install -r requirements.txt
      python main.py                  # process the video + AWS & Agentic loop
      python main.py --pick-line      # optional: click a new entry line
      python main.py --no-show        # faster, no preview window
      python main.py --benchmark-cool # run COOL Graviton performance benchmark
      python web_app.py               # launch working web endpoint on port 8000
====================================================================
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agentic_vision import AgenticVisionController
from aws_integration import AWSCloudManager
from opencv5_cool_engine import OpenCV5Engine, run_cool_benchmark

# ------------------------------------------------------------------ #
#  CONFIG
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "video")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LINE_FILE = os.path.join(BASE_DIR, "line_config.json")
TRACKER_FILE = os.path.join(BASE_DIR, "botsort_cows.yaml")
VIDEO_NAME = "your_video"

MODEL_NAME = "yolo26l.pt"   # yolo26n / s / m / l / x  (x = most accurate, slowest)
IMGSZ = 1280                # larger input = small / far cows are not missed
DET_CONF = 0.10             # low: weak/occluded boxes still help keep existing tracks alive
IOU = 0.7                   # keeps overlapping cows in a crowd as separate boxes
# COCO class ids: 19 = cow. 17 horse and 18 sheep are included because YOLO
# sometimes mislabels a cow for a few frames. Everything is shown as "COW".
DETECT_CLASSES = [19, 17, 18]

# Entry line in normalized coords (0..1 of frame width/height).
# Taken from your screenshot: the red line across the barn door.
# "inside" = any point inside the barn (decides the entry direction).
DEFAULT_LINE = {"a": [0.131, 0.564], "b": [0.222, 0.466], "inside": [0.165, 0.370]}

MIN_HITS = 5             # frames before a track gets a COW number
BOX_COLOR = "#38D9F5"    # one colour for every cow box
OUT_MIN = 4              # frames a cow must be seen OUTSIDE before its crossing can count
LINK_SEC = 1.5           # re-link a new tracker ID to a cow lost within this time ...
LINK_DIST = 0.45         # ... if it appears within 0.45 x cow size of where that cow was heading
JUMP = 0.6               # an ID moving > 0.6 x cow size in one frame = ID switched to another cow
TRAIL_LEN = 30
OUT_W, OUT_H = 1920, 1080
END_HOLD_SEC = 2.5       # final summary frame at the end of the video

TRACKER_YAML = """tracker_type: botsort
track_high_thresh: 0.30
track_low_thresh: 0.10
new_track_thresh: 0.40
track_buffer: 60
match_thresh: 0.80
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: False
model: auto
"""

# ------------------------------------------------------------------ #
#  COLORS / FONTS / DRAW HELPERS
# ------------------------------------------------------------------ #
C = {
    "bg": "#0A0E17", "panel": "#111827", "panel2": "#162036", "border": "#243049",
    "text": "#EAF0FA", "muted": "#8B98B5", "dark": "#0A0E17",
    "accent": "#00E5A0", "amber": "#FFB547", "pink": "#FF5C8A",
    "blue": "#4DA8FF", "purple": "#A07CFF", "red": "#FF4D4D",
}
ID_COLORS = ["#4DA8FF", "#FFB547", "#A07CFF", "#FF5C8A", "#3DD6F5",
             "#FF8A4D", "#C6F24D", "#F27BE5", "#7C9CFF", "#FFD84D"]
CMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
AA = cv2.LINE_AA


def bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


FONT_FILES = {
    True: ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
    False: ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"],
}
_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        f = None
        for path in FONT_FILES[bold]:
            try:
                f = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        if f is None:
            try:
                f = ImageFont.load_default(size=size)
            except Exception:
                f = ImageFont.load_default()
        _fonts[key] = f
    return _fonts[key]


class TextLayer:
    """Collects text, then draws it all in one PIL pass (smooth, nice fonts)."""

    def __init__(self):
        self.items = []

    def add(self, text, xy, size, color, bold=False, anchor="la"):
        col = rgb(color) if isinstance(color, str) else color
        self.items.append((str(text), (int(xy[0]), int(xy[1])), size, col, bold, anchor))

    @staticmethod
    def width(text, size, bold=False):
        return font(size, bold).getlength(str(text))

    def render(self, img):
        if not self.items:
            return img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for text, xy, size, col, bold, anchor in self.items:
            d.text(xy, text, font=font(size, bold), fill=col, anchor=anchor)
        self.items = []
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def rounded_rect(img, p1, p2, color, r=12, thickness=-1):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    r = int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        t = thickness
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t, AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t, AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t, AA)


def card(img, x1, y1, x2, y2, fill=C["panel"], border=C["border"], r=16):
    rounded_rect(img, (x1, y1), (x2, y2), bgr(fill), r, -1)
    rounded_rect(img, (x1, y1), (x2, y2), bgr(border), r, 1)


def pill(img, tl, x, y, text, size, fg, bg, border=None, pad=10, bold=True):
    tw = TextLayer.width(text, size, bold)
    h = int(size * 1.75)
    w = int(tw + 2 * pad)
    x = int(max(0, min(x, OUT_W - w - 1)))
    y = int(max(0, min(y, OUT_H - h - 1)))
    rounded_rect(img, (x, y), (x + w, y + h), bgr(bg), h // 2, -1)
    if border:
        rounded_rect(img, (x, y), (x + w, y + h), bgr(border), h // 2, 1)
    tl.add(text, (x + pad, y + h // 2), size, fg, bold, "lm")
    return w, h


def corner_box(img, x1, y1, x2, y2, col, th=3):
    l = int(max(8, min(x2 - x1, y2 - y1) * 0.22))
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * l, cy), col, th, AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * l), col, th, AA)


def fmt_time(sec):
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:05.2f}"


# ------------------------------------------------------------------ #
#  FILES / LINE
# ------------------------------------------------------------------ #
def find_video(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"[ERROR] Video not found: {explicit}")
        return explicit
    os.makedirs(VIDEO_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(VIDEO_DIR, VIDEO_NAME + ".*")))
    if not files:
        sys.exit(f"[ERROR] No video found. Put your video in:\n   {VIDEO_DIR}\n"
                 f"   and name it {VIDEO_NAME}.mp4")
    return files[0]


def load_line():
    if os.path.exists(LINE_FILE):
        with open(LINE_FILE) as f:
            print(f"[INFO] Using entry line from {os.path.basename(LINE_FILE)}")
            return json.load(f)
    return DEFAULT_LINE


def pick_line(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("[ERROR] Could not read the video.")
    H, W = frame.shape[:2]
    s = min(1280 / W, 720 / H, 1.0)
    disp = cv2.resize(frame, (int(W * s), int(H * s)))
    pts = []
    steps = ["Click 1: START of the entry line", "Click 2: END of the entry line",
             "Click 3: any point INSIDE the barn", "S = save   R = reset   Q = quit"]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((x / s, y / s))

    win = "Pick entry line"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        img = disp.copy()
        sp = [(int(p[0] * s), int(p[1] * s)) for p in pts]
        if len(sp) >= 2:
            cv2.line(img, sp[0], sp[1], (0, 200, 255), 3, AA)
        for i, p in enumerate(sp):
            cv2.circle(img, p, 6, (0, 255, 160) if i == 2 else (0, 200, 255), -1, AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(img, steps[len(pts)], (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, AA)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k in (ord("q"), 27):
            break
        elif k == ord("s") and len(pts) == 3:
            cfg = {"a": [pts[0][0] / W, pts[0][1] / H], "b": [pts[1][0] / W, pts[1][1] / H],
                   "inside": [pts[2][0] / W, pts[2][1] / H]}
            with open(LINE_FILE, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[OK] Saved line to {LINE_FILE}. Now run: python main.py")
            break
    cv2.destroyAllWindows()


# ------------------------------------------------------------------ #
#  MAIN APP
# ------------------------------------------------------------------ #
class CowCounter:
    VID_BOX = (20, 84, 1420, 872)     # video area on the dashboard
    RX1, RX2 = 1440, 1900             # right column

    def __init__(self, args):
        self.args = args
        self.video_path = find_video(args.video)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            sys.exit(f"[ERROR] Cannot open {self.video_path}")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 < fps < 240 else 30.0
        self.total = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        # ---- entry line geometry (frame pixels) ----
        line = load_line()
        self.A = np.array([line["a"][0] * self.W, line["a"][1] * self.H], float)
        self.B = np.array([line["b"][0] * self.W, line["b"][1] * self.H], float)
        P = np.array([line["inside"][0] * self.W, line["inside"][1] * self.H], float)
        self.L = float(np.linalg.norm(self.B - self.A)) or 1.0
        self.inside_sign = 1.0 if self._cross(P) > 0 else -1.0
        u = (self.B - self.A) / self.L
        n = np.array([-u[1], u[0]])
        mid = (self.A + self.B) / 2
        self.n_in = n if self.sdist(mid + n * 5) > 0 else -n   # unit normal toward inside
        self.hyst = max(3.0, 0.04 * self.L)
        self.near = float(np.clip(1.0 * self.L, 0.05 * self.W, 0.14 * self.W))

        # ---- dashboard placement ----
        bx1, by1, bx2, by2 = self.VID_BOX
        bw, bh = bx2 - bx1, by2 - by1
        self.s = min(bw / self.W, bh / self.H)
        self.dw, self.dh = int(self.W * self.s), int(self.H * self.s)
        self.ox = bx1 + (bw - self.dw) // 2
        self.oy = by1 + (bh - self.dh) // 2
        self.thumb_w = 428
        self.thumb_h = min(236, int(self.thumb_w * self.H / self.W))

        # ---- state ----
        self.hs = 4
        self.heat = np.zeros((self.H // self.hs + 1, self.W // self.hs + 1), np.float32)
        self.cows = {}          # our own cow identities (survive tracker ID changes)
        self.tid2cow = {}
        self.next_cid = 1
        self.link_frames = int(LINK_SEC * self.fps)
        self.start_frames = int(1.0 * self.fps)
        self.next_disp = 1
        self.count = 0
        self.events = []
        self.pops = []
        self.flash_until = -1
        self.bump_until = -1
        self.fidx = -1
        self.in_view = 0
        self.outside = 0
        self.series_count, self.series_view = [], []
        self.conf_hist = deque(maxlen=300)

        # ---- OpenCV 5 + COOL Engine, AWS Cloud Manager, and Agentic Vision Controller ----
        self.cv5_engine = OpenCV5Engine(width=self.W, height=self.H, enable_cool_opt=True)
        self.aws = AWSCloudManager(output_dir=OUTPUT_DIR)
        self.agent = AgenticVisionController(aws_manager=self.aws, output_dir=OUTPUT_DIR)
        self.latest_lighting = {"doorway_luminance": 65.0, "yard_luminance": 150.0, "contrast_ratio": 2.3}
        self.latest_flow = {"mean_flow_speed": 0.0, "doorway_flow_speed": 0.0, "congestion_index": 0.0}
        self.latest_welfare = {"average_gait_score": 95.0, "cow_locomotion": {}}

        name = os.path.basename(args.model).replace(".pt", "")
        self.model_label = (name[:6].upper() + "-" + name[6:].upper()) if name.lower().startswith("yolo26") and len(name) > 6 else name.upper()

        with open(TRACKER_FILE, "w") as f:
            f.write(TRACKER_YAML)
        from ultralytics import YOLO
        print(f"[INFO] Loading {args.model} (downloads automatically the first time)...")
        self.model = YOLO(args.model)
        self.bg = self.make_background()

    # ---------------- geometry ----------------
    def _cross(self, p):
        return (self.B[0] - self.A[0]) * (p[1] - self.A[1]) - (self.B[1] - self.A[1]) * (p[0] - self.A[0])

    def sdist(self, p):
        """Signed distance to the line. > 0 = inside the barn."""
        return self.inside_sign * self._cross(p) / self.L

    def proj(self, p):
        """Position along the line segment (0 = start, 1 = end)."""
        d = self.B - self.A
        return float(((p[0] - self.A[0]) * d[0] + (p[1] - self.A[1]) * d[1]) / (self.L ** 2))

    def to_c(self, p):
        return (int(self.ox + p[0] * self.s), int(self.oy + p[1] * self.s))

    # ---------------- detection + tracking ----------------
    def detect(self, frame):
        r = self.model.track(frame, persist=True, tracker=TRACKER_FILE, classes=DETECT_CLASSES,
                             conf=self.args.conf, iou=IOU, imgsz=self.args.imgsz, verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        ids = b.id.int().cpu().tolist()
        confs = b.conf.cpu().numpy()
        return [(int(i), box.astype(float), float(c)) for box, i, c in zip(xyxy, ids, confs)]

    def ensure_disp(self, cow):
        if cow["disp"] is None:
            cow["disp"] = self.next_disp
            self.next_disp += 1

    @staticmethod
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def link_lost_cow(self, p, box, used):
        """A new tracker ID appeared. If a cow was lost a moment ago right where
        this one is, it is the same cow (overlap / occlusion) -> keep its history."""
        size = max(box[2] - box[0], box[3] - box[1])
        best, best_d = None, None
        for cid, cow in self.cows.items():
            if cid in used or not cow["hist"]:
                continue
            gap = self.fidx - cow["last"]
            if gap <= 0 or gap > self.link_frames:
                continue
            if (cow["counted"] or cow["side"] == 1) and self.sdist(p) < self.hyst:
                continue   # a cow that went in doesn't come back out: this is the next cow
            pred = cow["hist"][-1] + cow["vel"] * min(gap, 10)
            csize = max(cow["box"][2] - cow["box"][0], cow["box"][3] - cow["box"][1])
            d = float(np.linalg.norm(p - pred))
            if d < LINK_DIST * max(size, csize) and (best_d is None or d < best_d):
                best, best_d = cid, d
        return best

    def register(self, cid, p, box, how="line"):
        cow = self.cows[cid]
        # same cow detected twice (two boxes / two IDs) crossing together -> count once
        for e in self.events[-6:]:
            if self.fidx - e["frame"] <= int(0.3 * self.fps) and self.iou(e["box"], box) > 0.65:
                cow["counted"], cow["entry_no"] = True, e["no"]
                return
        self.ensure_disp(cow)
        self.count += 1
        cow["counted"] = True
        cow["pending"] = False
        cow["how"] = how
        cow["entry_no"] = self.count
        ev_obj = {"no": self.count, "cow": cow["disp"], "track_id": cow["tid"],
                  "frame": self.fidx, "time": self.fidx / self.fps, "box": box.copy(), "how": how}
        self.events.append(ev_obj)
        gait_info = self.latest_welfare.get("cow_locomotion", {}).get(cow["disp"])
        self.aws.put_cow_entry_event(ev_obj, gait_info=gait_info)
        self.pops.append((np.array(p, float), self.fidx))
        self.flash_until = self.fidx + int(0.35 * self.fps)
        self.bump_until = self.fidx + int(0.45 * self.fps)

    def qualified(self, cow):
        """Is this a real cow that came from outside? Cows already in view when the
        video starts are trusted right away (they may be standing at the line)."""
        if cow["first"] <= self.start_frames:
            return cow["out_frames"] >= 1 and cow["hits"] >= 2
        return cow["out_frames"] >= OUT_MIN and cow.get("far_out") and cow["hits"] >= MIN_HITS

    def straddles_line(self, box):
        x1, y1, x2, y2 = box
        corners = [np.array(c) for c in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
        ds = [self.sdist(c) for c in corners]
        ts = [self.proj(c) for c in corners]
        return min(ds) < 0 < max(ds) and min(ts) <= 1.0 and max(ts) >= 0.0

    def check_door_losses(self, final=False):
        """A cow that came from the yard and vanished while its body was across the
        line (half inside the dark doorway) has crossed -> count it once."""
        wait = 1 if final else self.link_frames + 1
        for cid, cow in self.cows.items():
            if cow["counted"] or cow.get("lost_checked") or self.fidx - cow["last"] < wait:
                continue
            cow["lost_checked"] = True
            if cow["pending"] and cow["out_frames"] >= 1:
                self.register(cid, cow["hist"][-1], cow["box"], "line")
            elif (cow["side"] == -1 and self.qualified(cow) and self.straddles_line(cow["box"])
                    and len(cow["hist"]) >= 2 and cow["vel"] @ self.n_in > 0):
                self.register(cid, cow["hist"][-1], cow["box"], "doorway")

    def update(self, dets):
        f = self.fidx
        heat_tmp = np.zeros_like(self.heat)
        used = set()
        # known IDs first, so new IDs can only link to cows that really disappeared
        order = [d for d in dets if d[0] in self.tid2cow] + [d for d in dets if d[0] not in self.tid2cow]
        for tid, box, c in order:
            x1, y1, x2, y2 = box
            p = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            cid = self.tid2cow.get(tid)

            # ID switch guard: tracker ID jumped onto a different cow
            if cid is not None:
                cow = self.cows[cid]
                gap = f - cow["last"]
                size = max(cow["box"][2] - cow["box"][0], cow["box"][3] - cow["box"][1])
                if cow["hist"] and gap <= 5 and np.linalg.norm(p - cow["hist"][-1]) > JUMP * size * gap:
                    cow["tid"] = None
                    del self.tid2cow[tid]
                    cid = None
            if cid is not None and cid in used:
                cid = None
                self.tid2cow.pop(tid, None)

            if cid is None:
                cid = self.link_lost_cow(p, box, used)
                if cid is None:
                    cid = self.next_cid
                    self.next_cid += 1
                    self.cows[cid] = {"hist": deque(maxlen=TRAIL_LEN), "hits": 0, "side": 0,
                                      "out_frames": 0, "counted": False, "last": f, "box": box,
                                      "disp": None, "entry_no": None, "vel": np.zeros(2), "tid": None,
                                      "first": f, "pending": False, "how": None}
                old = self.cows[cid]["tid"]
                if old is not None and self.tid2cow.get(old) == cid:
                    del self.tid2cow[old]
                self.cows[cid]["tid"] = tid
                self.tid2cow[tid] = cid
            used.add(cid)

            cow = self.cows[cid]
            if cow["hist"]:
                gap = max(1, f - cow["last"])
                cow["vel"] = 0.7 * cow["vel"] + 0.3 * (p - cow["hist"][-1]) / gap
            cow["hits"] += 1
            cow["last"] = f
            cow["lost_checked"] = False
            cow["box"] = box
            cow["hist"].append(p)
            if cow["hits"] >= MIN_HITS:
                self.ensure_disp(cow)

            # ---- the counting rule: outside -> inside across the line = +1 (once per cow)
            d, t = self.sdist(p), self.proj(p)
            size = max(x2 - x1, y2 - y1)
            if cow["hits"] == 1 and cow["first"] <= self.start_frames and -self.hyst <= d <= 0:
                cow["side"], cow["out_frames"] = -1, 1      # standing on the line at video start
            if d < -self.hyst:
                cow["out_frames"] += 1
                cow["side"] = -1
                cow["far_out"] = cow.get("far_out", False) or d < -0.35 * size
            elif d > self.hyst:
                if cow["side"] == -1 and not cow["counted"] and -0.3 <= t <= 1.5:
                    if self.qualified(cow):
                        self.register(cid, p, box, "line")
                    elif cow["first"] <= self.start_frames or cow.get("far_out"):
                        cow["pending"] = True     # crossed, confirm once tracked a few frames
                cow["side"] = 1
            if cow["pending"] and not cow["counted"] and cow["hits"] >= 3:
                self.register(cid, p, box, "line")

            self.conf_hist.append(c)
            fx, fy = p[0] / self.hs, (y2 - 0.08 * (y2 - y1)) / self.hs   # footprint
            cv2.circle(heat_tmp, (int(fx), int(fy)), max(2, int((x2 - x1) / self.hs * 0.18)), 1.0, -1)
        self.heat += heat_tmp
        self.in_view = len(dets)
        self.check_door_losses()
        self.outside = sum(1 for cid in used if not self.cows[cid]["counted"] and self.cows[cid]["side"] <= 0)
        self.series_count.append(self.count)
        self.series_view.append(self.in_view)

    # ---------------- heatmap ----------------
    def heat_maps(self):
        if self.heat.max() <= 0:
            return None, None
        hb = cv2.GaussianBlur(self.heat, (0, 0), sigmaX=max(3.0, self.W / self.hs / 80))
        n = np.power(hb / (hb.max() + 1e-6), 0.55)
        return cv2.applyColorMap((n * 255).astype(np.uint8), CMAP), n

    # ---------------- static background ----------------
    def make_background(self):
        bg = np.zeros((OUT_H, OUT_W, 3), np.uint8)
        top, bot = np.array(bgr("#0E1424"), np.float32), np.array(bgr("#060910"), np.float32)
        g = np.linspace(0, 1, OUT_H)[:, None, None]
        bg[:] = (top * (1 - g) + bot * g).astype(np.uint8)
        bg[::32, ::32] = bgr("#1C2740")
        tl = TextLayer()

        # header
        rounded_rect(bg, (20, 16), (68, 64), bgr(C["accent"]), 12, -1)
        tl.add("SD", (44, 40), 20, C["dark"], True, "mm")
        tl.add("SMART DAIRY", (84, 14), 28, C["accent"], True)
        w = TextLayer.width("SMART DAIRY", 28, True)
        tl.add("AI COW ENTRY ANALYTICS", (84 + w + 14, 14), 28, C["text"], True)
        tl.add("OpenCV 5 + COOL (AWS Graviton)  •  YOLO26 + BoT-SORT  •  AWS Bedrock Agentic Vision Loop",
               (84, 52), 15, C["muted"])

        # video frame
        x1, y1, x2, y2 = self.VID_BOX
        card(bg, x1 - 2, y1 - 2, x2 + 2, y2 + 2, "#05070C", C["border"], 14)

        # right column
        R1, R2 = self.RX1, self.RX2
        card(bg, R1, 84, R2, 320)
        tl.add("COWS ENTERED BARN", (R1 + 24, 104), 16, C["muted"], True)
        tl.add("unique cows counted at the entry line", (R1 + 24, 286), 14, C["muted"])

        half = (R2 - R1 - 16) // 2
        small = [(R1, 336, "IN VIEW NOW", C["blue"]), (R1 + half + 16, 336, "STILL OUTSIDE", C["purple"]),
                 (R1, 472, "AVG CONFIDENCE", C["amber"]), (R1 + half + 16, 472, "LAST ENTRY", C["pink"])]
        for sx, sy, label, col in small:
            card(bg, sx, sy, sx + half, sy + 120)
            rounded_rect(bg, (sx + 1, sy + 18), (sx + 5, sy + 102), bgr(col), 2, -1)
            tl.add(label, (sx + 20, sy + 16), 13, C["muted"], True)

        card(bg, R1, 608, R2, 910)
        tl.add("MOVEMENT HEATMAP", (R1 + 24, 626), 16, C["muted"], True)
        lg = cv2.applyColorMap(np.tile(np.linspace(0, 255, 120).astype(np.uint8), (8, 1)), CMAP)
        bg[632:640, R2 - 24 - 120:R2 - 24] = lg
        tl.add("low", (R2 - 150, 636), 12, C["muted"], False, "rm")
        tl.add("high", (R2 - 24, 648), 12, C["muted"], False, "ra")

        card(bg, R1, 926, R2, 1060)
        tl.add("PIPELINE", (R1 + 24, 940), 14, C["muted"], True)
        tl.add("Detector", (R1 + 24, 968), 15, C["muted"])
        tl.add(f"{self.model_label}  @{self.args.imgsz}", (R2 - 24, 968), 15, C["text"], True, "ra")
        tl.add("Tracker", (R1 + 24, 992), 15, C["muted"])
        tl.add("BoT-SORT  +  camera motion comp.", (R2 - 24, 992), 15, C["text"], True, "ra")
        tl.add("Progress", (R1 + 24, 1012), 13, C["muted"])

        # bottom strip
        card(bg, 20, 888, 930, 1060)
        tl.add("ENTRY TIMELINE", (40, 902), 14, C["muted"], True)
        cv2.circle(bg, (720, 911), 5, bgr(C["accent"]), -1, AA)
        tl.add("Cows entered", (732, 911), 13, C["text"], False, "lm")
        cv2.circle(bg, (840, 911), 5, bgr(C["blue"]), -1, AA)
        tl.add("In view", (852, 911), 13, C["text"], False, "lm")
        card(bg, 946, 888, 1420, 1060)
        tl.add("RECENT ENTRIES", (966, 902), 14, C["muted"], True)
        return tl.render(bg)

    # ---------------- per-frame drawing ----------------
    def draw_line(self, img, tl, glow):
        a, b = self.to_c(self.A), self.to_c(self.B)
        flashing = self.fidx <= self.flash_until
        col = bgr(C["accent"] if flashing else C["amber"])
        if glow:
            cv2.line(img, a, b, col, 22 if flashing else 14, AA)
            return
        cv2.line(img, a, b, col, 4, AA)
        for p in (a, b):
            cv2.circle(img, p, 8, (255, 255, 255), -1, AA)
            cv2.circle(img, p, 5, col, -1, AA)
        mid = np.array(self.to_c((self.A + self.B) / 2), float)
        n = self.n_in
        start = (mid - n * 55).astype(int)
        end = (mid - n * 12).astype(int)
        cv2.arrowedLine(img, tuple(start), tuple(end), col, 3, AA, tipLength=0.35)
        lp = mid - n * 70
        pill(img, tl, lp[0] - 40, lp[1] - 6, "ENTRY LINE", 13,
             C["accent"] if flashing else C["amber"], C["dark"], C["accent"] if flashing else C["amber"])

    def draw_tracks(self, img, tl, dets, glow):
        for tid, box, c in dets:
            cid = self.tid2cow.get(tid)
            if cid is None:
                continue
            tr = self.cows[cid]
            col_hex = BOX_COLOR
            col = bgr(col_hex)
            pts = [self.to_c(p) for p in tr["hist"]]
            if glow:
                if len(pts) > 1:
                    cv2.polylines(img, [np.array(pts, np.int32)], False, col, 8, AA)
                continue
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], col, max(1, int(1 + 3 * k / len(pts))), AA)
            x1, y1 = self.to_c((box[0], box[1]))
            x2, y2 = self.to_c((box[2], box[3]))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, AA)
            corner_box(img, x1, y1, x2, y2, col, 3)
            cv2.circle(img, pts[-1], 5, col, -1, AA)
            cv2.circle(img, pts[-1], 2, (255, 255, 255), -1, AA)
            if tr["disp"] is not None:
                label = f"COW {tr['disp']}" + ("  ·  IN" if tr["counted"] else "")
                pill(img, tl, x1, max(self.oy + 2, y1 - 26), label, 12, C["dark"], col_hex, pad=7)

    def draw_pops(self, img, tl, glow):
        alive = []
        for p, f0 in self.pops:
            age = (self.fidx - f0) / self.fps
            if age > 0.9:
                continue
            alive.append((p, f0))
            c = self.to_c(p)
            if glow:
                cv2.circle(img, c, int(12 + age * 70), bgr(C["accent"]), 4, AA)
            else:
                tl.add("+1", (c[0], c[1] - 36 - int(age * 80)), 42, C["accent"], True, "mm")
        if not glow:
            self.pops = alive

    def draw_dashboard(self, img, tl, frame, hm_col, hm_n):
        R1, R2 = self.RX1, self.RX2
        t_now = max(0, self.fidx) / self.fps
        t_tot = self.total / self.fps if self.total else t_now

        # header right: live + time
        pulse = 0.5 + 0.5 * math.sin(self.fidx / self.fps * 2 * math.pi)
        rounded_rect(img, (1556, 22), (1650, 56), bgr("#2A1218"), 17, -1)
        cv2.circle(img, (1576, 39), int(5 + 2 * pulse), bgr(C["red"]), -1, AA)
        tl.add("LIVE", (1592, 39), 15, C["red"], True, "lm")
        tl.add(f"{fmt_time(t_now)} / {fmt_time(t_tot)}", (1900, 16), 22, C["text"], True, "ra")
        tl.add(f"Frame {self.fidx + 1}" + (f" / {self.total}" if self.total else ""),
               (1900, 50), 13, C["muted"], False, "ra")

        # video tag
        pill(img, tl, self.ox + 14, self.oy + 14, "CAM 01  •  BARN ENTRANCE", 13, C["text"], "#0A0E17", C["border"])

        # big counter + donut
        bump = self.fidx <= self.bump_until
        tl.add(str(self.count), (R1 + 22, 128), 128 if bump else 116, C["accent"] if bump else C["text"], True)
        seen = self.count + self.outside
        frac = min(1.0, self.count / seen) if seen else 0.0
        cx, cy, rr = R2 - 92, 200, 58
        cv2.circle(img, (cx, cy), rr, bgr(C["border"]), 10, AA)
        if frac > 0:
            cv2.ellipse(img, (cx, cy), (rr, rr), -90, 0, 360 * frac, bgr(C["accent"]), 10, AA)
        tl.add(f"{frac * 100:.0f}%", (cx, cy - 6), 22, C["text"], True, "mm")
        tl.add("inside", (cx, cy + 18), 12, C["muted"], False, "mm")

        # small cards
        half = (R2 - R1 - 16) // 2
        avg = (sum(self.conf_hist) / len(self.conf_hist) * 100) if self.conf_hist else 0
        tl.add(str(self.in_view), (R1 + 20, 380), 48, C["blue"], True)
        tl.add(str(self.outside), (R1 + half + 36, 380), 48, C["purple"], True)
        tl.add(f"{avg:.0f}%", (R1 + 20, 516), 48, C["amber"], True)
        if self.events:
            e = self.events[-1]
            tl.add(f"COW {e['cow']}", (R1 + half + 36, 522), 34, C["pink"], True)
            tl.add(f"at {fmt_time(e['time'])}", (R1 + half + 36, 566), 14, C["muted"])
        else:
            tl.add("—", (R1 + half + 36, 516), 44, C["pink"], True)

        # heatmap thumbnail
        tx, ty = R1 + 16, 664
        th_img = cv2.resize(frame, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_AREA)
        th_img = (th_img * 0.55).astype(np.uint8)
        if hm_col is not None:
            hc = cv2.resize(hm_col, (self.thumb_w, self.thumb_h))
            hn = cv2.resize(hm_n, (self.thumb_w, self.thumb_h))
            a = (np.clip((hn - 0.03) / 0.97, 0, 1) * 0.85)[..., None]
            th_img = (th_img * (1 - a) + hc * a).astype(np.uint8)
        img[ty:ty + self.thumb_h, tx:tx + self.thumb_w] = th_img
        rounded_rect(img, (tx, ty), (tx + self.thumb_w, ty + self.thumb_h), bgr(C["border"]), 6, 1)

        # progress
        prog = (self.fidx + 1) / self.total if self.total else 0
        tl.add(f"{prog * 100:.0f}%", (R2 - 24, 1012), 13, C["text"], True, "ra")
        rounded_rect(img, (R1 + 24, 1034), (R2 - 24, 1042), bgr(C["border"]), 4, -1)
        if prog > 0:
            rounded_rect(img, (R1 + 24, 1034), (int(R1 + 24 + (R2 - R1 - 48) * min(1, prog)), 1042),
                         bgr(C["accent"]), 4, -1)

        self.draw_timeline(img, tl)
        self.draw_entries(tl)

    def draw_timeline(self, img, tl):
        x1, y1, x2, y2 = 70, 932, 910, 1044
        n = len(self.series_count)
        if n == 0:
            return
        span = max(self.total, n, 2) - 1
        ymax = max(5, max(self.series_view), self.count)
        ymax = int(math.ceil(ymax / 5.0) * 5)
        for k in range(3):
            yy = int(y2 - (y2 - y1) * k / 2)
            cv2.line(img, (x1, yy), (x2, yy), bgr("#1E2840"), 1, AA)
            tl.add(str(int(ymax * k / 2)), (x1 - 10, yy), 12, C["muted"], False, "rm")
        step = max(1, n // 400)
        idx = list(range(0, n, step)) + [n - 1]

        def P(i, v):
            return (int(x1 + (x2 - x1) * i / span), int(y2 - (y2 - y1) * v / ymax))

        view = [P(i, self.series_view[i]) for i in idx]
        poly = np.array(view + [(view[-1][0], y2), (view[0][0], y2)], np.int32)
        cv2.fillPoly(img, [poly], bgr("#16304F"))
        cv2.polylines(img, [np.array(view, np.int32)], False, bgr(C["blue"]), 2, AA)
        steps_pts = []
        for j, i in enumerate(idx):
            p = P(i, self.series_count[i])
            if steps_pts:
                steps_pts.append((p[0], steps_pts[-1][1]))
            steps_pts.append(p)
        cv2.polylines(img, [np.array(steps_pts, np.int32)], False, bgr(C["accent"]), 3, AA)
        cv2.circle(img, steps_pts[-1], 6, bgr(C["accent"]), -1, AA)
        cv2.circle(img, steps_pts[-1], 2, (255, 255, 255), -1, AA)

    def draw_entries(self, tl):
        x, y = 966, 934
        if not self.events:
            tl.add("Waiting for cows to cross the entry line…", (x, y + 10), 15, C["muted"])
            return
        for k, e in enumerate(reversed(self.events[-4:])):
            yy = y + k * 30
            col = C["accent"] if k == 0 else C["muted"]
            tl.add(f"#{e['no']:02d}", (x, yy), 15, col, True)
            tl.add(f"COW {e['cow']} entered the barn", (x + 48, yy), 15, C["text"] if k == 0 else C["muted"],
                   k == 0)
            tl.add(fmt_time(e["time"]), (1400, yy), 15, col, True, "ra")

    def draw_final_banner(self, img, tl):
        x1, y1 = self.ox, self.oy
        roi = img[y1:y1 + self.dh, x1:x1 + self.dw]
        img[y1:y1 + self.dh, x1:x1 + self.dw] = (roi * 0.45).astype(np.uint8)
        cx, cy = x1 + self.dw // 2, y1 + self.dh // 2
        card(img, cx - 330, cy - 150, cx + 330, cy + 150, "#0E1626", C["accent"], 22)
        tl.add("TOTAL COWS ENTERED", (cx, cy - 92), 30, C["muted"], True, "mm")
        tl.add(str(self.count), (cx, cy + 10), 140, C["accent"], True, "mm")
        tl.add("counted at the barn entry line  •  YOLO26 + BoT-SORT", (cx, cy + 112), 17, C["muted"], False, "mm")

    def render(self, frame, dets, final=False):
        canvas = self.bg.copy()
        vid = cv2.resize(frame, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        hm_col, hm_n = self.heat_maps()
        if hm_col is not None:
            hc = cv2.resize(hm_col, (self.dw, self.dh))
            hn = cv2.resize(hm_n, (self.dw, self.dh))
            a = (np.clip((hn - 0.10) / 0.90, 0, 1) * 0.40)[..., None]
            vid = (vid * (1 - a) + hc * a).astype(np.uint8)
        canvas[self.oy:self.oy + self.dh, self.ox:self.ox + self.dw] = vid

        tl = TextLayer()
        glow = canvas.copy()
        self.draw_line(glow, tl, True)
        self.draw_tracks(glow, tl, dets, True)
        self.draw_pops(glow, tl, True)
        canvas = cv2.addWeighted(glow, 0.35, canvas, 0.65, 0)

        self.draw_line(canvas, tl, False)
        self.draw_tracks(canvas, tl, dets, False)
        self.draw_pops(canvas, tl, False)
        self.draw_dashboard(canvas, tl, frame, hm_col, hm_n)
        if final:
            canvas = tl.render(canvas)
            self.draw_final_banner(canvas, tl)
        return tl.render(canvas)

    # ---------------- run ----------------
    def run(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "cow_counter_dashboard.mp4")
        # Prefer H.264 ('avc1') so HTML5 browsers can play the MP4 directly; fallback to 'mp4v'
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), self.fps, (OUT_W, OUT_H))
        if not writer.isOpened():
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (OUT_W, OUT_H))
        show = not self.args.no_show
        win = "Smart Dairy - Cow Counter (press Q to stop)"
        if show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 720)

        print(f"[INFO] Video: {self.video_path}  ({self.W}x{self.H} @ {self.fps:.1f} fps, {self.total} frames)")
        t0, last_frame, last_dets = time.time(), None, []
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.fidx += 1
            # 1. OpenCV 5 + COOL Preprocessing & Dense Optical Flow Analysis
            proc_frame, self.latest_lighting = self.cv5_engine.enhance_doorway_lighting(
                frame, apply_clahe=self.agent.vision_params.get("clahe_shadow_recovery", False)
            )
            self.latest_flow = self.cv5_engine.analyze_crowd_optical_flow(frame, self.A, self.B)

            # 2. YOLO26 Detection & BoT-SORT Multi-Object Tracking
            dets = self.detect(proc_frame)
            self.update(dets)

            # 3. OpenCV 5 BEV Homography Locomotion & Agentic Vision Loop
            self.latest_welfare = self.cv5_engine.evaluate_cow_gait_and_welfare(self.cows, self.fps)
            avg_c = (sum(self.conf_hist) / len(self.conf_hist)) if self.conf_hist else 0.65
            self.agent.step(
                frame_idx=self.fidx,
                video_time_sec=self.fidx / self.fps,
                entered_count=self.count,
                in_view_count=self.in_view,
                outside_count=self.outside,
                avg_conf=avg_c,
                lighting_metrics=self.latest_lighting,
                flow_metrics=self.latest_flow,
                welfare_metrics=self.latest_welfare,
            )

            canvas = self.render(frame, dets)
            writer.write(canvas)
            last_frame, last_dets = frame, dets
            if show:
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("[INFO] Stopped by user.")
                    break
            if self.fidx % 30 == 0:
                sp = (self.fidx + 1) / max(0.001, (time.time() - t0))
                self.aws.publish_cloudwatch_metrics(
                    total_entered=self.count,
                    in_view=self.in_view,
                    outside_waiting=self.outside,
                    congestion_index=self.latest_flow.get("congestion_index", 0.0),
                    fps_throughput=sp,
                )
                print(f"  frame {self.fidx + 1}/{self.total or '?'}  |  in view {self.in_view}"
                      f"  |  entered {self.count}  |  agent actions {len(self.agent.decision_trace)}"
                      f"  |  {sp:.1f} fps")

        if last_frame is not None:
            self.check_door_losses(final=True)
            end = self.render(last_frame, last_dets, final=True)
            for _ in range(int(END_HOLD_SEC * self.fps)):
                writer.write(end)
            self.save_heatmap(last_frame)
        writer.release()
        self.cap.release()
        if show:
            cv2.destroyAllWindows()
        self.save_csv()
        rec = self.save_cow_records()

        # Sync final artifacts & state to AWS S3 / DynamoDB / local mirror
        hm_path = os.path.join(OUTPUT_DIR, "heatmap_final.png")
        csv_path = os.path.join(OUTPUT_DIR, "entries.csv")
        self.aws.upload_artifact_to_s3(out_path)
        self.aws.upload_artifact_to_s3(hm_path)
        self.aws.upload_artifact_to_s3(csv_path)
        self.aws.upload_artifact_to_s3(rec)
        agent_log_path = self.agent.save_state()
        aws_ledger_path = self.aws.save_ledger()

        print("\n" + "=" * 62)
        print(f"  TOTAL COWS ENTERED     : {self.count}")
        print(f"  TRACK IDs USED         : {self.next_disp - 1}")
        print(f"  AGENTIC DECISIONS      : {len(self.agent.decision_trace)}")
        print(f"  Dashboard video        : {out_path}")
        print(f"  Heatmap image          : {hm_path}")
        print(f"  Entry log (CSV)        : {csv_path}")
        print(f"  Every cow record       : {rec}")
        print(f"  Agentic Vision trace   : {agent_log_path}")
        print(f"  AWS Cloud ledger       : {aws_ledger_path}")
        print("=" * 62)

    def save_heatmap(self, frame):
        hm_col, hm_n = self.heat_maps()
        img = frame.copy()
        if hm_col is not None:
            hc = cv2.resize(hm_col, (self.W, self.H))
            hn = cv2.resize(hm_n, (self.W, self.H))
            a = (np.clip((hn - 0.03) / 0.97, 0, 1) * 0.7)[..., None]
            img = (img * (1 - a) + hc * a).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "heatmap_final.png"), img)

    def save_cow_records(self):
        """One row per cow: what happened to it during the video."""
        path = os.path.join(OUTPUT_DIR, "cow_records.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cow_number", "first_seen_sec", "last_seen_sec", "status", "entry_no", "entered_at_sec"])
            ev = {e["cow"]: e for e in self.events}
            for cow in sorted(self.cows.values(), key=lambda c: (c["disp"] is None, c["disp"] or 0)):
                if cow["disp"] is None:
                    continue            # flicker boxes that never became a real track
                if cow["counted"]:
                    status = "ENTERED (crossed line)" if cow["how"] == "line" else "ENTERED (vanished in doorway)"
                elif cow["side"] == 1 and cow["first"] <= self.start_frames:
                    status = "already inside at start"
                elif cow["side"] == 1:
                    status = "inside (not from yard)"
                else:
                    status = "still outside"
                e = ev.get(cow["disp"])
                w.writerow([cow["disp"], f"{cow['first'] / self.fps:.2f}", f"{cow['last'] / self.fps:.2f}",
                            status, e["no"] if e else "", f"{e['time']:.2f}" if e else ""])
        return path

    def save_csv(self):
        with open(os.path.join(OUTPUT_DIR, "entries.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entry_no", "cow_number", "tracker_id", "frame", "time_sec"])
            for e in self.events:
                w.writerow([e["no"], e["cow"], e["track_id"], e["frame"], f"{e['time']:.2f}"])


def main():
    ap = argparse.ArgumentParser(description="Smart Dairy - AI cow entry counter (OpenCV 5 + COOL + AWS + Agentic Vision)")
    ap.add_argument("--video", default=None, help="video path (default: video/your_video.*)")
    ap.add_argument("--model", default=MODEL_NAME, help="yolo26n/s/m/l/x .pt")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--conf", type=float, default=DET_CONF)
    ap.add_argument("--no-show", action="store_true", help="don't open the preview window")
    ap.add_argument("--pick-line", action="store_true", help="click a new entry line on the first frame")
    ap.add_argument("--benchmark-cool", action="store_true", help="run OpenCV 5 + COOL (AWS Graviton) benchmark and save report")
    args = ap.parse_args()

    if args.benchmark_cool:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        res = run_cool_benchmark()
        out_bench = os.path.join(OUTPUT_DIR, "cool_benchmark_report.json")
        with open(out_bench, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        print(f"\n[OK] Saved COOL benchmark report to {out_bench}")
        return

    if args.pick_line:
        pick_line(find_video(args.video))
        return
    CowCounter(args).run()


if __name__ == "__main__":
    main()