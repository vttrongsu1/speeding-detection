"""
Vehicle Speed Detection & Web Monitoring System (Flask + YOLO + ByteTrack + Perspective Transform)
Integrated Server Script with 3-Step Interactive Calibration & Synchronous Frame Capture (test.py)
"""

import argparse
import os
import threading
import time
import urllib.request
import warnings
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)


# Parse Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time Vehicle Speed Detection Web Application"
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model weights")
    parser.add_argument("--source", type=str, default="", help="Default video file path, webcam index, or RTSP stream")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Flask server host")
    parser.add_argument("--port", type=int, default=5000, help="Flask server port")
    return parser.parse_args()


ARGS = parse_arguments()
app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Allow up to 500MB uploads


# Helper to locate video file inside 'files' directory or relative paths
def resolve_video_path(filename_or_path):
    src = str(filename_or_path).strip()
    if not src:
        return None

    # Check if exact path exists
    if os.path.exists(src):
        return src

    # Check inside 'files/' directory
    files_dir_path = os.path.join("files", src)
    if os.path.exists(files_dir_path):
        return files_dir_path

    # Check parent 'files/' directory if running inside 'files'
    if os.path.exists(os.path.basename(src)):
        return os.path.basename(src)

    return src


# Camera-to-World Perspective Transformer
class Cam2WorldMapper:
    def __init__(self):
        self.M = None

    def find_perspective_transform(self, image_pts, world_pts):
        image_pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 1, 2)
        world_pts = np.asarray(world_pts, dtype=np.float32).reshape(-1, 1, 2)
        self.M = cv2.getPerspectiveTransform(image_pts, world_pts)
        return self.M

    def map(self, image_pts):
        if len(image_pts) == 0:
            return np.empty((0, 2))
        if self.M is None:
            return np.array(image_pts, dtype=np.float32)
        image_pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(image_pts, self.M).reshape(-1, 2)


# Speedometer to track vehicle movement with high precision
class Speedometer:
    def __init__(self, mapper, fps):
        self.mapper = mapper
        self.fps = fps
        # track_id -> list of (timestamp, x_world, y_world, x_img, y_img)
        self.tracks = {}
        # track_id -> list of smoothed speeds (km/h)
        self.speeds = {}
        # track_id -> float (maximum recorded speed)
        self.max_speeds = {}
        # track_id -> str (class name)
        self.classes = {}
        # track_id -> int (total frames tracked)
        self.frame_counts = defaultdict(int)

    def update(self, track_id, cx_img, cy_img, cls_name):
        now = time.time()
        # Map bottom-center anchor (cx_img, cy_img) to real-world meters
        world_pos = self.mapper.map([[cx_img, cy_img]])[0]
        x_w, y_w = world_pos[0], world_pos[1]

        if track_id not in self.tracks:
            self.tracks[track_id] = []
            self.speeds[track_id] = []
            self.max_speeds[track_id] = 0.0
            self.classes[track_id] = cls_name

        self.tracks[track_id].append((now, x_w, y_w, cx_img, cy_img))
        self.frame_counts[track_id] += 1

        # Keep history bounded (~2 seconds)
        if len(self.tracks[track_id]) > 60:
            self.tracks[track_id].pop(0)

        # Warm-up period: Ignore initial 8 frames to allow ByteTrack to stabilize
        if self.frame_counts[track_id] < 8:
            return

        # Calculate speed over a 5-frame sliding window
        history = self.tracks[track_id]
        if len(history) >= 5:
            prev_state = history[-5]
            curr_state = history[-1]
            dt = curr_state[0] - prev_state[0]
            
            if dt > 0:
                dist_m = np.linalg.norm(np.array([prev_state[1], prev_state[2]]) - np.array([curr_state[1], curr_state[2]]))
                speed_mps = dist_m / dt
                speed_kph = speed_mps * 3.6
                
                # Filter unrealistically high speed outliers (> 200 km/h)
                if speed_kph < 200:
                    # Exponential Moving Average (EMA) smoothing (alpha = 0.25)
                    if not self.speeds[track_id]:
                        smooth_speed = speed_kph
                    else:
                        alpha = 0.25
                        smooth_speed = alpha * speed_kph + (1.0 - alpha) * self.speeds[track_id][-1]
                    
                    self.speeds[track_id].append(smooth_speed)
                    if len(self.speeds[track_id]) > 10:
                        self.speeds[track_id].pop(0)
                        
                    if smooth_speed > self.max_speeds[track_id]:
                        self.max_speeds[track_id] = smooth_speed

    def get_speed(self, track_id):
        if track_id in self.speeds and self.speeds[track_id]:
            return self.speeds[track_id][-1]
        return 0.0

    def get_max_speed(self, track_id):
        return self.max_speeds.get(track_id, 0.0)


# Thread-Safe Shared Application State
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.first_frame_jpeg = None
        self.frame_w = 0
        self.frame_h = 0
        self.running = False
        self.paused_for_calibration = True
        self.source = None
        self.error_msg = None
        
        # Interactive Calibration Settings (Multi-zone support)
        self.zones = []
        self.start_signal = threading.Event()
        
        # Real-time Live Metrics
        self.active_vehicles = 0
        self.avg_speed = 0.0
        self.overspeed_count = 0
        self.total_vehicles = 0
        self.fps = 0.0
        
        # Vehicle logs
        self.vehicle_logs = []

    def snapshot(self):
        with self.lock:
            sorted_logs = sorted(self.vehicle_logs, key=lambda x: x["timestamp"], reverse=True)
            return {
                "running": self.running,
                "paused_for_calibration": self.paused_for_calibration,
                "source": self.source,
                "error_msg": self.error_msg,
                "frame_w": self.frame_w,
                "frame_h": self.frame_h,
                "active_vehicles": self.active_vehicles,
                "avg_speed": round(self.avg_speed, 1),
                "overspeed_count": self.overspeed_count,
                "total_vehicles": self.total_vehicles,
                "fps": round(self.fps, 1),
                "vehicle_logs": sorted_logs[:100],
                "zones": self.zones
            }


STATE = SharedState()
active_thread_id = None


def update_global_vehicle(tid, cls_name, speed, max_speed, is_violating):
    tid = int(tid)
    is_violating = bool(is_violating)
    with STATE.lock:
        now_str = time.strftime("%H:%M:%S")
        found = False
        for log in STATE.vehicle_logs:
            if log["track_id"] == tid:
                log["current_speed"] = int(float(speed))
                log["max_speed"] = int(float(max_speed))
                log["violating"] = is_violating
                found = True
                break
        if not found:
            STATE.vehicle_logs.append({
                "track_id": tid,
                "class": str(cls_name),
                "current_speed": int(float(speed)),
                "max_speed": int(float(max_speed)),
                "violating": is_violating,
                "timestamp": now_str
            })


def update_shared_metrics(active_tids, fps_smooth):
    with STATE.lock:
        STATE.active_vehicles = int(len(active_tids))
        STATE.fps = float(fps_smooth)
        
        speeds = []
        for tid in active_tids:
            for log in STATE.vehicle_logs:
                if log["track_id"] == int(tid) and log["current_speed"] > 0:
                    speeds.append(log["current_speed"])
                    break
        if speeds:
            STATE.avg_speed = float(sum(speeds) / len(speeds))
        else:
            STATE.avg_speed = 0.0
            
        STATE.total_vehicles = int(len(STATE.vehicle_logs))
        STATE.overspeed_count = int(sum(1 for log in STATE.vehicle_logs if log["violating"]))


# Synchronously capture first frame of video to guarantee immediate readiness
def capture_first_frame_sync(source_path):
    resolved = resolve_video_path(source_path)
    if not resolved or not os.path.exists(resolved):
        return False

    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        return False

    ok, first_frame = cap.read()
    cap.release()

    if ok:
        h, w = first_frame.shape[:2]
        ok_enc, buf = cv2.imencode(".jpg", first_frame)
        if ok_enc:
            with STATE.lock:
                STATE.first_frame_jpeg = buf.tobytes()
                STATE.latest_jpeg = STATE.first_frame_jpeg
                STATE.frame_w, STATE.frame_h = w, h

                STATE.zones = [{
                    "id": 1,
                    "name": "Vùng 1",
                    "points": [],
                    "road_width": None,
                    "road_length": None,
                    "speed_limit": None,
                    "calibration_status": "Chưa căn chỉnh"
                }]
            return True
    return False


# Estimate the horizon line from the convergence of the trapezoid's left and right sides
def compute_geometric_horizon(pts, frame_h):
    if len(pts) != 4:
        return 0.0
    
    # Points in order: 0: TL, 1: TR, 2: BR, 3: BL
    x1, y1 = pts[0]['x'], pts[0]['y']  # Top-Left
    x2, y2 = pts[1]['x'], pts[1]['y']  # Top-Right
    x3, y3 = pts[2]['x'], pts[2]['y']  # Bottom-Right
    x4, y4 = pts[3]['x'], pts[3]['y']  # Bottom-Left
    
    # Left edge line: connecting P1(TL) and P4(BL)
    dy1 = y4 - y1
    dx1 = x4 - x1
    
    # Right edge line: connecting P2(TR) and P3(BR)
    dy2 = y3 - y2
    dx2 = x3 - x2
    
    if dy1 == 0 or dy2 == 0:
        return -float(frame_h)
        
    m1 = dx1 / dy1
    c1 = x1 - m1 * y1
    
    m2 = dx2 / dy2
    c2 = x2 - m2 * y2
    
    # Intersection of left and right boundaries: m1 * y + c1 = m2 * y + c2 => y = (c2 - c1) / (m1 - m2)
    denom = m1 - m2
    if abs(denom) < 1e-5:
        # If lines are nearly parallel, default horizon is far above the image
        return -float(frame_h)
        
    y_horizon = (c2 - c1) / denom
    
    # Ensure horizon is above the highest point of the zone
    y_min = min(y1, y2, y3, y4)
    if y_horizon >= y_min:
        y_horizon = y_min - 100.0  # Safe default above the zone
        
    return y_horizon


# Background Video Processing Worker
def video_worker():
    global active_thread_id
    thread_id = threading.get_ident()
    active_thread_id = thread_id
    
    with STATE.lock:
        raw_src = STATE.source
        STATE.running = False
        STATE.paused_for_calibration = True
        STATE.start_signal.clear()
        STATE.error_msg = None

    resolved_src = resolve_video_path(raw_src)
    if not resolved_src or not os.path.exists(resolved_src):
        print(f"[Error] Cannot open video source: {raw_src}")
        with STATE.lock:
            STATE.error_msg = f"Cannot find video source file: {raw_src}"
        return
        
    print(f"[Worker Thread {thread_id}] Opening source: {resolved_src}")
    cap = cv2.VideoCapture(resolved_src)
    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {resolved_src}")
        with STATE.lock:
            STATE.error_msg = f"Cannot open video source: {raw_src}"
        return
        
    print(f"[Worker Thread {thread_id}] Waiting for user to configure points and click CHẠY...")
    # Wait until user clicks "CHẠY" (start_signal)
    while active_thread_id == thread_id:
        if STATE.start_signal.is_set():
            break
        time.sleep(0.1)
        
    if active_thread_id != thread_id:
        cap.release()
        return
        
    # Setup Perspective Mappers and Polygon Zones for all configured zones
    with STATE.lock:
        zones = STATE.zones
        w = STATE.frame_w
        h = STATE.frame_h
        STATE.running = True
        STATE.paused_for_calibration = False

    mappers = {}
    polygon_zones = {}
    speed_limits = {}
    
    # Pre-defined zone border colors: Orange, Cyan, Purple, Yellow
    zone_colors = [(0, 165, 255), (255, 182, 0), (239, 70, 217), (0, 255, 255)]

    auto_calibrate = {}
    calibration_horizon = {}
    calibration_samples = defaultdict(list)
    calibration_C = {}

    for idx, zone in enumerate(zones):
        zid = zone["id"]
        pts = zone["points"]
        speed_limits[zid] = float(zone.get("speed_limit", 60.0))
        
        road_w = zone.get("road_width")
        road_l = zone.get("road_length")
        
        if road_w is None or road_l is None or road_w == 0 or road_l == 0:
            auto_calibrate[zid] = True
            horizon = compute_geometric_horizon(pts, h)
            calibration_horizon[zid] = horizon
            # Default placeholders
            road_w = 10.0
            road_l = 30.0
            with STATE.lock:
                zone["calibration_status"] = "Đang chờ xe mẫu..."
            print(f"[Calibration] Zone {zid} needs Auto-Calibration. Geometric horizon estimated at y = {horizon:.1f}")
        else:
            auto_calibrate[zid] = False
            with STATE.lock:
                zone["calibration_status"] = "Đã cấu hình thủ công"
                
        mapper = Cam2WorldMapper()
        image_pts = [(p['x'], p['y']) for p in pts]
        world_pts = [(0, 0), (road_w, 0), (road_w, road_l), (0, road_l)]
        mapper.find_perspective_transform(image_pts, world_pts)
        
        mappers[zid] = mapper
        polygon_zones[zid] = sv.PolygonZone(polygon=np.array(image_pts, dtype=np.int32))

    # Load YOLO Model
    model_path = ARGS.model
    print(f"[Worker] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
        
    # Setup ByteTrack (Global tracker for unique consistent IDs)
    byte_track = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=0.35,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8
    )
    
    # Setup Supervision Annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=2
    )
    
    # Tracking variables (Global across all zones)
    coordinates = defaultdict(lambda: deque(maxlen=int(fps * 1.5)))
    smoothed_speeds = {}
    max_speeds_history = {}
    track_frame_counts = defaultdict(int)
    frame_idx = 0
    
    prev_t = time.time()
    fps_smooth = fps
    
    print("[Worker] 'CHẠY' signal received. Starting AI detection & speed estimation loop...")
    while active_thread_id == thread_id:
        ok, frame = cap.read()
        if not ok:
            if isinstance(resolved_src, str) and os.path.exists(resolved_src):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
                
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            
        frame_idx += 1
        
        # 1. Run YOLO object detection globally
        result = model(frame, imgsz=640, conf=0.35, classes=[2, 3, 5, 7], verbose=False)[0]
        global_detections = sv.Detections.from_ultralytics(result)
        
        # 2. Update track IDs globally
        global_detections = byte_track.update_with_detections(global_detections)
        
        annotated_frame = frame.copy()
        all_active_tids = set()
        
        # 3. Process each zone independently
        for idx, zone in enumerate(zones):
            zid = zone["id"]
            color = zone_colors[idx % len(zone_colors)]
            polygon_zone = polygon_zones[zid]
            speed_limit = speed_limits[zid]
            
            # Draw zone boundary polygon
            annotated_frame = cv2.polylines(annotated_frame, [polygon_zone.polygon], True, color, 2)
            
            # Filter detections within this specific zone
            mask = polygon_zone.trigger(detections=global_detections)
            zone_detections = global_detections[mask]
            
            if len(zone_detections) == 0:
                continue
                
            # Update Auto-Calibration using detected vehicle boxes in this zone
            if auto_calibrate.get(zid, False) and zone_detections.tracker_id is not None:
                updated_mapper = False
                for tracker_id, class_id, box in zip(zone_detections.tracker_id, zone_detections.class_id, zone_detections.xyxy):
                    cls_name = model.names[class_id]
                    std_widths = {
                        "car": 1.8,
                        "truck": 2.5,
                        "bus": 2.5,
                        "motorcycle": 0.8
                    }
                    if cls_name in std_widths:
                        w_std = std_widths[cls_name]
                        w_px = box[2] - box[0]
                        y_img = box[3]
                        y_horizon = calibration_horizon[zid]
                        
                        if y_img > y_horizon:
                            w_norm = w_px * (1.8 / w_std)
                            C_val = (y_img - y_horizon) / w_norm
                            calibration_samples[zid].append(C_val)
                            
                            # Keep sliding window of 500 samples
                            if len(calibration_samples[zid]) > 500:
                                calibration_samples[zid].pop(0)
                                
                            # Recalibrate at 3 samples, then every 10 samples thereafter
                            if len(calibration_samples[zid]) >= 3 and (len(calibration_samples[zid]) == 3 or len(calibration_samples[zid]) % 10 == 0):
                                C_avg = float(np.mean(calibration_samples[zid]))
                                calibration_C[zid] = C_avg
                                
                                # Estimate dimensions from geometric points & C
                                pts = zone["points"]
                                x1, y1 = pts[0]['x'], pts[0]['y'] # TL
                                x2, y2 = pts[1]['x'], pts[1]['y'] # TR
                                x3, y3 = pts[2]['x'], pts[2]['y'] # BR
                                x4, y4 = pts[3]['x'], pts[3]['y'] # BL
                                
                                y_bottom = (y3 + y4) / 2.0
                                y_top = (y1 + y2) / 2.0
                                
                                S_x_bottom = (1.8 * C_avg) / (y_bottom - y_horizon)
                                S_x_top = (1.8 * C_avg) / (y_top - y_horizon)
                                
                                W_bottom = abs(x3 - x4) * S_x_bottom
                                W_top = abs(x2 - x1) * S_x_top
                                road_w = (W_bottom + W_top) / 2.0
                                
                                f_cam = 1.2 * w
                                y_principal = h / 2.0
                                K_x = 1.8 * C_avg
                                K_y = K_x * np.sqrt(f_cam**2 + (y_principal - y_horizon)**2)
                                
                                road_l = K_y * abs(1.0 / (y_top - y_horizon) - 1.0 / (y_bottom - y_horizon))
                                
                                # Update shared state
                                with STATE.lock:
                                    for z_state in STATE.zones:
                                        if z_state["id"] == zid:
                                            z_state["road_width"] = float(round(road_w, 1))
                                            z_state["road_length"] = float(round(road_l, 1))
                                            z_state["calibration_status"] = f"Đã củng cố ({len(calibration_samples[zid])} mẫu)"
                                
                                # Rebuild and assign mapper
                                new_mapper = Cam2WorldMapper()
                                image_pts = [(p['x'], p['y']) for p in pts]
                                world_pts = [(0, 0), (road_w, 0), (road_w, road_l), (0, road_l)]
                                new_mapper.find_perspective_transform(image_pts, world_pts)
                                mappers[zid] = new_mapper
                                updated_mapper = True
                                
                if len(calibration_samples[zid]) < 3:
                    with STATE.lock:
                        for z_state in STATE.zones:
                            if z_state["id"] == zid:
                                z_state["calibration_status"] = f"Thu thập mẫu ({len(calibration_samples[zid])}/3)..."
                                
            # Get current mapper
            mapper = mappers[zid]
            
            # Get BOTTOM_CENTER road contact points
            points = zone_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            
            # Transform pixel points to meters
            points = mapper.map(points)
            
            labels = []
            
            if zone_detections.tracker_id is not None:
                for tracker_id, class_id, point in zip(zone_detections.tracker_id, zone_detections.class_id, points):
                    all_active_tids.add(tracker_id)
                    track_frame_counts[tracker_id] += 1
                    coordinates[tracker_id].append((frame_idx, point[0], point[1]))
                    
                    # Warm-up period: minimum 8 frames and 5 coordinate points
                    if track_frame_counts[tracker_id] < 8 or len(coordinates[tracker_id]) < 5:
                        labels.append(f"#{tracker_id} {model.names[class_id]} calculating...")
                        continue
                        
                    hist = list(coordinates[tracker_id])
                    step = min(5, len(hist) - 1)
                    prev_frame, prev_x, prev_y = hist[-1 - step]
                    curr_frame, curr_x, curr_y = hist[-1]
                    
                    dt_step = (curr_frame - prev_frame) / fps
                    if dt_step <= 0:
                        speed_inst = 0.0
                    else:
                        dist_m = np.sqrt((prev_x - curr_x) ** 2 + (prev_y - curr_y) ** 2)
                        speed_inst = (dist_m / dt_step) * 3.6
                        
                    if speed_inst > 200:
                        speed_inst = smoothed_speeds.get(tracker_id, 0.0)
                        
                    if tracker_id not in smoothed_speeds:
                        smooth_speed = speed_inst
                    else:
                        alpha = 0.25
                        smooth_speed = alpha * speed_inst + (1.0 - alpha) * smoothed_speeds[tracker_id]
                        
                    smoothed_speeds[tracker_id] = smooth_speed
                    max_speed = max(smooth_speed, max_speeds_history.get(tracker_id, 0.0))
                    max_speeds_history[tracker_id] = max_speed
                    is_violating = max_speed > speed_limit
                    
                    update_global_vehicle(tracker_id, model.names[class_id], smooth_speed, max_speed, is_violating)
                    
                    # Filter out noise speed under 5 km/h to prevent showing "0 km/h" for moving vehicles
                    if smooth_speed < 5.0:
                        labels.append(f"#{tracker_id} {model.names[class_id]} calculating...")
                    else:
                        labels.append(f"#{tracker_id} {model.names[class_id]} {int(round(smooth_speed))} km/h")
            
            # Annotate this zone's bounding boxes and labels
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=zone_detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=zone_detections, labels=labels
            )
            
        # Display active vehicle count & FPS overlay
        cv2.putText(annotated_frame, f"Active: {len(all_active_tids)} | FPS: {fps_smooth:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # Update metrics & FPS
        update_shared_metrics(all_active_tids, fps_smooth)
        
        # Encode frame to JPEG
        ok_enc, enc_buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok_enc:
            with STATE.lock:
                STATE.latest_jpeg = enc_buf.tobytes()
                
    cap.release()
    with STATE.lock:
        STATE.running = False
    print(f"[Worker Thread {thread_id}] Stopped.")


def restart_video_worker(new_source):
    global active_thread_id
    with STATE.lock:
        STATE.source = new_source
        STATE.start_signal.clear()
        STATE.running = False
        STATE.paused_for_calibration = True
        STATE.latest_jpeg = STATE.first_frame_jpeg
        STATE.vehicle_logs = []
        STATE.active_vehicles = 0
        STATE.avg_speed = 0.0
        STATE.overspeed_count = 0
        STATE.total_vehicles = 0
        
    active_thread_id = None
    time.sleep(0.3)
    
    worker = threading.Thread(target=video_worker, daemon=True)
    worker.start()


# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/first_frame.jpg")
def first_frame():
    with STATE.lock:
        data = STATE.first_frame_jpeg
    if data is None:
        return "No Video Loaded", 404
    return Response(data, mimetype="image/jpeg")


@app.route("/stream")
def stream():
    def generate():
        while True:
            with STATE.lock:
                data = STATE.latest_jpeg
            if data is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify(STATE.snapshot())


@app.route("/load_video", methods=["POST"])
def load_video():
    data = request.json
    if not data or "filename" not in data:
        return jsonify({"success": False, "error": "Chưa nhập tên file video"}), 400
        
    filename = data["filename"].strip()
    if not filename:
        return jsonify({"success": False, "error": "Tên file không được trống"}), 400
        
    resolved = resolve_video_path(filename)
    if not resolved or not os.path.exists(resolved):
        return jsonify({"success": False, "error": f"Không tìm thấy file '{filename}' trong thư mục 'files/'"}), 404

    # Synchronously read and encode first frame to guarantee 100% readiness
    ok = capture_first_frame_sync(resolved)
    if not ok:
        return jsonify({"success": False, "error": f"Không thể đọc khung hình đầu tiên từ file '{filename}'"}), 500

    restart_video_worker(resolved)
    return jsonify({
        "success": True, 
        "message": f"Đã tải thành công file {filename}.",
        "frame_w": STATE.frame_w,
        "frame_h": STATE.frame_h,
        "zones": STATE.zones
    })


@app.route("/start_processing", methods=["POST"])
def start_processing():
    data = request.json
    if not data or "zones" not in data:
        return jsonify({"success": False, "error": "Thiếu thông tin các vùng cấu hình"}), 400
        
    zones = data["zones"]
    for zone in zones:
        if "points" not in zone or len(zone["points"]) != 4:
            return jsonify({"success": False, "error": f"Vùng '{zone.get('name', 'không tên')}' phải có đủ 4 điểm"}), 400
            
    with STATE.lock:
        STATE.zones = zones
        STATE.start_signal.set()
        
    print(f"[Start Processing] {len(zones)} zones submitted. Unpausing worker!")
    return jsonify({"success": True})


@app.route("/stop_processing", methods=["POST"])
def stop_processing():
    global active_thread_id
    with STATE.lock:
        STATE.start_signal.clear()
        STATE.running = False
        STATE.paused_for_calibration = True
        STATE.latest_jpeg = STATE.first_frame_jpeg
        STATE.vehicle_logs = []
        STATE.active_vehicles = 0
        STATE.avg_speed = 0.0
        STATE.overspeed_count = 0
        STATE.total_vehicles = 0
        raw_src = STATE.source

    active_thread_id = None
    time.sleep(0.3)
    
    if raw_src:
        worker = threading.Thread(target=video_worker, daemon=True)
        worker.start()
        
    return jsonify({"success": True})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Không tìm thấy file tải lên"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Chưa chọn file"}), 400
        
    filename = file.filename
    target_dir = "files"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    filepath = os.path.join(target_dir, filename)
    file.save(filepath)
    
    # Synchronously read and encode first frame to guarantee 100% readiness
    ok = capture_first_frame_sync(filepath)
    if not ok:
        return jsonify({"success": False, "error": f"Không thể đọc khung hình đầu tiên từ file '{filename}'"}), 500

    restart_video_worker(filepath)
    return jsonify({
        "success": True, 
        "message": f"Đã tải lên và nạp thành công file {filename}.",
        "frame_w": STATE.frame_w,
        "frame_h": STATE.frame_h,
        "zones": STATE.zones
    })


if __name__ == "__main__":
    print(f"\n========================================================")
    print(f"  Vehicle Speed Detection Web Server Running")
    print(f"  URL: http://localhost:{ARGS.port}")
    print(f"========================================================\n")
    
    app.run(host=ARGS.host, port=ARGS.port, debug=False, threaded=True)