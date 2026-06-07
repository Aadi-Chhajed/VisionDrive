import cv2
import numpy as np
import time

# CONFIGURATION
VIDEO_PATH = "driving_video.mp4"
YOLO_CONFIG = "yolov3.cfg"
YOLO_WEIGHTS = "yolov3.weights"
YOLO_NAMES = "coco.names"
print("COMPLETE ADAS SYSTEM - Starting...")

#  LOAD YOLO
print("\n1. Loading YOLO model...")
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(YOLO_NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("   ✓ YOLO loaded successfully!")

# OPEN VIDEO
print("\n2. Opening video source...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("   ✗ Error: Cannot open video source!")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"   ✓ Video opened! FPS: {fps}")


#  LANE DETECTION FUNCTIONS
def detect_lanes(frame):
    """Detect lane lines using edge detection"""
    height, width = frame.shape[:2]

    # Region of interest for lanes
    roi_vertices = np.array([[
        [0, height],
        [width * 0.45, height * 0.6],
        [width * 0.55, height * 0.6],
        [width, height]
    ]], dtype=np.int32)

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Apply ROI mask
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 50,
                            minLineLength=100, maxLineGap=50)

    return lines, roi_vertices


def draw_lanes(frame, lines):
    """Draw detected lanes on frame"""
    if lines is None:
        return frame, None, None

    height, width = frame.shape[:2]
    left_lines = []
    right_lines = []

    # Separate left and right lanes
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) < 0.5:  # Skip horizontal lines
            continue

        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])

    # Average lines
    def average_lines(lines):
        if not lines:
            return None
        x_coords = []
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        if len(x_coords) > 0:
            return np.polyfit(y_coords, x_coords, 1)
        return None

    left_poly = average_lines(left_lines)
    right_poly = average_lines(right_lines)

    # Draw lane lines
    lane_overlay = np.zeros_like(frame)
    y1 = height
    y2 = int(height * 0.6)

    left_lane = None
    right_lane = None

    if left_poly is not None:
        x1 = int(left_poly[0] * y1 + left_poly[1])
        x2 = int(left_poly[0] * y2 + left_poly[1])
        left_lane = [x1, y1, x2, y2]
        cv2.line(lane_overlay, (x1, y1), (x2, y2), (255, 0, 0), 8)

    if right_poly is not None:
        x1 = int(right_poly[0] * y1 + right_poly[1])
        x2 = int(right_poly[0] * y2 + right_poly[1])
        right_lane = [x1, y1, x2, y2]
        cv2.line(lane_overlay, (x1, y1), (x2, y2), (255, 0, 0), 8)

    # Fill lane area
    if left_lane and right_lane:
        pts = np.array([
            [left_lane[0], left_lane[1]],
            [left_lane[2], left_lane[3]],
            [right_lane[2], right_lane[3]],
            [right_lane[0], right_lane[1]]
        ], dtype=np.int32)
        cv2.fillPoly(lane_overlay, [pts], (0, 255, 0))

    result = cv2.addWeighted(frame, 1, lane_overlay, 0.3, 0)
    return result, left_lane, right_lane


# ============ VEHICLE DETECTION ============
def detect_vehicles(frame):
    """Detect vehicles using YOLO"""
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in ['car', 'truck', 'bus', 'motorbike']:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    vehicles = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            vehicles.append({
                'bbox': (x, y, w, h),
                'confidence': confidences[i],
                'class': classes[class_ids[i]],
                'center': (x + w // 2, y + h // 2)
            })

    return vehicles


# ============ ROI MANAGEMENT ============
def create_vehicle_roi(frame_height, frame_width):
    """Create a fixed ROI for vehicle monitoring"""
    roi_vertices = np.array([[
        [int(frame_width * 0.2), int(frame_height * 0.5)],
        [int(frame_width * 0.8), int(frame_height * 0.5)],
        [int(frame_width * 0.8), frame_height],
        [int(frame_width * 0.2), frame_height]
    ]], dtype=np.int32)
    return roi_vertices


def is_vehicle_in_roi(vehicle, roi_vertices):
    """Check if vehicle center is within ROI"""
    center_x, center_y = vehicle['center']
    point = (center_x, center_y)
    result = cv2.pointPolygonTest(roi_vertices[0], point, False)
    return result >= 0


def filter_vehicles_in_roi(vehicles, roi_vertices):
    """Filter vehicles that are within ROI"""
    vehicles_in_roi = []
    for vehicle in vehicles:
        if is_vehicle_in_roi(vehicle, roi_vertices):
            vehicles_in_roi.append(vehicle)
    return vehicles_in_roi


def track_unique_vehicles(vehicles_in_roi, tracked_ids, id_counter, distance_threshold=50):
    """Track unique vehicles entering ROI and assign IDs"""
    new_tracked_ids = {}

    for vehicle in vehicles_in_roi:
        center = vehicle['center']
        matched = False

        # Check if this vehicle matches any existing tracked vehicle
        for vid, prev_center in tracked_ids.items():
            distance = np.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
            if distance < distance_threshold:
                new_tracked_ids[vid] = center
                vehicle['id'] = vid
                matched = True
                break

        # If no match found, assign new ID
        if not matched:
            new_id = id_counter[0]
            id_counter[0] += 1
            new_tracked_ids[new_id] = center
            vehicle['id'] = new_id

    return new_tracked_ids


# ============ COLLISION WARNING ============
def assess_collision_risk(vehicles, frame_height):
    """Assess collision risk based on vehicle position (for vehicles in ROI)"""
    if not vehicles:
        return 'safe', []

    risky_vehicles = []
    max_risk = 'safe'

    for vehicle in vehicles:
        x, y, w, h = vehicle['bbox']
        bottom_y = y + h
        area = w * h

        # Risk thresholds
        critical_y = frame_height * 0.85
        warning_y = frame_height * 0.70
        critical_area = (frame_height * 1280) * 0.12

        risk = 'safe'
        if bottom_y >= critical_y or area >= critical_area:
            risk = 'critical'
        elif bottom_y >= warning_y:
            risk = 'warning'

        if risk != 'safe':
            risky_vehicles.append({'vehicle': vehicle, 'risk': risk})
            if risk == 'critical':
                max_risk = 'critical'
            elif risk == 'warning' and max_risk != 'critical':
                max_risk = 'warning'

    return max_risk, risky_vehicles


# ============ LANE DEPARTURE WARNING ============
def check_lane_departure(left_lane, right_lane, frame_width):
    """Check if vehicle is departing from lane center"""
    if left_lane is None or right_lane is None:
        return 'unknown', 0, None

    vehicle_center = frame_width // 2
    lane_center = (left_lane[0] + right_lane[0]) // 2
    deviation = abs(vehicle_center - lane_center)

    direction = 'left' if vehicle_center < lane_center else 'right'

    warning_threshold = frame_width * 0.1
    critical_threshold = frame_width * 0.15

    if deviation >= critical_threshold:
        return 'critical', deviation, direction
    elif deviation >= warning_threshold:
        return 'warning', deviation, direction
    else:
        return 'safe', deviation, direction


# ============ MAIN LOOP ============
print("\n3. Starting ADAS system...")
print("\nControls:")
print("   - Press 'Q' to quit")
print("   - Press 'P' to pause")
print("   - Press 'S' to save screenshot")
print("\n" + "=" * 70 + "\n")

# Initialize variables BEFORE the loop
vehicle_count = 0
frame_count = 0
paused = False
vehicles = []
vehicles_in_roi = []
collision_risk = 'safe'
risky_vehicles = []
lane_status = 'unknown'
deviation = 0
direction = None
tracked_vehicle_ids = {}
unique_vehicle_counter = [0]  # Using list to make it mutable in function
total_unique_vehicles = 0
roi_vertices = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video or error reading frame")
            break

        frame_count += 1
        frame = cv2.resize(frame, (1280, 720))
        height, width = frame.shape[:2]

        # Create ROI for vehicle monitoring (once)
        if roi_vertices is None:
            roi_vertices = create_vehicle_roi(height, width)

        # Process every frame
        start_time = time.time()

        # 1. Detect lanes
        lines, roi_vertices = detect_lanes(frame)
        frame, left_lane, right_lane = draw_lanes(frame, lines)

        # 2. Detect vehicles (every 2 frames for performance)
        if frame_count % 2 == 0:
            vehicles = detect_vehicles(frame)
            vehicle_count = len(vehicles)

            # Filter vehicles within ROI
            vehicles_in_roi = filter_vehicles_in_roi(vehicles, roi_vertices)

            # Track unique vehicles
            tracked_vehicle_ids = track_unique_vehicles(
                vehicles_in_roi, tracked_vehicle_ids, unique_vehicle_counter
            )
            total_unique_vehicles = unique_vehicle_counter[0]

            # 3. Assess collision risk (only for vehicles in ROI)
            collision_risk, risky_vehicles = assess_collision_risk(vehicles_in_roi, height)

            # 4. Check lane departure
            lane_status, deviation, direction = check_lane_departure(
                left_lane, right_lane, width)

        # Draw vehicles (use the last detected vehicles)
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']

            # Check if this vehicle is in ROI
            in_roi = is_vehicle_in_roi(vehicle, roi_vertices)

            # Color based on risk (only for vehicles in ROI)
            if in_roi:
                is_risky = any(rv['vehicle'] == vehicle for rv in risky_vehicles)
                if is_risky:
                    risk = next(rv['risk'] for rv in risky_vehicles
                                if rv['vehicle'] == vehicle)
                    color = (0, 0, 255) if risk == 'critical' else (0, 165, 255)
                else:
                    color = (0, 255, 0)

                # Draw vehicle ID if tracked
                if 'id' in vehicle:
                    cv2.putText(frame, f"ID:{vehicle['id']}", (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                color = (128, 128, 128)  # Gray for vehicles outside ROI

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{vehicle['class']}: {vehicle['confidence']:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw ROI boundary
        cv2.polylines(frame, roi_vertices, True, (255, 255, 0), 3)
        cv2.putText(frame, "ROI", (roi_vertices[0][0][0] + 10, roi_vertices[0][0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Calculate FPS
        process_fps = 1.0 / (time.time() - start_time)

        # ============ DRAW UI ============
        # Info panel
        cv2.rectangle(frame, (0, 0), (450, 150), (50, 50, 50), -1)
        cv2.putText(frame, "ADAS SYSTEM ACTIVE", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"All Vehicles: {vehicle_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Vehicles in ROI: {len(vehicles_in_roi)}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Unique Count: {total_unique_vehicles}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {process_fps:.1f}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Collision warning
        if collision_risk == 'critical':
            cv2.rectangle(frame, (0, 130), (width, 210), (0, 0, 255), -1)
            cv2.putText(frame, "COLLISION ALERT!", (width // 2 - 200, 175),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        elif collision_risk == 'warning':
            cv2.rectangle(frame, (0, 130), (width, 190), (0, 165, 255), -1)
            cv2.putText(frame, "CAUTION: Vehicle Close", (50, 165),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Lane departure warning
        if lane_status == 'critical':
            cv2.rectangle(frame, (0, height - 90), (width, height - 30),
                          (0, 0, 255), -1)
            cv2.putText(frame, f"LANE DEPARTURE! Drifting {direction.upper()}",
                        (50, height - 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        elif lane_status == 'warning':
            cv2.rectangle(frame, (0, height - 70), (width, height - 30),
                          (0, 165, 255), -1)
            cv2.putText(frame, f"Lane Departure: Drifting {direction}",
                        (50, height - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Status indicators
        lane_detected = left_lane is not None and right_lane is not None
        cv2.rectangle(frame, (width - 220, 10), (width - 10, 80), (50, 50, 50), -1)

        lane_color = (0, 255, 0) if lane_detected else (0, 0, 255)
        cv2.putText(frame, f"{'✓' if lane_detected else '✗'} Lane Detection",
                    (width - 210, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_color, 1)

        vehicle_color = (0, 255, 0) if vehicle_count > 0 else (200, 200, 200)
        cv2.putText(frame, f"{'✓' if vehicle_count > 0 else '○'} Vehicle Detection",
                    (width - 210, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vehicle_color, 1)

        # Display frame
        cv2.imshow('Complete ADAS System - Press Q to quit', frame)

    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("⏸ Paused" if paused else "▶ Resumed")
    elif key == ord('s'):
        filename = f"adas_screenshot_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 70)
print("ADAS System stopped successfully!")
print("=" * 70)