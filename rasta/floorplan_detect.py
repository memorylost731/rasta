#!/usr/bin/env python3
"""
Open-source floor plan recognition engine.
Replaces RasterScan API with local OpenCV + morphological analysis.

Detects: walls (line segments), rooms (polygons), doors (bounding boxes)
Output: JSON compatible with rasterscan_to_reactplanner.py

Pipeline:
  1. Load image (PNG/JPG/BMP) or PDF (first page)
  2. Adaptive threshold → binary mask
  3. Morphological ops → isolate walls
  4. Hough lines → merge collinear segments → wall segments
  5. Flood fill on inverse → room polygons
  6. Gap analysis on walls → door candidates
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


# ── Types ──────────────────────────────────────────────

class Wall(NamedTuple):
    x1: int; y1: int; x2: int; y2: int

class Room(NamedTuple):
    vertices: list  # [{id, x, y}, ...]

class Door(NamedTuple):
    x1: int; y1: int; x2: int; y2: int


# ── Image Loading ──────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image or first page of PDF as BGR numpy array."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(path, first_page=1, last_page=1, dpi=200)
            img = np.array(pages[0])
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except ImportError:
            raise SystemExit("pdf2image required for PDF input: pip install pdf2image")
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot load image: {path}")
    return img


# ── Pre-processing ─────────────────────────────────────

def preprocess(img: np.ndarray) -> np.ndarray:
    """Convert to binary mask where walls are white (255)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold handles varying illumination
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=12
    )

    # Remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    return binary


def extract_walls_mask(binary: np.ndarray, min_thickness: int = 3) -> np.ndarray:
    """Extract wall-like structures using morphological filtering."""
    h, w = binary.shape

    # Horizontal walls
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 20), 1))
    h_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Vertical walls
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 20)))
    v_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Combine
    walls = cv2.bitwise_or(h_walls, v_walls)

    # Thicken slightly to close small gaps
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_thickness, min_thickness))
    walls = cv2.dilate(walls, dilate_k, iterations=1)

    return walls


# ── Wall Detection ─────────────────────────────────────

def detect_walls(walls_mask: np.ndarray, min_length: int = 30) -> list[Wall]:
    """Detect wall segments using Hough Line Transform + merging."""
    lines = cv2.HoughLinesP(
        walls_mask, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=min_length, maxLineGap=15
    )
    if lines is None:
        return []

    raw = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        raw.append(Wall(x1, y1, x2, y2))

    # Snap to horizontal/vertical (within 5 degrees)
    snapped = []
    for w in raw:
        dx = abs(w.x2 - w.x1)
        dy = abs(w.y2 - w.y1)
        length = math.hypot(dx, dy)
        if length < min_length:
            continue

        if dx > 0 and dy / dx < math.tan(math.radians(5)):
            # Horizontal — snap y
            y_avg = (w.y1 + w.y2) // 2
            snapped.append(Wall(min(w.x1, w.x2), y_avg, max(w.x1, w.x2), y_avg))
        elif dy > 0 and dx / dy < math.tan(math.radians(5)):
            # Vertical — snap x
            x_avg = (w.x1 + w.x2) // 2
            snapped.append(Wall(x_avg, min(w.y1, w.y2), x_avg, max(w.y1, w.y2)))
        else:
            # Diagonal — keep as-is
            snapped.append(w)

    # Merge collinear overlapping segments
    merged = merge_collinear(snapped)
    return merged


def merge_collinear(walls: list[Wall], tol: int = 10) -> list[Wall]:
    """Merge wall segments that are collinear and overlapping/nearby."""
    if not walls:
        return []

    horizontal = []
    vertical = []
    diagonal = []

    for w in walls:
        if w.y1 == w.y2:
            horizontal.append(w)
        elif w.x1 == w.x2:
            vertical.append(w)
        else:
            diagonal.append(w)

    result = []

    # Merge horizontal lines at same Y
    h_groups: dict[int, list] = {}
    for w in horizontal:
        y = w.y1
        placed = False
        for gy in list(h_groups.keys()):
            if abs(y - gy) <= tol:
                h_groups[gy].append(w)
                placed = True
                break
        if not placed:
            h_groups[y] = [w]

    for y, group in h_groups.items():
        intervals = sorted([(min(w.x1, w.x2), max(w.x1, w.x2)) for w in group])
        merged_intervals = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged_intervals[-1]
            if start <= prev_end + tol:
                merged_intervals[-1] = (prev_start, max(prev_end, end))
            else:
                merged_intervals.append((start, end))
        for s, e in merged_intervals:
            result.append(Wall(s, y, e, y))

    # Merge vertical lines at same X
    v_groups: dict[int, list] = {}
    for w in vertical:
        x = w.x1
        placed = False
        for gx in list(v_groups.keys()):
            if abs(x - gx) <= tol:
                v_groups[gx].append(w)
                placed = True
                break
        if not placed:
            v_groups[x] = [w]

    for x, group in v_groups.items():
        intervals = sorted([(min(w.y1, w.y2), max(w.y1, w.y2)) for w in group])
        merged_intervals = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged_intervals[-1]
            if start <= prev_end + tol:
                merged_intervals[-1] = (prev_start, max(prev_end, end))
            else:
                merged_intervals.append((start, end))
        for s, e in merged_intervals:
            result.append(Wall(x, s, x, e))

    result.extend(diagonal)
    return result


# ── Room Detection ─────────────────────────────────────

def detect_rooms(walls_mask: np.ndarray, min_area: int = 2000) -> list[Room]:
    """Detect rooms as enclosed regions via contour analysis."""
    # Close gaps in walls
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(walls_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Invert: rooms become white regions
    inverted = cv2.bitwise_not(closed)

    # Remove border-touching regions (outside area)
    h, w = inverted.shape
    flood = inverted.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 0)
    # flood now has border regions removed

    # Find contours of remaining white regions
    contours, _ = cv2.findContours(flood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Simplify polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        vertices = []
        for j, pt in enumerate(approx):
            x, y = int(pt[0][0]), int(pt[0][1])
            vertices.append({"id": str(j), "x": x, "y": y})

        rooms.append(Room(vertices=vertices))

    # Sort by area descending
    rooms.sort(key=lambda r: -_polygon_area(r.vertices))
    return rooms


def _polygon_area(vertices: list[dict]) -> float:
    """Shoelace formula for polygon area."""
    n = len(vertices)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i]["x"] * vertices[j]["y"]
        area -= vertices[j]["x"] * vertices[i]["y"]
    return abs(area) / 2


# ── Door Detection ─────────────────────────────────────

def detect_doors(binary: np.ndarray, walls_mask: np.ndarray, walls: list[Wall],
                 min_gap: int = 15, max_gap: int = 120) -> list[Door]:
    """Detect doors as gaps in walls + arc patterns."""
    doors = []

    # Method 1: Find gaps in wall lines
    for w in walls:
        length = math.hypot(w.x2 - w.x1, w.y2 - w.y1)
        if length < 50:
            continue

        # Sample along the wall looking for gaps in the mask
        is_horizontal = (w.y1 == w.y2)
        if is_horizontal:
            y = w.y1
            x_start, x_end = min(w.x1, w.x2), max(w.x1, w.x2)
            in_gap = False
            gap_start = 0
            for x in range(x_start, x_end + 1):
                wy = max(0, min(y, walls_mask.shape[0] - 1))
                wx = max(0, min(x, walls_mask.shape[1] - 1))
                is_wall = walls_mask[wy, wx] > 0
                # Also check a few pixels around
                neighborhood = walls_mask[max(0, wy - 3):wy + 4, wx:wx + 1]
                is_wall = np.any(neighborhood > 0) if neighborhood.size > 0 else is_wall

                if not is_wall and not in_gap:
                    in_gap = True
                    gap_start = x
                elif is_wall and in_gap:
                    gap_len = x - gap_start
                    if min_gap <= gap_len <= max_gap:
                        pad = 10
                        doors.append(Door(gap_start - pad, y - pad, x + pad, y + pad))
                    in_gap = False

        else:  # vertical
            x = w.x1
            y_start, y_end = min(w.y1, w.y2), max(w.y1, w.y2)
            in_gap = False
            gap_start = 0
            for y in range(y_start, y_end + 1):
                wy = max(0, min(y, walls_mask.shape[0] - 1))
                wx = max(0, min(x, walls_mask.shape[1] - 1))
                neighborhood = walls_mask[wy:wy + 1, max(0, wx - 3):wx + 4]
                is_wall = np.any(neighborhood > 0) if neighborhood.size > 0 else False

                if not is_wall and not in_gap:
                    in_gap = True
                    gap_start = y
                elif is_wall and in_gap:
                    gap_len = y - gap_start
                    if min_gap <= gap_len <= max_gap:
                        pad = 10
                        doors.append(Door(x - pad, gap_start - pad, x + pad, y + pad))
                    in_gap = False

    # Method 2: Detect door arcs (quarter circles in the original image)
    arc_doors = _detect_door_arcs(binary, walls_mask)
    doors.extend(arc_doors)

    # Deduplicate overlapping doors
    doors = _dedup_doors(doors)
    return doors


def _detect_door_arcs(binary: np.ndarray, walls_mask: np.ndarray) -> list[Door]:
    """Detect quarter-circle door swing arcs."""
    # Subtract walls from binary to isolate non-wall marks
    non_wall = cv2.subtract(binary, walls_mask)

    # Clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_wall = cv2.morphologyEx(non_wall, cv2.MORPH_OPEN, kernel)

    # Find contours that look like arcs (thin curved shapes)
    contours, _ = cv2.findContours(non_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        arc_len = cv2.arcLength(cnt, False)

        # Door arcs: moderate area, high perimeter-to-area ratio
        if area < 100 or area > 5000:
            continue
        if arc_len < 30:
            continue

        # Ratio check: arcs are thin (high perimeter relative to area)
        if arc_len > 0 and area / arc_len < 8:
            x, y, w, h = cv2.boundingRect(cnt)
            # Door-sized bounding box
            if 15 < w < 150 and 15 < h < 150:
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect < 3:  # roughly square-ish
                    doors.append(Door(x, y, x + w, y + h))

    return doors


def _dedup_doors(doors: list[Door], iou_thresh: float = 0.3) -> list[Door]:
    """Remove overlapping door detections."""
    if not doors:
        return []

    keep = []
    used = set()
    for i, d in enumerate(doors):
        if i in used:
            continue
        merged = d
        for j, d2 in enumerate(doors):
            if j <= i or j in used:
                continue
            if _iou(merged, d2) > iou_thresh:
                # Merge: take union bbox
                merged = Door(
                    min(merged.x1, d2.x1), min(merged.y1, d2.y1),
                    max(merged.x2, d2.x2), max(merged.y2, d2.y2)
                )
                used.add(j)
        keep.append(merged)
    return keep


def _iou(a: Door, b: Door) -> float:
    """Intersection over union of two bboxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Main Pipeline ──────────────────────────────────────

def analyze_floorplan(image_path: str, scale: float = 1.0) -> dict:
    """Full pipeline: image → {walls, rooms, doors, area, perimeter}."""
    img = load_image(image_path)

    # Resize if very large (keep processing fast)
    h, w = img.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Pre-process
    binary = preprocess(img)
    walls_mask = extract_walls_mask(binary)

    # Detect
    wall_segments = detect_walls(walls_mask, min_length=max(30, min(h, w) // 20))
    rooms = detect_rooms(walls_mask, min_area=max(2000, (h * w) // 200))
    doors = detect_doors(binary, walls_mask, wall_segments)

    # Scale coordinates
    def sc(v):
        return int(round(v * scale))

    # Format output (RasterScan-compatible)
    walls_out = []
    for wall in wall_segments:
        walls_out.append({
            "position": [[sc(wall.x1), sc(wall.y1)], [sc(wall.x2), sc(wall.y2)]]
        })

    rooms_out = []
    for room in rooms:
        room_verts = []
        for v in room.vertices:
            room_verts.append({"id": v["id"], "x": sc(v["x"]), "y": sc(v["y"])})
        rooms_out.append(room_verts)

    doors_out = []
    for door in doors:
        doors_out.append({
            "bbox": [sc(door.x1), sc(door.y1), sc(door.x2), sc(door.y2)]
        })

    # Compute total area and perimeter from rooms
    total_area = sum(_polygon_area(r) for r in rooms_out) if rooms_out else h * w
    total_perimeter = sum(
        math.hypot(w["position"][1][0] - w["position"][0][0],
                    w["position"][1][1] - w["position"][0][1])
        for w in walls_out
    )

    return {
        "message": "Floor plan analysis successful (open-source engine)",
        "data": {
            "area": int(total_area * scale * scale),
            "perimeter": round(total_perimeter * scale, 1),
            "walls": walls_out,
            "rooms": rooms_out,
            "doors": doors_out,
        }
    }


# ── Debug Visualization ───────────────────────────────

def visualize(image_path: str, result: dict, output_path: str = None):
    """Draw detection results on image for debugging."""
    img = load_image(image_path)
    h, w = img.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)))

    data = result["data"]

    # Draw walls (green)
    for wall in data["walls"]:
        p = wall["position"]
        cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 255, 0), 2)

    # Draw rooms (blue polygons)
    for room in data["rooms"]:
        pts = np.array([[v["x"], v["y"]] for v in room], dtype=np.int32)
        cv2.polylines(img, [pts], True, (255, 150, 0), 2)

    # Draw doors (red bboxes)
    for door in data["doors"]:
        b = door["bbox"]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    out = output_path or str(Path(image_path).with_suffix(".detected.png"))
    cv2.imwrite(out, img)
    print(f"Visualization saved: {out}")


# ── CLI ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Open-source floor plan recognition")
    parser.add_argument("input", help="Floor plan image (PNG/JPG/BMP) or PDF")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scale factor")
    parser.add_argument("--visualize", "-v", action="store_true", help="Save debug visualization")
    args = parser.parse_args()

    result = analyze_floorplan(args.input, scale=args.scale)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    stats = result["data"]
    print(f"OK: wrote {args.output}")
    print(f"Stats: walls={len(stats['walls'])} rooms={len(stats['rooms'])} doors={len(stats['doors'])}")
    print(f"Area: {stats['area']}  Perimeter: {stats['perimeter']}")

    if args.visualize:
        visualize(args.input, result)


if __name__ == "__main__":
    main()
