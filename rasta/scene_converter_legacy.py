#!/usr/bin/env python3
import json, sys, math, hashlib

def stable_id(prefix: str, payload: str) -> str:
    h = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"{prefix}{h}"

def snap_key(x: float, y: float, tol: float) -> tuple[int, int]:
    # Deduplicate points by tolerance
    return (int(round(x / tol)), int(round(y / tol)))

def main(inp_path: str, out_path: str, scale: float = 1.0, tol: float = 0.01):
    raw = json.load(open(inp_path, "r", encoding="utf-8"))
    data = raw.get("data", raw)

    walls = data.get("walls", []) or []
    rooms = data.get("rooms", []) or []

    # --- storage ---
    vertices = {}   # id -> vertex
    lines = {}      # id -> line
    areas = {}      # id -> area

    # maps for dedupe
    vmap = {}       # snap_key -> vertex_id

    def get_vertex_id(x: float, y: float) -> str:
        k = snap_key(x, y, tol)
        if k in vmap:
            return vmap[k]
        vid = stable_id("v_", f"{k[0]}_{k[1]}")
        vmap[k] = vid
        vertices[vid] = {
            "id": vid,
            "type": "",
            "name": "Vertex",
            "misc": {},
            "selected": False,
            "properties": {},
            "visible": True,
            "prototype": "vertices",
            "x": x,
            "y": y,
            "lines": [],
            "areas": []
        }
        return vid

    def add_line(v1: str, v2: str) -> str:
        # avoid zero-length lines
        if v1 == v2:
            return ""
        # stable line id from endpoints
        key = "|".join(sorted([v1, v2]))
        lid = stable_id("l_", key)
        if lid in lines:
            return lid

        lines[lid] = {
            "id": lid,
            "type": "wall",         # IMPORTANT: must exist in catalog (your demo has 'wall')
            "name": "Wall",
            "misc": {},
            "selected": False,
            "properties": {
                "height": {"length": 300},
                "thickness": {"length": 20},
                "opacity": 1,
                "textureA": "bricks",
                "textureB": "bricks"
            },
            "visible": True,
            "prototype": "lines",
            "vertices": [v1, v2],
            "holes": []
        }

        # backrefs
        vertices[v1]["lines"].append(lid)
        vertices[v2]["lines"].append(lid)
        return lid

    # --- build walls -> lines ---
    for w in walls:
        pos = w.get("position")
        if not pos or len(pos) < 2:
            continue
        (x1, y1) = pos[0]
        (x2, y2) = pos[1]
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        v1 = get_vertex_id(x1, y1)
        v2 = get_vertex_id(x2, y2)
        add_line(v1, v2)

    # --- build rooms -> areas ---
    for room_idx, room in enumerate(rooms):
        # room is a list of points: [{id,x,y}, ...]
        if not room or len(room) < 3:
            continue
        pts = [(p.get("x"), p.get("y")) for p in room if "x" in p and "y" in p]
        if len(pts) < 3:
            continue

        area_vids = []
        for (x, y) in pts:
            vid = get_vertex_id(x * scale, y * scale)
            area_vids.append(vid)

        # React-Planner expects polygon order; we keep RasterScan order.
        aid = stable_id("a_", f"room_{room_idx}_" + "_".join(area_vids))
        areas[aid] = {
            "id": aid,
            "type": "area",     # IMPORTANT: must exist in catalog (your demo has 'area')
            "name": "Area",
            "misc": {},
            "selected": False,
            "properties": {
                "patternColor": "#F5F4F4",
                "thickness": {"length": 0},
                "texture": "none"
            },
            "visible": True,
            "prototype": "areas",
            "vertices": area_vids,
            "holes": []
        }

        # backrefs
        for vid in area_vids:
            vertices[vid]["areas"].append(aid)

    # --- compute scene bounds ---
    xs = [v["x"] for v in vertices.values()] or [0]
    ys = [v["y"] for v in vertices.values()] or [0]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 200
    width = int(math.ceil((maxx - minx) + pad * 2))
    height = int(math.ceil((maxy - miny) + pad * 2))

    # NOTE: we do NOT shift coordinates; we keep them as RasterScan gives them.
    # If you want shift-to-positive later, we can add a safe transform.

    scene = {
        "unit": "cm",
        "layers": {
            "layer-1": {
                "id": "layer-1",
                "altitude": 0,
                "order": 0,
                "opacity": 1,
                "name": "default",
                "visible": True,
                "vertices": vertices,
                "lines": lines,
                "holes": {},
                "areas": areas,
                "items": {},
                "selected": {"lines": [], "holes": [], "items": [], "areas": [], "vertices": []}
            }
        },
        "grids": {
            "h1": {"id": "h1", "type": "horizontal-streak", "properties": {"step": 20, "colors": ["#808080", "#ddd", "#ddd", "#ddd", "#ddd"]}},
            "v1": {"id": "v1", "type": "vertical-streak", "properties": {"step": 20, "colors": ["#808080", "#ddd", "#ddd", "#ddd", "#ddd"]}}
        },
        "selectedLayer": "layer-1",
        "groups": {},
        "width": width if width > 0 else 3000,
        "height": height if height > 0 else 2000,
        "meta": {},
        "guides": {"horizontal": {}, "vertical": {}, "circular": {}}
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)

    print(f"OK: wrote {out_path}")
    print(f"stats: vertices={len(vertices)} lines={len(lines)} areas={len(areas)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: rasterscan_to_reactplanner.py <rasterscan_raw.json> <out_scene.json> [scale]")
        sys.exit(2)
    scale = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
    main(sys.argv[1], sys.argv[2], scale=scale, tol=0.01)
