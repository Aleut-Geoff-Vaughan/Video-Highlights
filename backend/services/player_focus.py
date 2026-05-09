from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


def resolve_player_roi_box(
    raw_roi: Optional[Dict[str, object]],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(raw_roi, dict) or frame_width <= 0 or frame_height <= 0:
        return None

    normalized = bool(raw_roi.get("normalized", False))

    if all(key in raw_roi for key in ("x1_norm", "y1_norm", "x2_norm", "y2_norm")):
        normalized = True
        x1 = float(raw_roi.get("x1_norm", 0.0))
        y1 = float(raw_roi.get("y1_norm", 0.0))
        x2 = float(raw_roi.get("x2_norm", 1.0))
        y2 = float(raw_roi.get("y2_norm", 1.0))
    elif all(key in raw_roi for key in ("x", "y", "w", "h")):
        x = float(raw_roi.get("x", 0.0))
        y = float(raw_roi.get("y", 0.0))
        w = float(raw_roi.get("w", 0.0))
        h = float(raw_roi.get("h", 0.0))
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
    else:
        return None

    if normalized:
        x1 *= frame_width
        x2 *= frame_width
        y1 *= frame_height
        y2 *= frame_height

    x1 = min(max(0.0, x1), float(frame_width))
    x2 = min(max(0.0, x2), float(frame_width))
    y1 = min(max(0.0, y1), float(frame_height))
    y2 = min(max(0.0, y2), float(frame_height))

    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return x1, y1, x2, y2


def box_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _box_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _point_time(point: object) -> float:
    return float(getattr(point, "t", 0.0))


def _point_center(point: object) -> Optional[Tuple[float, float]]:
    center = getattr(point, "xy", None)
    if center is None:
        return None
    return float(center[0]), float(center[1])


def _point_bbox(point: object) -> Optional[Tuple[float, float, float, float]]:
    bbox = getattr(point, "bbox", None)
    if bbox is None:
        return None
    try:
        return tuple(float(value) for value in bbox)
    except Exception:
        return None


def choose_target_track_id(
    tracks: Dict[int, Iterable[object]],
    user_box: Tuple[float, float, float, float],
    window_t: float = 3.0,
) -> Optional[int]:
    ux = (user_box[0] + user_box[2]) / 2.0
    uy = (user_box[1] + user_box[3]) / 2.0

    best_id: Optional[int] = None
    best_score: Optional[Tuple[float, float, float, float, int]] = None

    for track_id, raw_traj in tracks.items():
        early = [point for point in raw_traj if _point_time(point) <= window_t]
        if not early:
            continue

        ious = []
        inside_hits = 0
        dists = []
        for point in early:
            center = _point_center(point)
            if center is None:
                continue
            cx, cy = center
            dists.append(((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5)
            if user_box[0] <= cx <= user_box[2] and user_box[1] <= cy <= user_box[3]:
                inside_hits += 1

            bbox = _point_bbox(point)
            if bbox is not None:
                try:
                    ious.append(box_iou(user_box, bbox))
                except Exception:
                    pass

        if not dists:
            continue

        max_iou = max(ious) if ious else 0.0
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        inside_ratio = inside_hits / max(1, len(early))
        mean_dist = sum(dists) / len(dists)
        score = (
            max_iou,
            mean_iou,
            inside_ratio,
            -mean_dist,
            len(early),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_id = int(track_id)

    return best_id


def stitch_target_track(
    tracks: Dict[int, Iterable[object]],
    target_track_id: int,
    *,
    max_gap_seconds: float = 1.0,
    overlap_tolerance_seconds: float = 0.35,
    max_center_distance: float = 140.0,
    min_link_iou: float = 0.02,
) -> Tuple[List[int], List[object]]:
    if int(target_track_id) not in tracks:
        return [], []

    ordered_tracks: Dict[int, List[object]] = {}
    for track_id, raw_traj in tracks.items():
        points = list(raw_traj or [])
        if not points:
            continue
        ordered_tracks[int(track_id)] = sorted(points, key=_point_time)

    if int(target_track_id) not in ordered_tracks:
        return [], []

    stitched_ids = [int(target_track_id)]
    used_ids = {int(target_track_id)}
    stitched = list(ordered_tracks[int(target_track_id)])

    while stitched:
        tail = stitched[-1]
        tail_t = _point_time(tail)
        tail_center = _point_center(tail)
        if tail_center is None:
            break
        tail_bbox = _point_bbox(tail)
        tail_area = _box_area(tail_bbox) if tail_bbox is not None else 0.0
        dynamic_max_distance = max(
            float(max_center_distance),
            (tail_bbox[2] - tail_bbox[0]) * 1.75 if tail_bbox is not None else 0.0,
        )

        best_candidate_id: Optional[int] = None
        best_candidate_points: List[object] = []
        best_candidate_score: Optional[Tuple[float, float, float, float, float, int]] = None

        for track_id, candidate_points in ordered_tracks.items():
            if track_id in used_ids:
                continue

            window_points = [
                point
                for point in candidate_points
                if (tail_t - float(overlap_tolerance_seconds)) <= _point_time(point) <= (tail_t + float(max_gap_seconds))
            ]
            if not window_points:
                continue

            anchor = min(window_points, key=_point_time)
            anchor_center = _point_center(anchor)
            if anchor_center is None:
                continue

            gap = _point_time(anchor) - tail_t
            center_distance = ((anchor_center[0] - tail_center[0]) ** 2 + (anchor_center[1] - tail_center[1]) ** 2) ** 0.5

            anchor_bbox = _point_bbox(anchor)
            link_iou = box_iou(tail_bbox, anchor_bbox) if tail_bbox is not None and anchor_bbox is not None else 0.0
            anchor_area = _box_area(anchor_bbox) if anchor_bbox is not None else 0.0
            size_similarity = 0.0
            if tail_area > 0.0 and anchor_area > 0.0:
                size_similarity = min(tail_area, anchor_area) / max(tail_area, anchor_area)

            if center_distance > dynamic_max_distance and link_iou < float(min_link_iou):
                continue

            append_points = [point for point in candidate_points if _point_time(point) > (tail_t + 1e-6)]
            if not append_points:
                continue

            score = (
                link_iou,
                size_similarity,
                -max(0.0, gap),
                -center_distance,
                -abs(gap),
                len(append_points),
            )
            if best_candidate_score is None or score > best_candidate_score:
                best_candidate_score = score
                best_candidate_id = track_id
                best_candidate_points = append_points

        if best_candidate_id is None or not best_candidate_points:
            break

        used_ids.add(best_candidate_id)
        stitched_ids.append(best_candidate_id)
        stitched.extend(best_candidate_points)

    stitched.sort(key=_point_time)
    return stitched_ids, stitched
