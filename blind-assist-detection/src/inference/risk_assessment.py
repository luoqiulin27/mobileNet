from __future__ import annotations

from dataclasses import asdict, dataclass


HIGH_RISK_LABELS = {
    "person",
    "pedestrian",
    "vehicle",
    "rider",
    "animal",
    "stairs",
    "pole",
    "obstacle",
    "indoor_obstacle",
    "bike_rack",
}

MEDIUM_RISK_LABELS = {
    "chair",
    "table",
    "sofa",
    "door",
    "cabinet",
    "box_bag",
}


@dataclass
class RiskAssessment:
    label: str
    score: float
    box: tuple[float, float, float, float]
    area_ratio: float
    center_offset: float
    horizontal_zone: str
    distance_level: str
    risk_level: str
    risk_score: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["box"] = [round(v, 2) for v in self.box]
        payload["score"] = round(self.score, 4)
        payload["area_ratio"] = round(self.area_ratio, 4)
        payload["center_offset"] = round(self.center_offset, 4)
        payload["risk_score"] = round(self.risk_score, 4)
        return payload


def _horizontal_zone(cx_ratio: float) -> str:
    if 0.4 <= cx_ratio <= 0.6:
        return "center"
    if 0.25 <= cx_ratio < 0.4 or 0.6 < cx_ratio <= 0.75:
        return "near-center"
    return "side"


def _distance_level(area_ratio: float, bottom_ratio: float) -> str:
    if area_ratio >= 0.12 or bottom_ratio >= 0.88:
        return "near"
    if area_ratio >= 0.04 or bottom_ratio >= 0.72:
        return "medium"
    return "far"


def _base_label_weight(label: str) -> float:
    if label in HIGH_RISK_LABELS:
        return 1.0
    if label in MEDIUM_RISK_LABELS:
        return 0.65
    return 0.45


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def assess_detection_risk(detection, image_size: tuple[int, int]) -> RiskAssessment:
    image_width, image_height = image_size
    x1, y1, x2, y2 = detection.box
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    image_area = max(float(image_width * image_height), 1.0)
    area_ratio = max(min((width * height) / image_area, 1.0), 0.0)

    cx = (x1 + x2) / 2
    cy_bottom = y2
    cx_ratio = min(max(cx / max(float(image_width), 1.0), 0.0), 1.0)
    bottom_ratio = min(max(cy_bottom / max(float(image_height), 1.0), 0.0), 1.0)

    zone = _horizontal_zone(cx_ratio)
    distance = _distance_level(area_ratio, bottom_ratio)

    zone_weight = {"center": 1.0, "near-center": 0.75, "side": 0.45}[zone]
    distance_weight = {"near": 1.0, "medium": 0.7, "far": 0.4}[distance]
    confidence_weight = min(max(detection.score, 0.0), 1.0)
    label_weight = _base_label_weight(detection.label)
    center_offset = abs(cx_ratio - 0.5) * 2

    risk_score = 0.4 * label_weight + 0.25 * zone_weight + 0.25 * distance_weight + 0.1 * confidence_weight
    level = _risk_level(risk_score)
    return RiskAssessment(
        label=detection.label,
        score=detection.score,
        box=detection.box,
        area_ratio=area_ratio,
        center_offset=center_offset,
        horizontal_zone=zone,
        distance_level=distance,
        risk_level=level,
        risk_score=risk_score,
    )


def summarize_risks(detections: list, image_size: tuple[int, int]) -> dict:
    items = [assess_detection_risk(item, image_size) for item in detections]
    items.sort(key=lambda item: item.risk_score, reverse=True)
    counts = {"high": 0, "medium": 0, "low": 0}
    for item in items:
        counts[item.risk_level] += 1
    return {
        "items": [item.to_dict() for item in items],
        "summary": {
            "highest_risk_level": items[0].risk_level if items else "low",
            "highest_risk_score": items[0].risk_score if items else 0.0,
            "counts": counts,
        },
    }
