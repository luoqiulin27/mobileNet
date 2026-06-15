from __future__ import annotations

import base64
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import Detection, DetectionPredictor
from src.inference.scene_classifier import SceneClassifier, ScenePrediction


@dataclass(frozen=True)
class DemoMode:
    key: str
    label: str
    config_path: Path
    checkpoint_candidates: tuple[Path, ...]
    default_conf_threshold: float
    description: str


@dataclass(frozen=True)
class SelectOption:
    key: str
    label: str
    description: str


@dataclass
class SceneDecision:
    requested_mode: str
    selected_mode: str
    selected_label: str
    automatic: bool
    fallback_used: bool
    reason: str
    scene_label: str | None = None
    scene_confidence: float | None = None
    scene_probabilities: dict[str, float] | None = None
    detector_scores: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "selected_label": self.selected_label,
            "automatic": self.automatic,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "scene_label": self.scene_label,
            "scene_confidence": self.scene_confidence,
            "scene_probabilities": self.scene_probabilities or {},
            "detector_scores": self.detector_scores or {},
        }


MODES: dict[str, DemoMode] = {
    "outdoor": DemoMode(
        key="outdoor",
        label="室外 SANPO 障碍物识别",
        config_path=PROJECT_ROOT / "src" / "configs" / "ssd_default.yaml",
        checkpoint_candidates=(
            PROJECT_ROOT / "outputs" / "checkpoints" / "best.pth",
            PROJECT_ROOT / "outputs" / "checkpoints" / "last.pth",
        ),
        default_conf_threshold=0.05,
        description="适合道路、校园、户外通行场景，类别包含人、车、骑行者、动物、楼梯、杆体、自行车架和通用障碍物。",
    ),
    "indoor": DemoMode(
        key="indoor",
        label="室内 SUNRGBD 障碍物识别",
        config_path=PROJECT_ROOT / "src" / "configs" / "ssd_sunrgbd_indoor.yaml",
        checkpoint_candidates=(
            PROJECT_ROOT / "outputs" / "runs" / "sunrgbd_indoor_12class" / "checkpoints" / "best.pth",
            PROJECT_ROOT / "outputs" / "runs" / "sunrgbd_indoor_12class" / "checkpoints" / "last.pth",
        ),
        default_conf_threshold=0.05,
        description="适合办公室、卧室、教室、走廊等室内场景，类别包含座椅、桌子、沙发、床、门、储物柜、箱包和室内障碍物。",
    ),
}

SELECT_OPTIONS: tuple[SelectOption, ...] = (
    SelectOption("auto", "自动判断室内/室外", "先识别场景，再自动调用对应障碍物检测模型；低置信度时双模型兜底。"),
    SelectOption("outdoor", MODES["outdoor"].label, MODES["outdoor"].description),
    SelectOption("indoor", MODES["indoor"].label, MODES["indoor"].description),
)

SCENE_CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "runs" / "scene_indoor_outdoor" / "checkpoints" / "best.pth",
    PROJECT_ROOT / "outputs" / "runs" / "scene_indoor_outdoor" / "checkpoints" / "last.pth",
)
SCENE_CONFIDENCE_THRESHOLD = 0.75

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "web_demo" / "templates"))
predictor_cache: dict[tuple[str, str, float], DetectionPredictor] = {}
scene_classifier_cache: dict[tuple[str, float], SceneClassifier] = {}


def get_requested_mode_key() -> str:
    key = request.form.get("mode") or request.args.get("mode") or "auto"
    return key if key in {"auto", *MODES.keys()} else "auto"


def get_threshold_mode(requested_key: str) -> DemoMode:
    if requested_key in MODES:
        return MODES[requested_key]
    return MODES["outdoor"]


def resolve_checkpoint_path(mode: DemoMode) -> Path | None:
    for path in mode.checkpoint_candidates:
        if path.exists():
            return path
    return None


def resolve_scene_checkpoint_path() -> Path | None:
    for path in SCENE_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return None


def get_predictor(mode: DemoMode) -> DetectionPredictor:
    checkpoint_path = resolve_checkpoint_path(mode)
    if checkpoint_path is None:
        raise FileNotFoundError(f"{mode.label} 还没有可用 checkpoint。")

    cache_key = (mode.key, str(checkpoint_path), checkpoint_path.stat().st_mtime)
    if cache_key not in predictor_cache:
        for old_key in list(predictor_cache):
            if old_key[0] == mode.key:
                predictor_cache.pop(old_key, None)
        predictor_cache[cache_key] = DetectionPredictor(mode.config_path, checkpoint_path, gpu=0)
    return predictor_cache[cache_key]


def get_scene_classifier() -> SceneClassifier | None:
    checkpoint_path = resolve_scene_checkpoint_path()
    if checkpoint_path is None:
        return None

    cache_key = (str(checkpoint_path), checkpoint_path.stat().st_mtime)
    if cache_key not in scene_classifier_cache:
        scene_classifier_cache.clear()
        scene_classifier_cache[cache_key] = SceneClassifier(checkpoint_path, gpu=0)
    return scene_classifier_cache[cache_key]


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_image_from_request() -> Image.Image:
    upload = request.files.get("image")
    if not upload or not upload.filename:
        raise ValueError("请先选择一张图片。")
    return Image.open(upload.stream).convert("RGB")


def parse_conf_threshold(mode: DemoMode) -> float:
    raw = request.form.get("conf_threshold", str(mode.default_conf_threshold))
    try:
        value = float(raw)
    except ValueError:
        value = mode.default_conf_threshold
    return min(max(value, 0.01), 0.9)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def mode_status(mode: DemoMode) -> dict:
    checkpoint_path = resolve_checkpoint_path(mode)
    metrics_path = (
        PROJECT_ROOT / "outputs" / "metrics" / "eval_sunrgbd_val_best.json"
        if mode.key == "indoor"
        else PROJECT_ROOT / "outputs" / "metrics" / "eval_sanpo_val_best.json"
    )
    return {
        "key": mode.key,
        "label": mode.label,
        "description": mode.description,
        "checkpoint_ready": checkpoint_path is not None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_name": checkpoint_path.name if checkpoint_path else "未找到",
        "metrics": read_json(metrics_path),
    }


def scene_status() -> dict:
    checkpoint_path = resolve_scene_checkpoint_path()
    history_path = PROJECT_ROOT / "outputs" / "runs" / "scene_indoor_outdoor" / "metrics" / "history.json"
    history = read_json(history_path)
    best_acc = None
    if isinstance(history, list) and history:
        best_acc = max(float(row.get("val_acc", 0.0)) for row in history)
    return {
        "checkpoint_ready": checkpoint_path is not None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_name": checkpoint_path.name if checkpoint_path else "未找到",
        "confidence_threshold": SCENE_CONFIDENCE_THRESHOLD,
        "best_val_acc": best_acc,
    }


def detection_score(detections: list[Detection]) -> float:
    if not detections:
        return 0.0
    top = detections[:5]
    avg_confidence = sum(item.score for item in top) / len(top)
    count_score = min(len(detections), 5) / 5
    return avg_confidence * 0.8 + count_score * 0.2


def run_detector(mode_key: str, image: Image.Image, conf_threshold: float) -> tuple[DetectionPredictor, list[Detection]]:
    mode = MODES[mode_key]
    predictor = get_predictor(mode)
    detections = predictor.predict(image, conf_threshold=conf_threshold)
    return predictor, detections


def automatic_detect(
    image: Image.Image,
    conf_threshold: float,
) -> tuple[DetectionPredictor, list[Detection], SceneDecision]:
    scene_classifier = get_scene_classifier()
    scene_prediction: ScenePrediction | None = None

    if scene_classifier is not None:
        scene_prediction = scene_classifier.predict(image)
        if (
            scene_prediction.label in MODES
            and scene_prediction.confidence >= SCENE_CONFIDENCE_THRESHOLD
        ):
            predictor, detections = run_detector(scene_prediction.label, image, conf_threshold)
            decision = SceneDecision(
                requested_mode="auto",
                selected_mode=scene_prediction.label,
                selected_label=MODES[scene_prediction.label].label,
                automatic=True,
                fallback_used=False,
                reason="场景分类置信度足够，直接调用对应检测模型。",
                scene_label=scene_prediction.label,
                scene_confidence=scene_prediction.confidence,
                scene_probabilities=scene_prediction.probabilities,
            )
            return predictor, detections, decision

    detector_outputs: dict[str, tuple[DetectionPredictor, list[Detection]]] = {}
    detector_scores: dict[str, float] = {}
    for mode_key in ("outdoor", "indoor"):
        if resolve_checkpoint_path(MODES[mode_key]) is None:
            continue
        try:
            predictor, detections = run_detector(mode_key, image, conf_threshold)
            detector_outputs[mode_key] = (predictor, detections)
            detector_scores[mode_key] = detection_score(detections)
        except Exception:
            continue

    if not detector_outputs:
        raise FileNotFoundError("没有可用的检测模型 checkpoint，请先训练模型。")

    selected_key = max(detector_scores, key=detector_scores.get)
    predictor, detections = detector_outputs[selected_key]
    reason = "场景分类器暂不可用，已同时运行两个检测模型并选择结果更可信的一侧。"
    if scene_prediction is not None:
        reason = "场景分类置信度偏低，已同时运行两个检测模型并选择结果更可信的一侧。"

    decision = SceneDecision(
        requested_mode="auto",
        selected_mode=selected_key,
        selected_label=MODES[selected_key].label,
        automatic=True,
        fallback_used=True,
        reason=reason,
        scene_label=scene_prediction.label if scene_prediction else None,
        scene_confidence=scene_prediction.confidence if scene_prediction else None,
        scene_probabilities=scene_prediction.probabilities if scene_prediction else None,
        detector_scores=detector_scores,
    )
    return predictor, detections, decision


def detect(
    requested_key: str,
    image: Image.Image,
    conf_threshold: float,
) -> tuple[DetectionPredictor, list[Detection], SceneDecision]:
    if requested_key == "auto":
        return automatic_detect(image, conf_threshold)

    predictor, detections = run_detector(requested_key, image, conf_threshold)
    decision = SceneDecision(
        requested_mode=requested_key,
        selected_mode=requested_key,
        selected_label=MODES[requested_key].label,
        automatic=False,
        fallback_used=False,
        reason="手动选择模式，未执行自动场景切换。",
    )
    return predictor, detections, decision


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "default_mode": "auto",
            "scene": scene_status(),
            "modes": {key: mode_status(mode) for key, mode in MODES.items()},
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        requested_key = get_requested_mode_key()
        threshold_mode = get_threshold_mode(requested_key)
        image = parse_image_from_request()
        conf_threshold = parse_conf_threshold(threshold_mode)
        _, detections, decision = detect(requested_key, image, conf_threshold)
        selected_mode = MODES[decision.selected_mode]
        return jsonify(
            {
                "ok": True,
                "mode": decision.selected_mode,
                "requested_mode": requested_key,
                "scene_decision": decision.to_dict(),
                "count": len(detections),
                "checkpoint": str(resolve_checkpoint_path(selected_mode)),
                "conf_threshold": conf_threshold,
                "detections": [item.to_dict() for item in detections],
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/", methods=["GET", "POST"])
def index():
    requested_key = get_requested_mode_key()
    threshold_mode = get_threshold_mode(requested_key)
    conf_threshold = threshold_mode.default_conf_threshold
    detections: list[Detection] = []
    result_image = None
    error = None
    decision = SceneDecision(
        requested_mode=requested_key,
        selected_mode="outdoor" if requested_key == "auto" else requested_key,
        selected_label="等待上传后自动判断" if requested_key == "auto" else MODES[requested_key].label,
        automatic=requested_key == "auto",
        fallback_used=False,
        reason="上传图片后会自动判断室内/室外。" if requested_key == "auto" else "当前为手动调试模式。",
    )

    if request.method == "POST":
        try:
            image = parse_image_from_request()
            conf_threshold = parse_conf_threshold(threshold_mode)
            predictor, detections, decision = detect(requested_key, image, conf_threshold)
            result_image = image_to_base64(predictor.render(image, detections))
        except Exception as exc:
            error = str(exc)

    selected_status = mode_status(MODES[decision.selected_mode]) if decision.selected_mode in MODES else None
    return render_template(
        "index.html",
        select_options=SELECT_OPTIONS,
        modes={key: mode_status(mode) for key, mode in MODES.items()},
        scene_status=scene_status(),
        requested_key=requested_key,
        selected_status=selected_status,
        decision=decision,
        detections=detections,
        result_image=result_image,
        error=error,
        conf_threshold=conf_threshold,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
