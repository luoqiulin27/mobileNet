from __future__ import annotations

import base64
import io
import json
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
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


# ==================== 摄像头管理器 ====================

class CameraManager:
    """管理摄像头连接和帧捕获"""

    def __init__(self):
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.camera_id = 0

    def start(self, camera_id=0):
        """启动摄像头"""
        if self.is_running:
            self.stop()
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        self.is_running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        """停止摄像头"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        with self.lock:
            self.current_frame = None

    def _capture_loop(self):
        """持续捕获帧"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
            else:
                break
            time.sleep(0.03)  # ~30fps

    def get_frame(self):
        """获取当前帧"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def is_active(self):
        """检查摄像头是否活跃"""
        return self.is_running


camera_manager = CameraManager()


def process_frame(frame, mode, conf_threshold):
    """处理单帧视频"""
    # BGR -> RGB -> PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 执行检测
    predictor, detections, decision = detect(mode, image, conf_threshold)

    # 渲染结果
    result_image = predictor.render(image, detections)

    # PIL Image -> JPEG bytes
    buffer = io.BytesIO()
    result_image.save(buffer, format="JPEG", quality=85)
    jpeg_bytes = buffer.getvalue()
    rendered_frame = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

    return jpeg_bytes, rendered_frame, detections, decision


def frame_to_jpeg_bytes(frame):
    """将 OpenCV 帧转换为 JPEG bytes"""
    _, jpeg_array = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_array.tobytes()


def build_response(payload: dict, status_code: int = 200):
    response = jsonify(payload)
    response.status_code = status_code
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


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


def read_json(path: Path) -> dict | list | None:
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
        "default_conf_threshold": mode.default_conf_threshold,
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


def build_prediction_payload(
    requested_key: str,
    image: Image.Image,
    conf_threshold: float,
) -> dict:
    predictor, detections, decision = detect(requested_key, image, conf_threshold)
    result_image = image_to_base64(predictor.render(image, detections))
    selected_mode = MODES[decision.selected_mode]
    return {
        "ok": True,
        "requested_mode": requested_key,
        "mode": decision.selected_mode,
        "scene_decision": decision.to_dict(),
        "count": len(detections),
        "checkpoint": str(resolve_checkpoint_path(selected_mode)),
        "conf_threshold": conf_threshold,
        "result_image": result_image,
        "detections": [item.to_dict() for item in detections],
    }


@app.route("/health", methods=["GET"])
def health():
    return build_response(
        {
            "ok": True,
            "default_mode": "auto",
            "scene": scene_status(),
            "modes": {key: mode_status(mode) for key, mode in MODES.items()},
            "select_options": [option.__dict__ for option in SELECT_OPTIONS],
        }
    )


@app.route("/api/meta", methods=["GET"])
def api_meta():
    return build_response(
        {
            "ok": True,
            "scene": scene_status(),
            "modes": {key: mode_status(mode) for key, mode in MODES.items()},
            "select_options": [option.__dict__ for option in SELECT_OPTIONS],
            "default_mode": "auto",
        }
    )


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        return build_response({"ok": True})

    try:
        requested_key = get_requested_mode_key()
        threshold_mode = get_threshold_mode(requested_key)
        image = parse_image_from_request()
        conf_threshold = parse_conf_threshold(threshold_mode)
        return build_response(build_prediction_payload(requested_key, image, conf_threshold))
    except Exception as exc:
        return build_response({"ok": False, "error": str(exc)}, 400)


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


# ==================== 摄像头 API ====================

@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    """启动摄像头"""
    try:
        camera_id = request.json.get("camera_id", 0) if request.is_json else 0
        camera_manager.start(camera_id)
        return build_response({"ok": True, "message": "摄像头已启动"})
    except Exception as exc:
        return build_response({"ok": False, "error": str(exc)}, 400)


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    """停止摄像头"""
    camera_manager.stop()
    return build_response({"ok": True, "message": "摄像头已停止"})


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status():
    """获取摄像头状态"""
    return build_response({
        "ok": True,
        "is_active": camera_manager.is_active(),
        "camera_id": camera_manager.camera_id
    })


@app.route("/api/camera/stream")
def api_camera_stream():
    """MJPEG 视频流"""
    mode = request.args.get("mode")
    conf_threshold = request.args.get("conf_threshold")
    detect_mode = mode is not None and conf_threshold is not None

    if detect_mode:
        conf_threshold = float(conf_threshold)

    def generate():
        while camera_manager.is_active():
            frame = camera_manager.get_frame()
            if frame is not None:
                try:
                    if detect_mode:
                        jpeg_bytes, _, _, _ = process_frame(frame, mode, conf_threshold)
                    else:
                        jpeg_bytes = frame_to_jpeg_bytes(frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                except Exception:
                    # 如果处理失败，返回原始帧
                    jpeg_bytes = frame_to_jpeg_bytes(frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            time.sleep(0.05)  # ~20fps

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/camera/frame", methods=["GET"])
def api_camera_frame():
    """获取单帧检测结果"""
    if not camera_manager.is_active():
        return build_response({"ok": False, "error": "摄像头未启动"}, 400)

    frame = camera_manager.get_frame()
    if frame is None:
        return build_response({"ok": False, "error": "无法获取帧"}, 400)

    mode = request.args.get("mode", "auto")
    conf_threshold = float(request.args.get("conf_threshold", "0.05"))

    try:
        jpeg_bytes, detections, decision = process_frame(frame, mode, conf_threshold)
        result_base64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        return build_response({
            "ok": True,
            "result_image": result_base64,
            "detections": [d.to_dict() for d in detections],
            "scene_decision": decision.to_dict(),
            "count": len(detections)
        })
    except Exception as exc:
        return build_response({"ok": False, "error": str(exc)}, 400)


# ==================== 视频文件处理 ====================

# 存储视频处理状态和控制
video_processing_status = {
    "is_processing": False,
    "is_paused": False,
    "progress": 0,
    "total_frames": 0,
    "current_frame": 0,
    "error": None,
    "mode": "auto",
    "conf_threshold": 0.05,
    "replay_dir": None,
    "replay_fps": 25,
}

# 视频流控制
video_stream_state = {
    "is_streaming": False,
    "current_frame_jpeg": None,
    "lock": threading.Lock()
}


def process_video_file(video_path, mode, conf_threshold):
    """处理视频文件，生成带检测框的视频流并保存结果视频"""
    global video_processing_status, video_stream_state

    print(f"[VIDEO] 开始处理视频: {video_path}, mode={mode}, conf={conf_threshold}")

    video_processing_status["is_processing"] = True
    video_processing_status["is_paused"] = False
    video_processing_status["progress"] = 0
    video_processing_status["error"] = None
    video_processing_status["mode"] = mode
    video_processing_status["conf_threshold"] = conf_threshold
    video_processing_status["current_frame"] = 0
    video_processing_status["total_frames"] = 0
    video_processing_status["output_path"] = None
    replay_dir = PROJECT_ROOT / "temp" / "video_replay"
    if replay_dir.exists():
        shutil.rmtree(replay_dir, ignore_errors=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    video_processing_status["replay_dir"] = str(replay_dir)
    video_stream_state["is_streaming"] = True

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_processing_status["total_frames"] = total_frames
        video_processing_status["replay_fps"] = fps

        print(f"[VIDEO] 视频信息: {total_frames} frames, {fps} fps, {width}x{height}")

        # 创建输出视频文件
        output_dir = PROJECT_ROOT / "temp"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "result_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            # 检查是否暂停
            while video_processing_status["is_paused"] and video_processing_status["is_processing"]:
                time.sleep(0.1)

            # 检查是否停止
            if not video_processing_status["is_processing"]:
                print("[VIDEO] 处理被停止")
                break

            ret, frame = cap.read()
            if not ret:
                print(f"[VIDEO] 读取帧失败，已处理 {frame_count} 帧")
                break

            frame_count += 1
            video_processing_status["current_frame"] = frame_count
            video_processing_status["progress"] = int(frame_count / total_frames * 100)

            if frame_count % 30 == 0:  # 每30帧打印一次
                print(f"[VIDEO] 处理进度: {frame_count}/{total_frames} ({video_processing_status['progress']}%)")

            # 处理每一帧，添加检测框
            try:
                jpeg_bytes, result_frame, detections, decision = process_frame(frame, mode, conf_threshold)
                # 将检测结果绘制到原始帧上用于保存
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                predictor = get_predictor(get_threshold_mode(mode))
                result_image = predictor.render(image, detections)
                result_frame = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[VIDEO] 帧 {frame_count} 处理失败: {e}")
                # 处理失败时使用原始帧
                jpeg_bytes = frame_to_jpeg_bytes(frame)
                result_frame = frame

            # 保存到输出视频
            out.write(result_frame)
            replay_frame_path = replay_dir / f"{frame_count:06d}.jpg"
            cv2.imwrite(str(replay_frame_path), result_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # 更新当前帧用于流显示
            with video_stream_state["lock"]:
                video_stream_state["current_frame_jpeg"] = jpeg_bytes

            # 控制播放速度
            time.sleep(1.0 / fps)

        cap.release()
        out.release()

        video_processing_status["is_processing"] = False
        video_processing_status["progress"] = 100
        video_processing_status["output_path"] = str(output_path)
        video_processing_status["replay_dir"] = str(replay_dir)
        video_stream_state["is_streaming"] = False

        print(f"[VIDEO] 处理完成，输出文件: {output_path}")

    except Exception as exc:
        print(f"[VIDEO] 处理异常: {exc}")
        video_processing_status["is_processing"] = False
        video_stream_state["is_streaming"] = False
        video_processing_status["error"] = str(exc)
        if 'out' in locals():
            out.release()


@app.route("/api/video/upload", methods=["POST"])
def api_video_upload():
    """上传视频文件"""
    # 如果正在处理，先停止
    if video_processing_status["is_processing"]:
        video_processing_status["is_processing"] = False
        time.sleep(0.5)

    video = request.files.get("video")
    if not video or not video.filename:
        return build_response({"ok": False, "error": "请选择视频文件"}, 400)

    # 保存视频文件
    temp_dir = PROJECT_ROOT / "temp"
    temp_dir.mkdir(exist_ok=True)
    video_path = temp_dir / "uploaded_video.mp4"
    video.save(video_path)

    # 获取参数
    mode = request.form.get("mode", "auto")
    conf_threshold = float(request.form.get("conf_threshold", "0.05"))

    # 启动后台处理线程
    threading.Thread(
        target=process_video_file,
        args=(video_path, mode, conf_threshold),
        daemon=True
    ).start()

    return build_response({"ok": True, "message": "视频上传成功，开始处理"})


@app.route("/api/video/status", methods=["GET"])
def api_video_status():
    """获取视频处理状态"""
    return build_response({
        "ok": True,
        **video_processing_status
    })


@app.route("/api/video/pause", methods=["POST"])
def api_video_pause():
    """暂停视频处理"""
    video_processing_status["is_paused"] = True
    return build_response({"ok": True, "message": "已暂停"})


@app.route("/api/video/resume", methods=["POST"])
def api_video_resume():
    """继续视频处理"""
    video_processing_status["is_paused"] = False
    return build_response({"ok": True, "message": "已继续"})


@app.route("/api/video/stop", methods=["POST"])
def api_video_stop():
    """停止视频处理"""
    video_processing_status["is_processing"] = False
    video_processing_status["is_paused"] = False
    return build_response({"ok": True, "message": "已停止"})


@app.route("/api/video/stream")
def api_video_stream():
    """MJPEG 视频流，返回带检测框的视频帧"""
    def generate():
        last_frame = None
        while True:
            # 检查是否应该继续
            is_streaming = video_stream_state["is_streaming"]
            is_processing = video_processing_status["is_processing"]

            with video_stream_state["lock"]:
                frame_data = video_stream_state["current_frame_jpeg"]

            if frame_data is not None:
                # 确保 frame_data 是 bytes 类型
                if isinstance(frame_data, bytes):
                    jpeg_bytes = frame_data
                else:
                    jpeg_bytes = frame_data.tobytes()

                # 只在帧更新时发送
                if jpeg_bytes != last_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                    last_frame = jpeg_bytes
            else:
                # 如果没有帧数据，等待一小段时间
                time.sleep(0.1)

            # 如果停止了且没有更多帧，退出
            if not is_processing and not is_streaming:
                # 发送最后一帧
                with video_stream_state["lock"]:
                    final_frame = video_stream_state["current_frame_jpeg"]
                if final_frame is not None and final_frame != last_frame:
                    if isinstance(final_frame, bytes):
                        jpeg_bytes = final_frame
                    else:
                        jpeg_bytes = final_frame.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                break

            time.sleep(0.03)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/video/replay")
def api_video_replay():
    """回放处理完成后的结果视频，返回 MJPEG 帧流用于前端稳定预览"""
    replay_dir_raw = video_processing_status.get("replay_dir")
    replay_dir = Path(replay_dir_raw) if replay_dir_raw else None
    if replay_dir is None or not replay_dir.exists():
        return build_response({"ok": False, "error": "视频回放帧不存在"}, 404)

    def generate():
        frame_paths = sorted(replay_dir.glob("*.jpg"))
        if not frame_paths:
            return

        fps = video_processing_status.get("replay_fps") or 25
        frame_delay = max(1.0 / fps, 0.03)

        try:
            while True:
                for frame_path in frame_paths:
                    jpeg_bytes = frame_path.read_bytes()
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n'
                    )
                    time.sleep(frame_delay)
        except GeneratorExit:
            pass

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.route("/api/video/download")
def api_video_download():
    """下载处理后的视频文件"""
    output_path = video_processing_status.get("output_path")
    if not output_path or not Path(output_path).exists():
        return build_response({"ok": False, "error": "视频文件不存在"}, 404)

    from flask import send_file
    response = send_file(
        output_path,
        mimetype='video/mp4',
        as_attachment=False,  # 改为 False，让浏览器播放而不是下载
        download_name='detected_video.mp4'
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Accept-Ranges"] = "bytes"
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
