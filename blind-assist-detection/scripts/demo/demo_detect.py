"""
检测演示脚本 - 摄像头/视频
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.models.ssd_mobilenet import SSDMobileNetV2

# 类别颜色
COLORS = [
    (0, 200, 255),   # pedestrian
    (0, 255, 0),     # vehicle
    (255, 200, 0),   # rider
    (200, 0, 255),   # animal
    (0, 0, 255),     # stairs
    (255, 255, 0),   # pole
    (255, 128, 0),   # obstacle
    (128, 0, 255),   # furniture
]


def draw_detections(image, result, classes):
    """在图像上绘制检测结果"""
    img = np.array(image).copy()
    h, w = img.shape[:2]

    boxes = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    labels = result["labels"].cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        cls = int(labels[i]) - 1
        if cls < 0 or cls >= len(classes):
            continue

        color = COLORS[cls % len(COLORS)]
        score = scores[i]

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = "{} {:.0f}%".format(classes[cls], score * 100)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 顶部信息
    cv2.rectangle(img, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(img, "Objects: {}".format(len(boxes)), (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/ssd_default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pth")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:{}".format(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    # 加载类别
    with open(config["data"]["classes_file"], "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    # 加载模型
    input_size = config["model"]["input_size"][0]
    model = SSDMobileNetV2(
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        input_size=input_size,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("[Demo] 加载: {}".format(args.checkpoint))
    model.eval()

    # 视频源
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        print("[Demo] 请指定 --video 或 --camera")
        return

    print("[Demo] 按 q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((input_size, input_size), Image.BILINEAR)

        from torchvision import transforms
        tensor = transforms.ToTensor()(image)
        tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.224])(tensor)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            results = model.detect(tensor,
                                   conf_threshold=config["inference"]["conf_threshold"],
                                   nms_threshold=config["inference"]["nms_threshold"])

        result = results[0]
        vis = draw_detections(image, result, classes)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        cv2.imshow("Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
