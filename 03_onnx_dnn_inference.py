"""
03 - 导出 ONNX + OpenCV DNN 推理 + 延迟测试（不需要 GPU）

目的：验证「训练好的模型能在产线工控机上跑起来」，测出端到端延迟。
这是面试时最值钱的数字之一：CPU 单张推理 < X ms，能否满足产线节拍。

用法：
    python 03_onnx_dnn_inference.py --weights runs/detect/neu_det/weights/best.pt
    python 03_onnx_dnn_inference.py --weights best.pt --source data/NEU-DET/valid/images --runs 100
"""

import argparse
import json
import os
import glob
import time
import numpy as np
import cv2

NEU_CLASSES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]


def export_onnx(weights, imgsz=640, opset=12):
    """把 .pt 导出为 ONNX（简化图、静态输入、便于 OpenCV DNN 加载）。"""
    from ultralytics import YOLO
    print(f"导出 ONNX：{weights}")
    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset,
                             simplify=True, dynamic=False)
    print(f"[OK] ONNX 已保存：{onnx_path}")
    return onnx_path


def build_blob(img, imgsz=640):
    """图像预处理：letterbox 缩放 + 归一化 + NCHW，与训练保持一致。"""
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)   # letterbox 灰边
    canvas[:nh, :nw] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (imgsz, imgsz),
                                 swapRB=True, crop=False)
    return blob, scale


def nms(boxes, scores, conf_thres=0.25, iou_thres=0.45):
    """标准 NMS。boxes 为 xyxy 格式。"""
    if len(boxes) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(
        [list(b) for b in boxes], [float(s) for s in scores], conf_thres, iou_thres
    )
    return [int(i) for i in np.array(idxs).flatten()] if len(idxs) else []


def parse_yolo_output(out, scale, imgsz=640, conf_thres=0.25):
    """
    解析 YOLOv8 ONNX 输出。
    输出形状 (1, 4+nc, 8400)：前 4 行是 cx,cy,w,h（像素单位，相对 640），后 nc 行是各类分数。
    """
    pred = out[0]                       # (4+nc, 8400)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                   # 有的导出是 (8400, 4+nc)，统一成 (8400, 4+nc)
    num_cls = pred.shape[1] - 4

    boxes, scores, class_ids = [], [], []
    for row in pred:
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        cls_scores = row[4:4 + num_cls]
        cid = int(np.argmax(cls_scores))
        conf = float(cls_scores[cid])
        if conf < conf_thres:
            continue
        # cx,cy,w,h → xyxy，再除以 scale 还原到原图坐标
        x1 = (cx - bw / 2) / scale
        y1 = (cy - bh / 2) / scale
        x2 = (cx + bw / 2) / scale
        y2 = (cy + bh / 2) / scale
        boxes.append([x1, y1, x2 - x1, y2 - y1])   # NMSBoxes 要 xywh
        scores.append(conf)
        class_ids.append(cid)
    return boxes, scores, class_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/neu_det/weights/best.pt")
    ap.add_argument("--onnx", default="", help="已有 ONNX 可直接指定，跳过导出")
    ap.add_argument("--source", default="", help="测试图片目录或单张图；留空用随机图")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=100, help="延迟测试重复次数")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    onnx_path = args.onnx
    if not onnx_path:
        onnx_path = export_onnx(args.weights, args.imgsz)
    if not os.path.exists(onnx_path):
        print(f"[错误] ONNX 不存在：{onnx_path}")
        return

    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print(f"[OK] OpenCV DNN 已加载：{onnx_path}")

    # 准备测试图
    if args.source and os.path.isdir(args.source):
        imgs = sorted(glob.glob(os.path.join(args.source, "*.jpg")))[:20]
        imgs = [cv2.imread(p) for p in imgs]
        imgs = [i for i in imgs if i is not None]
    elif args.source and os.path.isfile(args.source):
        imgs = [cv2.imread(args.source)]
    else:
        imgs = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
    if not imgs:
        print("[错误] 没有可用测试图")
        return

    # ---- 延迟测试（端到端：blob 构建 + 前向 + 后处理）----
    latencies = []
    for i in range(args.runs):
        img = imgs[i % len(imgs)]
        t0 = time.perf_counter()
        blob, scale = build_blob(img, args.imgsz)
        net.setInput(blob)
        out = net.forward()
        boxes, scores, cids = parse_yolo_output(out, scale, args.imgsz, args.conf)
        keep = nms(boxes, scores, args.conf)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)      # ms

    lat = np.array(latencies)
    summary = dict(
        onnx=onnx_path,
        imgsz=args.imgsz,
        runs=args.runs,
        mean_ms=float(lat.mean()),
        std_ms=float(lat.std()),
        p50_ms=float(np.percentile(lat, 50)),
        p95_ms=float(np.percentile(lat, 95)),
        min_ms=float(lat.min()),
        max_ms=float(lat.max()),
        fps=float(1000.0 / lat.mean()),
        device="CPU (OpenCV DNN)",
    )
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/03_dnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 可视化一张结果
    img = imgs[0].copy()
    blob, scale = build_blob(imgs[0], args.imgsz)
    net.setInput(blob)
    out = net.forward()
    boxes, scores, cids = parse_yolo_output(out, scale, args.imgsz, args.conf)
    keep = nms(boxes, scores, args.conf)
    for i in keep:
        x, y, w, h = boxes[i]
        cls_name = NEU_CLASSES[cids[i]] if cids[i] < len(NEU_CLASSES) else str(cids[i])
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 2)
        cv2.putText(img, f"{cls_name} {scores[i]:.2f}", (int(x), max(int(y) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.imwrite("outputs/03_dnn_result.jpg", img)

    print("\n" + "=" * 56)
    print(f"端到端延迟（{args.runs} 次，{summary['device']}）")
    print(f"  平均   : {summary['mean_ms']:.1f} ms")
    print(f"  P50    : {summary['p50_ms']:.1f} ms")
    print(f"  P95    : {summary['p95_ms']:.1f} ms")
    print(f"  最快   : {summary['min_ms']:.1f} ms   最慢: {summary['max_ms']:.1f} ms")
    print(f"  吞吐   : {summary['fps']:.1f} FPS")
    print("=" * 56)
    print(f"检出目标数：{len(keep)}（示意图已存 outputs/03_dnn_result.jpg）")
    print("\n>>> 面试用数字：把『平均 X ms、约 Y FPS』记下来。")
    print(">>> 对照产线节拍：一般要求单件 < 200ms，若你测出 < 100ms 就是能过线的。")


if __name__ == "__main__":
    main()
