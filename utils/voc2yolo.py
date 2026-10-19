"""
工具：VOC(XML) 标注 -> YOLO(txt) 标注

NEU-DET 原版给的是 VOC XML 标注，Ultralytics 需要 YOLO txt 格式：
    <class_id> <cx> <cy> <w> <h>     （全部相对图像宽高归一化，取值 0~1）

用法：
    python utils/voc2yolo.py --src data/NEU-DET/ANNOTATIONS --dst data/NEU-DET/labels --classes crazing inclusion patches pitted_surface rolled-in_scale scratches
"""

import argparse
import os
import glob
import xml.etree.ElementTree as ET

# NEU-DET 的 6 类，顺序决定 class id，必须与 02_yolo_neu_det.py 保持一致
DEFAULT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def convert_one(xml_path, dst_dir, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    if img_w == 0 or img_h == 0:
        return False

    lines = []
    for obj in root.iter("object"):
        cls_name = obj.find("name").text.strip()
        # NEU-DET 有的标注用 "rolled-in_scale" 有的用 "rolled_in_scale"，做一次兼容
        if cls_name not in classes:
            alt = cls_name.replace("_", "-")
            cls_name = alt if alt in classes else cls_name
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # 归一化
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # 裁剪到 0~1，防止标注越界
        cx, cy, w, h = [min(max(v, 0.0), 1.0) for v in (cx, cy, w, h)]
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(xml_path))[0]
    with open(os.path.join(dst_dir, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="XML 标注目录")
    ap.add_argument("--dst", required=True, help="输出 txt 目录")
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    args = ap.parse_args()

    xmls = glob.glob(os.path.join(args.src, "*.xml"))
    if not xmls:
        print(f"[错误] {args.src} 下没找到 .xml")
        return

    ok = 0
    for p in xmls:
        if convert_one(p, args.dst, args.classes):
            ok += 1

    print(f"[OK] 转换完成：{ok}/{len(xmls)} -> {args.dst}")
    print(f"类别顺序（决定 class id）：{args.classes}")
    print(">>> 下一步：确认 images 与 labels 目录结构后，跑 02_yolo_neu_det.py")


if __name__ == "__main__":
    main()
