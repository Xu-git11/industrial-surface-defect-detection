"""
MVTec AD 下载助手

说明：
    MVTec 官网的「按类别下载」链接指向 myDrive.ch 网盘，该网盘自 2025 年底起经常 404。
    此脚本内置的候选源（myDrive 社区镜像）大概率也已失效。
    如果脚本跑不通，请改用 `download_mvtec_hf.py`（HuggingFace 镜像）或先用 `prepare_demo_data.py`。

用法：
    # 方式 1：自动尝试内置候选源（大概率失败，仅作保留）
    python download_mvtec_ad.py

    # 方式 2：手动指定链接（从官网表单/邮件里复制）
    python download_mvtec_ad.py --url "https://...你的下载链接..."

    # 方式 3：指定保存位置
    python download_mvtec_ad.py --out data/mvtec_anomaly_detection.tar.xz

替代方案：
    - python download_mvtec_hf.py --category bottle   # HuggingFace 镜像，推荐
    - python prepare_demo_data.py                     # 本地合成数据，保底

特性：
    - 断点续传：中断后再跑一次会接着下，不会从头开始
    - 实时进度：显示已下载大小、百分比、速度
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error

# 候选下载源（2025 年底起 myDrive 已不稳定，这些链接大概率 404）
CANDIDATE_URLS = [
    # myDrive 社区镜像（已失效概率高）
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
]

OFFICIAL_PAGE = "https://www.mvtec.com/company/research/datasets/mvtec-ad"

# 数据集压缩包大小（字节），用于显示进度百分比；实际以服务器返回为准
EXPECTED_SIZE = 4.9 * 1024 * 1024 * 1024


def fmt_size(n):
    """字节数转人类可读。"""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_with_resume(url, out_path):
    """带断点续传的下载。返回 True 表示成功。"""
    tmp_path = out_path + ".part"

    # 已下载的部分
    if os.path.exists(tmp_path):
        downloaded = os.path.getsize(tmp_path)
        print(f"[续传] 发现未完成的下载：{fmt_size(downloaded)}")
    else:
        downloaded = 0

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable = 已下载完
            print("[完成] 文件已完整下载")
            if os.path.exists(tmp_path):
                os.rename(tmp_path, out_path)
            return True
        print(f"[失败] HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")
        return False

    # 总大小
    content_range = resp.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = int(content_range.split("/")[-1])
    else:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        if total and downloaded:
            total += downloaded

    if total == 0:
        total = EXPECTED_SIZE
        print(f"[提示] 服务器未返回文件大小，按预期 {fmt_size(total)} 估算进度")

    mode = "ab" if downloaded > 0 else "wb"
    start_time = time.time()
    last_report = 0

    print(f"[下载] 目标：{fmt_size(total)}")

    try:
        with open(tmp_path, mode) as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                # 每 2 秒报告一次进度
                now = time.time()
                if now - last_report >= 2:
                    elapsed = now - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    pct = 100.0 * downloaded / total if total else 0
                    eta = (total - downloaded) / speed if speed > 0 else 0
                    print(f"   {pct:5.1f}%  {fmt_size(downloaded)} / {fmt_size(total)}"
                          f"  {fmt_size(speed)}/s  剩余约 {eta/60:.0f} 分钟")
                    last_report = now
    except KeyboardInterrupt:
        print("\n[中断] 已保存进度，再次运行本脚本会断点续传")
        return False

    # 下载完成，改掉 .part 后缀
    os.rename(tmp_path, out_path)
    print(f"\n[完成] 已保存到 {out_path}  ({fmt_size(os.path.getsize(out_path))})")
    return True


def main():
    ap = argparse.ArgumentParser(description="下载 MVTec AD 数据集")
    ap.add_argument("--url", help="手动指定下载链接（从官网表单或邮件里复制）")
    ap.add_argument("--out", default="data/mvtec_anomaly_detection.tar.xz",
                    help="保存路径")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 已经下好了？
    if os.path.exists(args.out):
        size = os.path.getsize(args.out)
        print(f"[跳过] {args.out} 已存在（{fmt_size(size)}），无需重新下载")
        print(f"       若想重新下载，先删除该文件再运行")
        return

    urls = [args.url] if args.url else CANDIDATE_URLS

    for url in urls:
        print(f"\n{'='*60}")
        print(f"尝试：{url[:80]}{'...' if len(url) > 80 else ''}")
        print("="*60)
        if download_with_resume(url, args.out):
            print("\n下一步：解压")
            print(f'   tar -xf "{args.out}"')
            return
        print("该源失败，尝试下一个...")

    # 全部失败
    print("\n" + "="*60)
    print("自动下载失败。请手动下载：")
    print(f"  1. 打开 {OFFICIAL_PAGE}")
    print("  2. 填写表单（姓名/邮箱/机构/用途）")
    print("  3. 拿到下载链接后，用 --url 传给本脚本：")
    print(f'     python {os.path.basename(__file__)} --url "你的链接"')
    print("  4. 或者直接用下载工具（Motrix / 迅雷）下，支持断点续传")
    print("="*60)


if __name__ == "__main__":
    main()
