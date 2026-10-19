"""
OpenCV 安装验证脚本
------------------
作用：装完 OpenCV 后跑这个脚本，确认环境没问题，同时把第 1 周要用的
     五个预处理操作全部跑一遍，并输出你可以记下来的具体数字。

用法：python check_cv.py
"""

import sys
import platform

print("=" * 60)
print("环境检查")
print("=" * 60)
print(f"Python 版本 : {sys.version.split()[0]}")
print(f"系统        : {platform.system()} {platform.machine()}")

try:
    import numpy as np
except ImportError:
    print("\n[失败] 没装 NumPy。请先运行：pip install numpy")
    sys.exit(1)
print(f"NumPy 版本  : {np.__version__}")

try:
    import cv2
except ImportError:
    print("\n[失败] 没装 OpenCV。请运行：pip install opencv-python==4.14.0.94")
    sys.exit(1)

print(f"OpenCV 版本 : {cv2.__version__}")
print()

# ---------------------------------------------------------------
# 造一张"工业检测风格"的测试图：金属灰底 + 一道划痕 + 一个凹坑 + 噪点
# ---------------------------------------------------------------
np.random.seed(42)
h, w = 300, 400

color = np.full((h, w, 3), 160, dtype=np.uint8)          # 金属灰底
cv2.line(color, (60, 120), (340, 124), (30, 30, 30), 2)   # 划痕（深色细线）
cv2.circle(color, (200, 210), 15, (50, 50, 50), -1)       # 凹坑（深色圆斑）

# 撒 2% 椒盐噪点，模拟传感器噪声
sp_mask = np.random.rand(h, w) < 0.02
color[sp_mask] = np.random.choice([0, 255], size=(int(sp_mask.sum()), 3)).astype(np.uint8)

print("=" * 60)
print("第 1 周五个预处理操作 —— 逐个验证")
print("=" * 60)

ok = []

# 1. 灰度化
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
print(f"[1] 灰度化      -> shape={gray.shape}  通道数从 3 降到 1，计算量约降为 1/3")
ok.append(("灰度化", gray.ndim == 2))

# 2. 滤波去噪（中值去椒盐 / 高斯保边缘）
median = cv2.medianBlur(gray, 5)
gauss = cv2.GaussianBlur(gray, (5, 5), 0)
print(f"[2] 滤波去噪    -> 中值滤波(5x5)、高斯滤波(5x5) 均完成")
ok.append(("中值滤波", median.shape == gray.shape))
ok.append(("高斯滤波", gauss.shape == gray.shape))

# 3. Otsu 阈值分割
_, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
auto_thresh = _ if isinstance(_, (int, float)) else None
print(f"[3] Otsu 阈值   -> 自动算出阈值 = {auto_thresh:.1f}，前景像素 {int((otsu == 0).sum())} 个")
ok.append(("Otsu 阈值", otsu.shape == gray.shape))

# 4. Canny 边缘提取
edges = cv2.Canny(gauss, 50, 150)
print(f"[4] Canny 边缘  -> 边缘像素 {int((edges > 0).sum())} 个")
ok.append(("Canny 边缘", edges.shape == gray.shape))

# 5. 形态学闭运算（把断裂的缺陷区域连成整块）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
print(f"[5] 形态学闭运算 -> 前景像素 {int((closed == 0).sum())} 个（比闭运算前更连贯）")
ok.append(("形态学闭运算", closed.shape == gray.shape))

print()
print("=" * 60)
print("第 2 周检测算法 —— 逐个验证")
print("=" * 60)

# 6. Blob 分析（连通域）
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    (closed == 0).astype(np.uint8), connectivity=8
)
blobs = num_labels - 1  # 减掉背景
print(f"[6] Blob 分析   -> 检出 {blobs} 个连通域（缺陷候选）")
for i in range(1, min(num_labels, 6)):
    x, y, bw, bh, area = stats[i]
    cx, cy = centroids[i]
    print(f"      缺陷{i}: 面积={area:5d}px  外接框={bw}x{bh}  质心=({cx:.0f},{cy:.0f})")
ok.append(("Blob 分析", num_labels >= 1))

# 7. 模板匹配
tpl = color[100:140, 180:230].copy()                       # 从原图抠一块当模板
res = cv2.matchTemplate(gray, cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)
print(f"[7] 模板匹配    -> 最佳匹配得分 {max_val:.3f}，位置 {max_loc}（1.0 为完全一致）")
ok.append(("模板匹配", res.shape[0] > 0))

# 8. 轮廓提取
contours, _ = cv2.findContours((closed == 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"[8] 轮廓提取    -> 找到 {len(contours)} 条外轮廓")
ok.append(("轮廓提取", isinstance(contours, (list, tuple))))

# 9. 尺寸测量（最小外接矩形）
if len(contours) > 0:
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (cx, cy), (rw, rh), angle = rect
    print(f"[9] 尺寸测量    -> 最大缺陷外接矩形 {rw:.1f} x {rh:.1f} px，旋转角 {angle:.1f}°")
    ok.append(("尺寸测量", True))
else:
    ok.append(("尺寸测量", False))

# 10. 图形界面（有些无显示器的环境会失败，属正常）
gui_ok = True
try:
    cv2.imshow("test", gray)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"[10] 窗口显示   -> 正常（能弹出 imshow 窗口）")
except Exception as e:
    gui_ok = False
    print(f"[10] 窗口显示   -> 不可用（{type(e).__name__}），用 cv2.imwrite 存图代替即可，不影响学习")
ok.append(("窗口显示", gui_ok))

# 11. 存图（验证 imwrite，把你跑出的中间结果存下来，面试可以展示）
import os
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(out_dir, exist_ok=True)
for name, arr in [("01_原图", color), ("02_灰度", gray), ("03_中值滤波", median),
                  ("04_Otsu二值", otsu), ("05_Canny边缘", edges), ("06_闭运算", closed)]:
    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), arr)
print(f"[11] 存图        -> 中间结果已保存到 output/ 目录（共 6 张）")

# ---------------------------------------------------------------
print()
print("=" * 60)
failed = [n for n, s in ok if not s]
if failed:
    print(f"结果：{len(ok) - len(failed)}/{len(ok)} 项通过")
    print(f"未通过（通常不影响使用）：{', '.join(failed)}")
else:
    print(f"结果：全部 {len(ok)} 项通过 —— 环境没问题，可以开始第 1 周了")
print("=" * 60)
print()
print("把上面这些数字记下来（Otsu 阈值、边缘像素数、缺陷面积等），")
print("面试时讲得出具体数字，比说『我学过 OpenCV』有用得多。")
