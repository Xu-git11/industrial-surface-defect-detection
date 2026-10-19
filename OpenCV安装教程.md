# OpenCV 安装教程（Windows · 2026 年 10 月核对版）

> 面向零基础。按顺序做，不用跳步。
> 你的机器情况已确认：**Windows 64 位 + Python 3.13.9（Anaconda）**，可以正常安装。
> 版本号与依赖信息核对自 PyPI 官方接口（2026-10-17）；5.0 的破坏性变化依据 OpenCV 官方迁移文档。

---

## 零、先决定装哪个版本（这一步别跳过）

截至 2026 年 10 月，PyPI 上 OpenCV 有**两条并行版本线**，这是最容易踩的第一个坑：

| 版本 | 发布时间 | 说明 |
|---|---|---|
| **opencv-python 4.14.0.94** | 2026-07-28 | 4.x 线最新版，稳定主线 |
| **opencv-python 5.0.0.93** | 2026-07-02 | 8 年来首个大版本，2026-06-04 发布 |

Windows 64 位安装包：`opencv_python-4.14.0.94-cp37-abi3-win_amd64.whl`，**39.3 MB**。
`cp37-abi3` 表示用的是稳定 ABI，**Python 3.7 到 3.14 都能装**，你机器上的 3.13.9 没问题。

**结论：装 4.14.0.94。** 理由：

1. **所有中文教程、书、CSDN/CV 笔记都是 4.x**。你是小白，遇到报错要搜，5.0 的资料现在几乎没有。
2. **5.0 移掉了一些东西**：C 语言老接口全删、`readNetFromDarknet()`（读 YOLOv3/v4 权重）没了、Haar 级联和 HOG 移到 contrib。你第 3 周要跑的 YOLO 和常见检测代码，网上现成的多半是按 4.x 写的。
3. **5.0 才发布 4 个月**，ultralytics、各类开源项目的兼容还没完全跟上。
4. **你要投的岗位，产线上跑的还是 4.x**。面试不会因为你用了 5.0 加分。

> 好消息：Python 层面 5.0 对 `imread / cvtColor / 滤波 / 阈值 / Canny / 形态学` 这些基础 API **没变化**，你学的东西以后迁移成本几乎为零。所以先学 4.x 不亏。

---

## 一、装之前先确认 Python

打开**命令提示符**（Win + R → 输入 `cmd` → 回车），粘贴：

```
python --version
```

- 显示 `Python 3.13.9` 之类 → 有 Python，继续
- 报"不是内部或外部命令" → 说明没装或没加 PATH，看下面

### 如果你还没装 Python

1. 打开 https://www.python.org/downloads/
2. 下 **Python 3.11、3.12 或 3.13**（别下 3.14，部分视觉库还没跟上）
3. **安装时务必勾选 `Add Python to PATH`**（最关键的一步，不勾后面全是坑）
4. 装完重开 cmd，再跑 `python --version` 确认

> 你已经装了 Anaconda（Python 3.13.9），其实不用再装。但注意：**用 Anaconda 的话，请用开始菜单里的 "Anaconda Prompt" 而不是普通 cmd**，否则可能装在 base 环境里却找不到。

---

## 一·五、要不要建虚拟环境（建议建，5 分钟的事）

**结论：建。** 不是因为"最佳实践"这种空话，是因为你第 3 周要装 PyTorch。

### 判据：看你装什么

| 你要装的 | 建议 | 原因 |
|---|---|---|
| 只有 OpenCV + numpy + matplotlib | 不建也能用 | 就三件套，污染面小 |
| **加上 ultralytics（YOLO）** ← 你的情况 | **必须建** | 它会连带装 PyTorch |

**为什么 PyTorch 是分水岭：**

1. `ultralytics` 会拖进 torch（约 200 MB），外加 numpy、sympy、filelock、jinja2、networkx 一整串依赖，torch 会把其中几个**锁死在特定版本**。
2. 你用的是 Anaconda，**base 环境的依赖图是 conda 管的，conda 不知道 pip 往里塞了什么**。往 base 里 pip 装大包是公认的高风险操作，出问题往往不是立刻报错，而是某天 conda 突然动不了。
3. base 环境坏了，最坏要重装 Anaconda。**重装的成本远高于现在花 5 分钟建个环境。**

> 顺带确认一件事：torch 最新版 2.14.1 的 Windows wheel 覆盖 `cp310`–`cp314`，你机器上的 Python 3.13.9 **能装**，不用为此降级 Python。

### 最省事的做法：双击 `setup_env.bat`

它会自动完成：建 `cv_env` 环境 → 装 OpenCV 4.14.0.94 + numpy + matplotlib → 跑 `check_cv.py` 验证。装不上还会自动换清华源、阿里云源重试。

### 想手动做，就这四条命令

```
python -m venv cv_env
cv_env\Scripts\activate
python -m pip install opencv-python==4.14.0.94 numpy matplotlib
python check_cv.py
```

**第 2 条就是"激活"。** 激活成功的标志是**命令行最前面出现 `(cv_env)`**：

```
(cv_env) D:\UserData\Desktop\视觉项目>
```

看到这个前缀，后面所有 `python` 和 `pip` 命令才是在这个环境里执行。

### 如果 `python -m venv` 创建失败

Anaconda 的 Python 偶尔会在 `ensurepip` 这一步报错（环境里 pip 没装上）。改用 conda 建，效果一样：

```
conda create -n cv_env python=3.13 -y
conda activate cv_env
python -m pip install opencv-python==4.14.0.94 numpy matplotlib
```

激活成功的标志同样是命令行前面出现 `(cv_env)`。

> conda 方式要联网下载 Python 和基础包，比 venv 慢一些。两种方式选一个就行，**别混着用**。

### 三个必踩的坑，提前说

**坑 1：新开一个命令行窗口，就得重新激活一次。**
激活只对当前窗口有效。你在 A 窗口装好包，在 B 窗口 `import cv2` 会报 `ModuleNotFoundError`——这是新手最常遇到的"明明装了却找不到"。

嫌每次敲路径麻烦，可以在项目目录里放一个 `进入环境.bat`，内容一行：
```
cmd /k "cv_env\Scripts\activate"
```

**坑 2：VS Code 不会自动跟着激活。**
按 `Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `视觉项目\cv_env\Scripts\python.exe`。选完左下角会显示 `cv_env`。

**坑 3：PyCharm 同理。**
`Settings` → `Python Interpreter` → 添加 → 选 `cv_env\Scripts\python.exe`。

### 如果实在嫌麻烦

那至少要守住这条线：**base 环境里只装 OpenCV + numpy + matplotlib，等第 3 周要跑 YOLO 时再建环境**。别在 base 里装 torch。

---

## 二、三种装法（推荐第一种）

### 方式 A：pip 一行命令（推荐）

在 cmd 或 Anaconda Prompt 里粘贴：

```
python -m pip install --upgrade pip
```

```
python -m pip install opencv-python==4.14.0.94 numpy
```

- 用 `python -m pip` 而不是直接 `pip`，可以避免"pip 装到别的 Python 上"这个经典坑
- 下载约 39 MB（连同 NumPy 共约 60 MB），正常网速 1–3 分钟
- **注意**：OpenCV 4.14 在 Python 3.9 及以上要求 **`numpy>=2`**。如果你以前装过 NumPy 1.x，pip 会自动升级；万一和你别的老包冲突，就新建一个虚拟环境隔离

**注意：别同时装 `opencv-python` 和 `opencv-contrib-python`。**
两个包会冲突（都提供 `cv2` 模块）。contrib 里是 SIFT、SURF、一些专利算法，**初学阶段用不到**，先别装。真需要时再卸载重装：

```
python -m pip uninstall opencv-python
python -m pip install opencv-contrib-python==4.14.0.94
```

### 方式 B：下载慢 / 超时 → 换清华镜像

```
python -m pip install opencv-python==4.14.0.94 numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其他可用镜像：阿里云 `https://mirrors.aliyun.com/pypi/simple/`、中科大 `https://pypi.mirrors.ustc.edu.cn/simple/`

### 方式 C：Anaconda（不太推荐）

```
conda install -c conda-forge opencv
```

conda 源里的版本通常落后，而且和 pip 混用容易出依赖冲突。**你要是已经习惯 pip 就别用 conda 装。**

---

## 三、验证装好了没有

### 30 秒快速验证

```
python -c "import cv2; print(cv2.__version__)"
```

打印出 `4.14` 开头就成功了（完整是 `4.14.0`）。

### 完整验证（推荐，跑一遍等于把第 1 周的课预习了）

在本文件夹里：

- **双击 `install_opencv.bat`** —— 自动装完并跑验证
- 或者手动跑：`python check_cv.py`

这个脚本会做三件事：

1. 打印 Python / NumPy / OpenCV 版本
2. 造一张"金属件 + 划痕 + 凹坑 + 噪点"的模拟工业图
3. 把第 1、2 周要用的操作**全部跑一遍**并给出具体数字：

| 步骤 | 你会看到的输出 |
|---|---|
| 灰度化 | 通道数 3 → 1 |
| 中值 / 高斯滤波 | 噪点被压掉 |
| Otsu 阈值 | **自动算出的阈值是多少** |
| Canny 边缘 | 边缘像素个数 |
| 形态学闭运算 | 缺陷区域被连通 |
| Blob 分析 | 检出几个缺陷、各自面积和质心 |
| 模板匹配 | 匹配得分（1.0 为完全一致） |
| 轮廓提取 / 尺寸测量 | 缺陷外接矩形尺寸 |

跑完还会把 6 张中间结果图存到 `output/` 目录，**这些图就是以后面试可以展示的东西**。

> **把脚本输出的数字记下来。** 面试说"我用 Otsu 自动阈值分割，阈值算出是 XXX，Canny 提取到 XXXX 个边缘像素"，比说"我学过 OpenCV"有力十倍。

---

## 四、报错怎么办（按出现频率排）

### 1. `'pip' 不是内部或外部命令`

改用 `python -m pip` 开头。或者重装 Python 勾选 Add to PATH。

### 2. 下载卡住 / ReadTimeout

换清华源（见方式 B）。或者加超时参数：

```
python -m pip install --default-timeout=100 opencv-python==4.14.0.94 numpy
```

### 3. `ImportError: numpy.core.multiarray failed to import`

NumPy 版本和 OpenCV 不匹配。升级 NumPy：

```
python -m pip install -U numpy
```

### 4. `ImportError: DLL load failed while importing cv2`

常见原因：
- Python 是 32 位但装了 64 位包（或反过来）→ `python -c "import platform; print(platform.architecture())"` 确认是 64bit
- 缺 Visual C++ 运行库 → 装微软常用运行库合集（搜 "Microsoft Visual C++ Redistributable" 下载最新版）
- 最省事的解法：卸载重装

```
python -m pip uninstall opencv-python numpy
python -m pip install opencv-python==4.14.0.94 numpy
```

### 5. `cv2.imshow()` 报错 / 窗口闪退

无图形界面环境（远程桌面、某些服务器）会这样。**不影响学习**，改用存图：

```python
cv2.imwrite("output.png", img)   # 代替 cv2.imshow
```

### 6. 装完了，但 VS Code 里 `import cv2` 报错

VS Code 用的解释器和你装包的不是同一个。按 `Ctrl+Shift+P` → 输入 `Python: Select Interpreter` → 选你装包的那个 Python（Anaconda 的话选 `D:\pythonSoftstore\anaconda3\python.exe`）。

### 7. 权限不足（Permission denied）

装在用户目录，不加 sudo：

```
python -m pip install --user opencv-python==4.14.0.94 numpy
```

---

## 五、顺手把这些也装了（第 3 周才用得到，但现在装省事）

**先确认命令行前面有 `(cv_env)`**，然后再装：

```
python -m pip install matplotlib ultralytics onnx onnxruntime
```

> ⚠️ 没看到 `(cv_env)` 前缀就别执行——这条命令会把 200 MB+ 的 PyTorch 塞进 base 环境，正是第一·五节说的那个坑。

| 包 | 干什么 | 什么时候用 |
|---|---|---|
| matplotlib | 画图、显示对比结果 | 第 1 周 |
| ultralytics | YOLOv8 训练与推理 | 第 3 周 |
| onnx / onnxruntime | 模型导出与部署测速 | 第 3 周 |

> `ultralytics` 会连带装 PyTorch（约 200 MB+），时间较长。不想等就第 3 周再装。

---

## 六、装完立刻做这一件事

创建一个文件 `hello.py`，粘贴下面这段，跑起来：

```python
import cv2
import numpy as np

# 造一张 400x300 的灰色图，画一道深色划痕
img = np.full((300, 400), 160, dtype=np.uint8)
cv2.line(img, (60, 120), (340, 124), 30, 2)

# 中值滤波去噪
blur = cv2.medianBlur(img, 5)

# Otsu 自动阈值，把划痕（暗）从背景（亮）里分出来
th, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu 自动算出的阈值：", th)

# 数一下划痕有多少个像素
print("缺陷像素数：", int((binary == 0).sum()))

cv2.imwrite("result.png", binary)
print("已保存 result.png")
```

能打印出阈值和像素数、`result.png` 能打开看到一条黑线——**你的机器视觉第一公里就走完了。**

---

## 七、常见疑问

**Q：一定要用 Anaconda 吗？**
不用。Anaconda 只是方便管理环境。你已经装了就直接用，用 Anaconda Prompt 操作。

**Q：要不要建虚拟环境？**
建议建，理由和具体命令见 **第一·五节**。一句话版：只装 OpenCV 三件套可以不用建，但你第 3 周要跑 YOLO 就得装 PyTorch，那时候必须隔离——不如现在就建，省得第 3 周推倒重来。双击 `setup_env.bat` 全自动。

**Q：装了 4.x 以后想换成 5.0 怎么办？**

```
python -m pip uninstall opencv-python
python -m pip install opencv-python==5.0.0.93
```

**Q：工业界用 Halcon / VisionMaster，我学 OpenCV 有用吗？**
有用，而且是最优解。OpenCV 是**唯一你能合法免费拿到、能写进 GitHub、面试官能当场验证**的。Halcon 和 VisionMaster 是同类算子库，有 OpenCV 基础上手就是一两周的事——面试时这么说，反而是加分项。
