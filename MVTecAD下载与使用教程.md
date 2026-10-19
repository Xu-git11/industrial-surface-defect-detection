# MVTec AD 下载与使用的完整教程

> 面向 Windows + 零基础。写于 2026-10-19，版本信息核对自 MVTec 官网与公开数据集索引。

---

## 零、先选对版本：MVTec AD 还是 MVTec AD 2？

**这两个不是同一数据集的新旧版本，是两个独立数据集。别下错。**

| | **MVTec AD**（2019） | **MVTec AD 2**（2025） |
|---|---|---|
| 类别数 | **15 个**（物件 + 纹理） | **8 个** |
| 图像数 | 5000+ | 8000+ |
| 难度 | 基准级，多数方法已接近饱和 | 更难，SOTA 平均 AU-PRO 仍低于 60% |
| 特点 | 经典基准，教程/代码生态最全 | 透明物体、重叠物体、暗场/背光照明、极小缺陷 |
| 许可证 | CC BY-SA 4.0 | 见官网 |

**结论：下 MVTec AD（2019 版）。** 理由：

1. 你的 `01_traditional_opencv.py` 是按它的目录结构写的，**开箱即用**
2. 中文教程、博客、面试资料 99% 指的是这一版
3. 它的 `bottle` 类最经典，是讲"传统方法"的标准案例
4. MVTec AD 2 的目录结构和评估服务器（`benchmark.mvtec.com`）不一样，脚本要改

> 等你把 01 脚本跑通、拿到数字之后，想深入了解再看 AD 2。现在别分散火力。

---

## 一、下载

### 官方地址

```
https://www.mvtec.com/company/research/datasets/mvtec-ad
```

（官网已改版，旧链接 `.../company/research/datasets/mvtec-ad` 会跳转到 `.../research-teaching/datasets/mvtec-ad`，两个都能到）

### 下载流程

1. 打开上面的网址，往下找到 **Download** 区域
2. 填表单：通常要 **姓名、邮箱、机构/公司、用途说明**
3. 提交后，**多数情况会立即开始下载**；也有用户反馈会发一封邮件，邮件里给下载链接
4. 文件名：`mvtec_anomaly_detection.tar.xz`，**约 4.9 GB**

> 如果没收到邮件，检查垃圾箱；还是没有就换一个邮箱（建议用学校邮箱或 Gmail，某些企业邮箱会拦）

### 下载的三个坑

**坑 1：4.9 GB 别用浏览器默认下载。** 断网就得从头来。推荐用下载工具：

- **Motrix**（免费开源，支持断点续传）：motrix.app
- **迅雷**：把下载链接丢进去
- 或者用我写的脚本（见下一节）

**坑 2：磁盘要留够空间。** 压缩包 4.9 GB + 解压后约 5 GB，**至少留 12 GB**（解压过程还有临时开销）。

**坑 3：解压后才是文件夹。** `.tar.xz` 是 Linux 常见的双层压缩，Windows 资源管理器不能直接打开，看第三节。

---

## 一·五、下载文件放哪个文件夹（先看这个）

**结论：放到 `D:\UserData\Desktop\视觉项目\data\`**

三个理由：

1. **你的项目本来就在 D 盘**——`D:\UserData\Desktop\视觉项目`，不用另外找地方
2. `01_traditional_opencv.py` 默认去 `data/mvtec_anomaly_detection/` 找数据，放这儿开箱即用
3. D 盘可用空间充足（实测 148 GB）

> ⚠️ **浏览器默认下载目录在 C 盘，不是 D 盘。**
> Chrome / Edge 默认存到 `C:\Users\Lenovo\Downloads`。

| 你怎么下 | 文件会落在哪 | 你要做什么 |
|---|---|---|
| 运行 `download_mvtec_ad.py` | 自动放到 `视觉项目\data\` | 什么都不用做 |
| **浏览器直接点下载链接** | `C:\Users\Lenovo\Downloads` | **下完剪切到 `视觉项目\data\`** |
| 迅雷 / Motrix | 新建任务时自己选 | 直接指定 `D:\UserData\Desktop\视觉项目\data\` |

也可以改浏览器默认位置（设置 → 下载内容 → 位置 → 改到 D 盘），但只下一次，手动移一下更快。

### 空间账

| 项目 | 大小 | 说明 |
|---|---|---|
| 压缩包 | 4.9 GB | 下载时占 |
| 解压后 | ~5 GB | 全部 15 类 |
| **峰值占用** | **~10 GB** | 压缩包还没删时 |
| 你的 D 盘可用 | 148 GB | 充足，不用清盘 |

**省空间做法**：解压完 bottle 一个类就把 `.tar.xz` 删掉，立刻回收 4.9 GB。

### 只做 bottle 类（推荐，省时间省空间）

解压时只解 bottle，其它 14 类留在压缩包里：

```cmd
cd /d "D:\UserData\Desktop\视觉项目\data"
tar -xf mvtec_anomaly_detection.tar.xz mvtec_anomaly_detection/bottle
```

这样实际占用只有几百 MB，而不是 5 GB。

---

## 二、官网链接失效怎么办（2025 年底起常见）

MVTec AD 官网的「按类别下载」链接指向 myDrive.ch 网盘，**该网盘已不稳定/经常 404**。你看到的「页面未找到」就是这个原因，不是操作问题。

### 替代方案（按推荐顺序）

| 方案 | 操作 | 优点 | 缺点 |
|---|---|---|---|
| **A. HuggingFace 镜像脚本** | `python download_mvtec_hf.py --category bottle` | 只下 bottle，自动保存成项目需要的结构 | 需要安装 `datasets` 库；国内访问可能不稳定 |
| **B. Kaggle 镜像** | 注册 Kaggle → 搜 `ipythonx/mvtec-ad` → 手动下载 | 整包稳定，约 5 GB | 需要注册 + 手动解压 |
| **C. 先用合成数据** | `python prepare_demo_data.py` | 立即跑通，不依赖外网 | 不是真实工业图，面试时要诚实说明 |

> 建议：**先执行 C（合成数据）保证流程跑通，同时尝试 A（HuggingFace）下载真实数据**。不要卡在下载上。

### 方案 A：HuggingFace 镜像（首选）

```cmd
cd /d "D:\UserData\Desktop\视觉项目"
python -m pip install datasets
python download_mvtec_hf.py --category bottle
```

脚本默认走 HuggingFace 国内镜像 `https://hf-mirror.com`，下载 `TheoM55/mvtec_all_objects_split` 数据集的 bottle 类，并自动整理成：

```
data/mvtec_anomaly_detection/bottle/
├── train/good/
├── test/good/
├── test/broken_large/
├── test/broken_small/
├── test/contamination/
└── ground_truth/
    ├── broken_large/
    ├── broken_small/
    └── contamination/
```

如果 HuggingFace 也访问失败，脚本会给出明确提示，并 fallback 到「先用合成数据」。

### 方案 B：Kaggle 镜像

1. 注册 Kaggle 账号（可用邮箱或 Google 账号）
2. 打开 https://www.kaggle.com/datasets/ipythonx/mvtec-ad
3. 点 **Download**（约 5 GB）
4. 解压到 `D:\UserData\Desktop\视觉项目\data\mvtec_anomaly_detection\`
5. 如果只想要 bottle，在压缩包里只把 `bottle/` 文件夹拖出来

### 方案 C：先用合成数据（保底）

如果 A 和 B 都失败，不要停在这儿：

```cmd
python prepare_demo_data.py
python 01_traditional_opencv.py --data data/mvtec_ad_demo --category bottle
```

这组数据能产出完整的 IoU / Precision / Recall / 误检率 / 漏检率，足够你理解流程和准备面试。面试时诚实说「这是本地合成数据，真实 MVTec AD 我正在下载」即可。

---

## 三、Windows 解压 .tar.xz（关键步骤）

`.tar.xz` 是「先 tar 打包、再 xz 压缩」的两层结构，必须正确处理。

### 方式 A：命令行 tar（推荐，一步到位）

Windows 10（1803 之后）和 Windows 11 **自带 tar**，不用装任何东西。

1. 把 `mvtec_anomaly_detection.tar.xz` 放到 `D:\UserData\Desktop\视觉项目\data\`
2. 打开 cmd，执行：

```cmd
cd /d "D:\UserData\Desktop\视觉项目\data"
tar -xf mvtec_anomaly_detection.tar.xz
```

等几分钟，会生成 `mvtec_anomaly_detection\` 文件夹。

**只想解压 bottle 一个类别**（省时间省空间）：

```cmd
tar -xf mvtec_anomaly_detection.tar.xz mvtec_anomaly_detection/bottle
```

### 方式 B：7-Zip（图形界面，要两步）

1. 装 7-Zip：7-zip.org
2. 右键 `mvtec_anomaly_detection.tar.xz` → **7-Zip → 解压到当前文件夹**
   → 得到 `mvtec_anomaly_detection.tar`（还是个大文件）
3. 右键 `mvtec_anomaly_detection.tar` → **7-Zip → 解压到 "mvtec_anomaly_detection\"**
   → 得到真正的文件夹

**省空间技巧**：第 2 步之后，双击打开 `.tar`（7-Zip 能像文件夹一样打开它），**只把 `bottle` 文件夹拖出来**。这样不用解压全部 5 GB。

### 方式 C：WinRAR / Bandizip

新版本都能一步打开 `.tar.xz`，双击后直接拖出需要的文件夹。

---

## 四、目录结构（解压后长这样）

```
mvtec_anomaly_detection/
├── bottle/
│   ├── train/
│   │   └── good/                    ← 只有正常件（训练用）
│   ├── test/
│   │   ├── good/                    ← 正常件（测试用，算误检率）
│   │   ├── broken_large/            ← 大破损
│   │   ├── broken_small/            ← 小破损
│   │   └── contamination/           ← 污染
│   └── ground_truth/
│       ├── broken_large/            ← 像素级 mask，文件名 xxx_mask.png
│       ├── broken_small/
│       └── contamination/
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
```

**15 个类别**：10 个物件（bottle、cable、capsule、hazelnut、metal_nut、pill、screw、toothbrush、transistor、zipper）+ 5 个纹理（carpet、grid、leather、tile、wood）。

**关键点**：`ground_truth` 里**没有** `good` 子目录——因为正常件没有缺陷区域要标注。你的脚本里也是这么判断的：只有坏件才去比对 ground truth。

---

## 五、放进项目并跑起来

假设你把数据集解压到了 `D:\UserData\Desktop\视觉项目\data\mvtec_anomaly_detection\`。

### 方式 1：直接运行（推荐）

脚本默认路径就是官方解压后的文件夹名，不需要改名：

```cmd
cd /d "D:\UserData\Desktop\视觉项目"
cv_env\Scripts\activate
python 01_traditional_opencv.py --category bottle
```

### 跑全部 15 个类别（慢，先别做）

```cmd
python 01_traditional_opencv.py --category all
```

### 输出长这样

```
类别            IoU  Precision    Recall    误检率%   漏检率%
------------------------------------------------------------------
bottle         0.xxx    0.xxx    0.xxx        xx.x      xx.x

结果已保存：outputs/01_traditional/metrics.json
可视化图片：outputs/01_traditional/
```

**把表格里的 IoU / 误检率 / 漏检率记下来**，这就是简历里那几个 `__` 要填的数字。

可视化图在 `outputs/01_traditional/`，每张图从左到右是：原图 | Otsu 分割 | 你的预测 | 真实标注。面试时可以直接展示。

---

## 六、常见问题

**Q：下载要多久？**
4.9 GB，100 Mbps 宽带理论约 7 分钟，实际受 MVTec 服务器带宽限制，通常 20–60 分钟。用下载工具挂着就行。

**Q：解压要多久？**
全量解压 5 GB 大概 10–30 分钟（取决于硬盘，SSD 快很多）。只解 bottle 类几十秒。

**Q：一定要下全量 15 类吗？**
不用。**先只解 bottle 一个类**跑通流程。等你确认脚本能跑、拿到数字了，再决定要不要下其他的。

**Q：学校网/公司网下不动怎么办？**
换手机热点，或试试国内数据集镜像站（HyperAI 超神经等社区有收录 `mvtec_anomaly_detection.tar.xz`，4.9 GB，许可证 CC BY-SA 4.0）。注意核验来源。

**Q：解压报 "路径太长" 错误？**
Windows 有 260 字符路径限制。把数据集放在**浅一点的路径**，比如 `D:\data\`，别放在好几层嵌套的文件夹里。

**Q：跑出来误检率特别高（比如 50%+）正常吗？**
正常。传统方法（Otsu + 自适应阈值 + Blob）在真实 MVTec 数据上表现本来就一般——这正是你要的面试素材：**"传统方法在这类数据上误检率偏高，主要来自光照不均和正常纹理被误判，这也是我理解到打光比算法更重要的原因"**。有对比才有故事。

**Q：许可证能商用吗？**
MVTec AD 是 **CC BY-SA 4.0**：可以免费使用（包括商业用途），但**必须署名**，且衍生作品要用相同协议。你做项目和面试展示完全没问题。

---

## 七、时间账（给你个预期）

| 步骤 | 耗时 |
|---|---|
| 填表 + 等链接 | 5–30 分钟（可能要等邮件） |
| 下载 4.9 GB | 20–60 分钟（挂后台） |
| 解压 bottle 类 | < 1 分钟 |
| 跑 01 脚本 | < 1 分钟 |
| **合计（只做 bottle）** | **约 1–1.5 小时**（大部分在等下载） |

> 下载挂着的时候，别干等——去把简历里的电话邮箱填了，那是目前优先级更高的事。
