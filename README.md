# ✈️ 基于 Qwen3-VL 的飞机方位判断系统

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

本项目通过微调 **Qwen3-VL-8B** 多模态大语言模型，实现从单张飞机照片中**自动推断相机相对于飞机的方位**（8 分类：正前方、右前方、正右方、右后方、正后方、左后方、正左方、左前方）。

系统可广泛应用于无人机视觉引导、航空态势感知、飞行器自主着陆等场景。

---

## 📌 项目亮点

- 🧠 **多模态空间推理**：利用视觉语言模型理解 2D 图像中的 3D 空间关系
- 🔧 **高效微调**：基于 LoRA 技术，仅训练 **~1%** 的参数即可达到优异性能
- 🚀 **一键部署**：提供 Gradio Web 界面，支持上传图片实时推理
- 🎯 **高精度**：真实测试集准确率 **88.5%**，相邻类别误差 **<10%**

---

## 📁 项目结构

    .
    ├── app.py # Gradio Web 推理入口
    ├── requirements.txt # Python 依赖
    ├── inference.sh # 快速推理脚本（可选）
    ├── train.sh # 微调脚本（可选）
    ├── loras/ # LoRA 适配器权重（已微调的模型部分）
    ├──example_image/ #数据集示例
    ├── models/ # 基础模型（需自行下载，不包含在本仓库）
    ├── mq_9b/ # 原始数据集（不包含在本仓库）
    └── README.md # 项目说明

text

> **注意**：由于文件体积较大，完整模型权重及数据集未包含在仓库，可以根据自己的情况获取。

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- CUDA 11.8+（推荐 32GB 显存，最低 24 GB）
- Git LFS（如需从 HuggingFace 下载模型）

### 2. 安装依赖
```bash
git clone https://github.com/zhaoxiaotai/qwen-aircraft-orientation.git
cd qwen-aircraft-orientation
python3 -m venv qwen3_vl # 也可以用conda创建
source qwen3_vl/bin/activate # Linux/macOS；Windows 使用 `qwen3_vl\Scripts\activate`
pip install -r requirements.txt
```
### 3. 获取模型权重

本项目微调得到的 LoRA 适配器 已包含在 loras/ 目录中。你需要搭配基础模型 Qwen/Qwen3-VL-8B-Instruct 使用。

1. 方式一：从 HuggingFace 下载
    ```bash
    huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/Qwen3-VL-8B-Instruct
    ```

2. 方式二：手动下载基础模型（国内推荐）
    ```bash
    pip install modelscope
    modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct
    ```
然后将 app.py 中的 BASE_MODEL_PATH 改为对应的本地路径（或保持 Qwen/Qwen3-VL-8B-Instruct 在线加载）。
### 4. 训练以及推理
- 训练
    ```bash
    train.sh
    ```
- 推理
  ```bash
  inference.sh
  ```
### 5. 运行 Web 服务
```bash
python app.py
```
启动后，在本地浏览器打开 http://localhost:7860 即可上传图片测试。

如需远程访问（如服务器部署），可设置 server_name="0.0.0.0" 并使用 SSH 隧道转发。

## 🧠 技术细节
### 数据构建
利用已有的 3D 坐标（飞机位置、相机位置、机头/机尾三维坐标）自动计算相对方位角，并离散化为 8 个类别。每张图片对应一个对话样本：

```json
{
  "conversations": [
    {"from": "human", "value": "判断相机相对于飞机的方位"},
    {"from": "gpt", "value": "右前方"}
  ],
  "image": "/path/to/aircraft.png"
}
```

数据集总量：5,000+ 条

训练/验证集划分：90% / 10%

### 模型微调
- 基础模型：Qwen/Qwen3-VL-8B-Instruct
- 微调框架：ms-swift（或 LLaMA-Factory）
- 微调方法：LoRA（r=8, alpha=32）
- 训练参数：学习率：1e-4
- 批量大小：2（梯度累积 4）
- 训练轮数：5
- 最大图像 token 数：1024（对应 max_pixels=1024*28*28）

### 模型性能
| 指标 | 数值 |
|:------:|:------:|
| 测试集准确率 (Top-1) | **88.5%** |
| 相邻类别误差率 | **< 10%** |
| 推理延迟 (V100) | **~2-3 秒/张** |

## 🖥️ 部署与演示
### 本地部署
```bash
python app.py
```

### 服务器部署（SSH 隧道）
1. 在服务器上运行 python app.py（监听 0.0.0.0:7860）

2. 在本地终端建立隧道：
    ```bash
    ssh -CNg -L 7860:127.0.0.1:7860 user@server_ip -p port
    ```
3. 浏览器访问 http://127.0.0.1:7860


### 🔮 未来计划
- 支持连续角度回归（0°~360°）

- 增加距离估计和俯仰角预测

- 部署到边缘设备（Jetson Orin / Raspberry Pi）

- 集成视频流实时推理

### 🙏 致谢
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 模型团队

- [ms-swift](https://github.com/modelscope/ms-swift) 微调框架

- [Gradio](https://gradio.org.cn/guides/quickstart) 快速界面库

### 📄 许可证
- 本项目采用 [MIT License]() 开源。


### ⭐ Star 支持
如果本项目对你有帮助，欢迎点个 Star ⭐ 支持一下！
