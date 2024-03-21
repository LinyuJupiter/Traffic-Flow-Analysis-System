# Traffic-Flow-Analysis-System

<p align="center">
  <a href="./README_en.md">English</a> |
  <a href="./README.md">简体中文</a>
</p>


本项目是一个使用 Yolov8 模型构建的车流量分析系统。

<img src="./animated.gif" alt="演示图片">

## 特性

- **可识别车型**: 可以识别小轿车、货车、公交车等车型。
- **可统计车流量**: 可以分别统计来往两方向的车流量情况。
- **可识别车牌**: 对于清晰的视频可以识别车牌。
- **更多功能**: 还在开发中。

## 安装

要运行此项目，您需要安装 Python 3.8 或更高版本。首先克隆此仓库：

```bash
git clone https://github.com/LinYujupiter/Traffic-Flow-Analysis-System.git
cd Traffic-Flow-Analysis-System
```

## 环境配置

### 使用 Conda

如果您使用 Conda，可以按照以下步骤设置和激活虚拟环境：

1. **创建虚拟环境**：

   ```bash
   conda create -n Traffic-Flow-Analysis-System python=3.8
   ```

2. **激活虚拟环境**：

   ```bash
   conda activate Traffic-Flow-Analysis-System
   ```

3. **安装依赖**：

   在激活的虚拟环境中，运行：

   ```bash
   pip install -r requirements.txt
   ```

### 不使用 Conda

如果您不使用 Conda，可以直接使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

## 运行

在安装了所有依赖之后，您可以通过以下命令启动程序：

```bash
python3 GUI.py
```

## 使用

启动程序后，您可以通过出现的UI界面指引来操作。

## 开发

- **ultralytics**: 用于调用Yolov8模型。
- **Tesser act-OCR**: 用于识别文字。
- **tkinter**: 用于构建UI界面。

## 贡献

我们欢迎任何形式的贡献，无论是新功能的提议、代码改进还是问题报告。请确保遵循最佳实践和代码风格指南。
