
# 🌿 Weed Detection Using YOLOv8 and ESP32-CAM

This is an AI-based **real-time weed detection system** built for precision agriculture. It uses a trained **YOLOv8 model** to distinguish between **weeds and chilli crops** from live video feed captured by an **ESP32-CAM** module. The goal is to automate weed identification to improve crop yield and reduce herbicide usage.

---

## 🚀 Features

- 🧠 Deep Learning-based object detection using YOLOv8
- 📷 Real-time video streaming via ESP32-CAM
- 🌱 Differentiates between weed and chilli crop
- 💻 Model inference runs on PC/laptop
- 🔌 USB/Serial communication for image transfer
- 🎯 High-accuracy detection using custom dataset

---

## 🧠 Model Details

- Architecture: **YOLOv8n**
- Framework: **Ultralytics YOLOv8**
- Dataset: Custom dataset (chilli & weed images)
- Labeling: Done with **Roboflow**
- Training platform: Google Colab

---

## 🛠️ Tech Stack

| Component        | Technology                      |
|------------------|----------------------------------|
| 🖥️ Detection Model | YOLOv8 (PyTorch, Ultralytics)     |
| 📷 Camera         | ESP32-CAM                       |
| 📡 Communication  | HTTP (or serial via USB)        |
| 🐍 Language       | Python                          |
| 📦 Packages       | OpenCV, Ultralytics, Flask (optional for web) |


---

## 🔌 How It Works

1. 📷 ESP32-CAM captures a frame (stream or snapshot).
2. 💻 The image is sent to the Python-based detection system.
3. 🧠 YOLOv8 detects and classifies objects (weed or chilli).
4. 🎯 Results are displayed on screen with bounding boxes.

---


## 🎓 Project Applications

- Smart farming and precision agriculture
- Automated weeding system
- Herbicide reduction
- Crop health monitoring

