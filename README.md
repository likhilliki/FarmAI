# 🌿 Crop Disease Detection AI

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)](https://ultralytics.com/)

---

## 🏆 Overview
This is a **real-time plant disease detection app** using **YOLOv8** and **Streamlit**.  
It allows detection of diseases in leaves of **13 plant species (~30 disease classes)** using the **PlantDoc dataset**.

**Features:**
- Upload images or use webcam for detection  
- Bounding boxes with disease class & confidence  
- Suggested treatment for each disease  
- Detection history gallery  
- Public deployment ready  

---

## 🌱 Supported Plant Species & Disease Classes

**Plant Species (13):**  
Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Rice, Soybean, Squash, Strawberry, Tomato, Raspberry  

**Sample Disease Classes (~30):**
- Tomato: Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Target Spot, Spider Mites, Healthy  
- Potato: Early Blight, Late Blight, Healthy  
- Apple: Scab, Rust, Healthy  
- Pepper: Bacterial Spot, Healthy  
- Cherry: Powdery Mildew, Healthy  
- Grape: Black Rot, Esca, Leaf Blight, Healthy  
- Peach: Bacterial Spot, Healthy  
- Rice: Leaf Blast, Brown Spot, Healthy  
- Soybean: Healthy, Mosaic Virus  
- Squash: Powdery Mildew  
- Strawberry: Leaf Scorch, Healthy  
- Raspberry: Leaf Spot, Healthy  

---

## 📂 Dataset
- **Dataset:** PlantDoc 2  
- **YOLO Format:** Images + labels  
- **Folder structure:**
```text
PlantDoc-2/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
## ⚙️ Installation

### Clone the repo
```bash
git clone <your-repo-url>
cd crop-disease-detection
### create env
```bash
conda create -n crop python=3.10 -y
conda activate crop
### Install dependencies
```bash
pip install -r requirements.txt
### requirements.txt includes:
```text
torch
ultralytics
streamlit
opencv-python
numpy
Pillow
pyngrok

### 🚀 Running the App
Start Streamlit
```bash
streamlit run app.py

###🧠 Usage

Upload Images: Detect one or multiple leaf images

Webcam Detection: Real-time detection using your camera

View Results: Bounding boxes, disease class, and confidence score

Treatment Advice: Suggested treatment for detected diseases

History Gallery: Browse all previous detections


Powered by YOLOv8, Streamlit, and PlantDoc dataset
