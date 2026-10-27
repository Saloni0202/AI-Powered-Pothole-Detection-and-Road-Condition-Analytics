# ◈ RoadSight AI
### AI-Powered Pothole Detection and Road Condition Analytics

**RoadSight AI** is a computer vision project that uses a trained **YOLO11n object detection model** to identify potential potholes in road images. It provides an interactive Streamlit application where users can upload an image, view detected potholes with bounding boxes and confidence scores, and download the annotated result.

The goal is simple: explore how AI can support road inspection by helping people identify potential road damage more efficiently.

## ✨ Features

- **Pothole Detection:** Detect potential potholes using a trained YOLO11n model.
- **Interactive Demo:** Upload road images and view predictions in real time.
- **Adjustable Confidence Threshold:** Control which detections are displayed.
- **Visual Results:** View bounding boxes, detection labels, confidence scores, and approximate inference time.
- **Evaluation Evidence:** Explore training metrics, curves, and confusion matrix visualisations when the corresponding artifacts are available.
- **Methodology & Responsible Use:** Learn how the model works, understand its limitations, and explore possible future improvements.

## 🖥️ Application Preview

The application is organised into five sections:

| Section | What it covers |
|---|---|
| **Purpose** | The road safety problem and motivation behind the project |
| **Live Demo** | Upload an image and inspect pothole detections |
| **Method** | The model, dataset format, and detection pipeline |
| **Evidence** | Training metrics and evaluation visualisations |
| **Responsible Use** | Limitations, practical considerations, and future scope |

## ⚙️ How It Works

The application follows a straightforward computer vision pipeline:

1. **Input:** Upload a road image in JPG, JPEG, PNG, or WEBP format.
2. **Preprocessing:** The image is loaded and converted to RGB format.
3. **Inference:** The trained `best.pt` checkpoint processes the image using Ultralytics YOLO.
4. **Detection Filtering:** Predictions below the selected confidence threshold are excluded.
5. **Visualisation:** The application displays bounding boxes, labels, and confidence scores.
6. **Output:** Review the detections, inspect approximate inference time, and download the annotated image.

```text
Road Image
    ↓
Image Preprocessing
    ↓
Trained YOLO11n Model (best.pt)
    ↓
Pothole Predictions
    ↓
Confidence Threshold
    ↓
Annotated Image + Detection Scores
    ↓
Download Results
```

## 🧠 Model & Training

The project uses **YOLO11n (Nano)**, a lightweight object detection model suited to responsive inference and demonstration applications.

The model was fine-tuned using Ultralytics and PyTorch on a **3,000-image subset** of a Kaggle pothole dataset containing 13,767 images.

| Training detail | Value |
|---|---|
| Model | YOLO11n |
| Framework | Ultralytics, PyTorch |
| Dataset subset | 3,000 images |
| Maximum configured epochs | 250 |
| Batch size | 4 |
| Selected checkpoint | `best.pt` |

The trained checkpoint is included in this repository. The Streamlit application loads these saved weights to make predictions on new images; it does not retrain the model when the application starts.

### Reported Evaluation Metrics

| Metric | Reported value |
|---|---:|
| Precision | 84.6% |
| Recall | 75.0% |
| mAP@50 | 82.6% |
| mAP@50–95 | 57.3% |

These are the project's reported evaluation results. The **Evidence** section can display metrics and visualisations from the saved training outputs when the corresponding files are included.

## 🛠️ Tech Stack

- **Python** — application logic and model integration
- **YOLO11n / Ultralytics** — object detection and inference
- **PyTorch** — deep learning framework
- **OpenCV & Pillow** — image processing and visualisation support
- **Streamlit** — interactive web application

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your GitHub repository URL and folder name.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, upload a road image, and explore the detections.

**Important:** Keep `best.pt` in the repository root beside `app.py`, or place it in one of the model locations supported by the application.

## 📁 Project Structure

```text
RoadSight-AI/
├── .streamlit/       # Streamlit configuration
├── assets/           # Supporting visuals and evaluation artifacts
├── app.py            # Streamlit application
├── best.pt           # Trained YOLO11n checkpoint
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## ⚠️ Limitations & Responsible Use

- Predictions depend on image quality, lighting, viewpoint, and the road surface.
- Confidence scores indicate model confidence, not guaranteed correctness.
- The model identifies visual pothole candidates; it does not measure pothole depth or determine road safety.
- Inference time varies with the available hardware and model warm-up.
- Results should be reviewed by a person before being used to inform road maintenance decisions.

## 🔭 Future Scope

- Extend detection to road videos and dashcam footage.
- Add GPS-based location tagging where appropriate.
- Build a verified pothole reporting and maintenance workflow.
- Evaluate performance across different road types, weather conditions, and lighting environments.

## 👩‍💻 Project Information

**Project:** RoadSight AI – AI-Powered Pothole Detection and Road Condition Analytics  
**Author:** Saloni Maijorwar  

---

*RoadSight AI is a computer vision prototype designed to demonstrate AI-assisted pothole detection and visual analysis. It supports human inspection rather than replacing professional road assessments.*
