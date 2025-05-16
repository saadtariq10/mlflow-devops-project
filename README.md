# MLflow DevOps Project

## 📌 Overview
This project demonstrates a complete MLflow-based machine learning pipeline with experiment tracking, model versioning, and deployment capabilities. It uses PyTorch for deep learning and provides a structured approach to managing the ML lifecycle.

## 🎯 Features
- Experiment tracking with MLflow
- Model training and versioning
- Model inference pipeline
- Comprehensive logging and metrics tracking
- Dataset management
- Reproducible ML workflows

## 🔧 Tech Stack
- **MLflow**: For experiment tracking and model management
- **PyTorch**: Deep learning framework
- **scikit-learn**: For data preprocessing and metrics
- **Python**: Primary programming language
- **Rich**: For enhanced terminal outputs
- **tqdm**: For progress tracking

## 📁 Project Structure
```
.
├── src/
│   ├── train_and_log.py    # Training pipeline with MLflow logging
│   └── inference.py        # Model inference implementation
├── dataset/                # Dataset storage
├── mlruns/                 # MLflow experiment tracking data
├── requirements.txt        # Project dependencies
└── .gitignore             # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Git
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/saadtariq10/mlflow-devops-project>
   cd devops_project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .\.venv\Scripts\activate
   # On Unix or MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training a Model
To train a model and log experiments with MLflow:
```bash
python src/train_and_log.py
```

### Running Inference
To perform inference using a trained model:
```bash
python src/inference.py
```

## 📊 MLflow Tracking
The project uses MLflow for experiment tracking. To view the MLflow UI:
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

## 📈 Experiment Tracking
- Model parameters and hyperparameters are tracked
- Training and validation metrics are logged
- Model artifacts are saved
- Experiment comparisons are available through MLflow UI

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
BSIT-8-Morning, NUML Islamabad

## 🙏 Acknowledgments
- MLflow team for the excellent experiment tracking framework
- PyTorch team for the deep learning framework
- All contributors and maintainers
