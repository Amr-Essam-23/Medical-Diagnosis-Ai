# 🩺 Medical AI Assistant: Multi-Model & Clinical Triage Diagnosis

An advanced deep learning-based medical diagnostic tool that processes clinical symptoms using **BioBERT**, **LSTM**, and **Baseline** models to predict conditions, feature patient triage risk assessment, and persist history using **MongoDB**.

## 🚀 Features
- **Multi-Model Diagnosis:** Uses BioBERT (Transformer), Bidirectional LSTM, and Naive Bayes models for symptom classification.
- **Patient Context & Triage Risk:** Accepts Patient Age & Gender, automatically tagging emergency high-risk conditions.
- **Persistent Storage:** Integrated MongoDB database to store disease data, prediction logs, model status, and usage statistics.
- **Interactive Dashboard:** Modern web interface built with Flask and Bootstrap for diagnosis, triage logging, and system control.
- **Google Colab Training:** Includes script for GPU training and seamless BioBERT weight extraction.

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning & DL:** PyTorch, Transformers (HuggingFace), TensorFlow/Keras, Scikit-learn
- **Natural Language Processing:** BioBERT, NLTK
- **Database:** MongoDB
- **Frontend & Web Scraping:** HTML5, Bootstrap 5, BeautifulSoup4, Requests

## 📂 Project Structure
- `app_all.py`: Main Flask application with the full UI and pre-loaded model inference.
- `src/model_LSTM.py`: LSTM model architecture and training logic.
- `src/baseline_model.py`: TF-IDF Naive Bayes baseline pipeline.
- `src/db_utils.py`: MongoDB interaction layer and statistics logging.
- `src/preprocess.py`: Text cleaning and NLP preprocessing pipeline.
- `scraper.py`: NHS web scraping tool for disease dataset collection.
- `medical_diagnosis_ai_colab.py`: Google Colab script for BioBERT model training.

## 📦 How to Setup Model Weights (BioBERT)
Due to GitHub's file size limits (>100MB), heavy model weights (`model.safetensors`) are trained separately:
1. Open `medical_diagnosis_ai_colab.py` in **Google Colab** with **GPU** enabled.
2. Run the notebook to train the BioBERT model and download the output files.
3. Place the downloaded `model.safetensors` file into the local directory: `models/medical_model/model.safetensors`.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Amr-Essam-23/Medical-Diagnosis-Ai.git](https://github.com/Amr-Essam-23/Medical-Diagnosis-Ai.git)
   cd Medical-Diagnosis-Ai
Install dependencies:

Bash
pip install -r requirements.txt
Ensure MongoDB is running:
The application requires a local MongoDB instance running at mongodb://localhost:27017/.

Run the application:

Bash
python app_all.py
Open http://127.0.0.1:5000 in your browser.

📖 How to Use
Input Context: Enter patient details (Age, Gender, Symptoms).

Predict: Click "Get Diagnosis" to analyze symptoms using the BioBERT Transformer.

Triage & History: View predicted condition, confidence score, triage flag (NORMAL/HIGH RISK), and check MongoDB history logs.

Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
