from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import threading
import os
import re
import torch
import json
from datetime import datetime
from pymongo import MongoClient
from src.baseline_model import BaselineModel
from src.model_LSTM import MedicalModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TRANSFORMER_PATH = "models/medical_model"
LSTM_PREFIX      = "models/current_model"
BASELINE_FILE    = "models/baseline_model.pkl"
DATASET_FILE     = "data/medical_dataset.csv"

status = {"current_task": "idle"}
START_TIME = datetime.now()

# ==========================================
# 🍃 الإتصال بـ MongoDB وتغذية البيانات
# ==========================================
try:
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["medical_db"]
    history_collection  = db["predictions_history"]
    diseases_collection = db["diseases"]
    models_collection   = db["system_models"]
    stats_collection    = db["system_stats"]
    print("✅ Connected to MongoDB successfully!")

    # 1. تغذية بيانات الأمراض
    if os.path.exists(DATASET_FILE) and diseases_collection.count_documents({}) == 0:
        df_dataset = pd.read_csv(DATASET_FILE)
        records = df_dataset.to_dict(orient='records')
        diseases_collection.insert_many(records)
        print("📥 Medical Dataset loaded into MongoDB!")

except Exception as e:
    print(f"⚠️ MongoDB Connection Error: {e}")

# ==========================================
# تحميل موديل الترانسفورمر في الميموري
# ==========================================
abs_transformer_path = os.path.abspath(TRANSFORMER_PATH)

if os.path.exists(abs_transformer_path):
    print("⏳ Loading Transformer into Memory...")
    transformer_tok = AutoTokenizer.from_pretrained(abs_transformer_path)
    transformer_mdl = AutoModelForSequenceClassification.from_pretrained(abs_transformer_path)
    transformer_mdl.eval()
    
    labels_file = os.path.join(abs_transformer_path, "labels.json")
    with open(labels_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    transformer_id2label = {int(v): k for k, v in label_map.items()}
    transformer_status = True
    print("✅ Transformer Loaded Successfully!")
else:
    transformer_tok = None
    transformer_mdl = None
    transformer_id2label = {}
    transformer_status = False
    print("⚠️ Transformer Folder Not Found!")

# ==========================================
# 🔄 2. حفظ وحالة الموديلات المتاحة في الداتابيز (Models Info)
# ==========================================
def sync_system_models():
    try:
        available_models = [
            {
                "model_id": "baseline",
                "name": "Baseline Naive Bayes / TF-IDF",
                "accuracy": "97.98%",
                "status": "Active" if os.path.exists(BASELINE_FILE) else "Offline",
                "last_checked": datetime.now()
            },
            {
                "model_id": "lstm",
                "name": "Deep Learning LSTM",
                "accuracy": "98.79%",
                "status": "Active" if os.path.exists(f"{LSTM_PREFIX}_model.pth") or os.path.exists(LSTM_PREFIX) else "Offline",
                "last_checked": datetime.now()
            },
            {
                "model_id": "transformer",
                "name": "BioBERT Transformer",
                "accuracy": "99.99%",
                "status": "Active" if transformer_status else "Offline",
                "last_checked": datetime.now()
            }
        ]
        models_collection.delete_many({}) # تحديث البيانات القائمة
        models_collection.insert_many(available_models)
        print("📊 System Models status updated in MongoDB!")
    except Exception as e:
        print(f"Failed to sync models: {e}")

sync_system_models()


def clean_symptoms(text):
    text = re.sub(
        r"The patient experiences?|Patient (has|reports?|experiences?)",
        "", text, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def get_disease_details(condition_name):
    try:
        disease_info = diseases_collection.find_one({"condition": {"$regex": f"^{condition_name}$", "$options": "i"}})
        if disease_info:
            return {
                "warnings":        disease_info.get('warnings', 'No specific warnings found.'),
                "recommendations": disease_info.get('recommendations', 'No specific recommendations found.'),
                "causes":          disease_info.get('causes', 'Information not available.')
            }
    except Exception as e:
        print(f"Error fetching from MongoDB: {e}")

    return {
        "warnings":        "Data not found.",
        "recommendations": "Data not found.",
        "causes":          "Data not found."
    }


@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Medical Diagnosis System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #0f172a; color: white; padding: 20px; font-family: 'Segoe UI', Tahoma, sans-serif; }
        .card { background: #1e293b; border: none; color: white; margin-bottom: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .prediction-item { background: #2d3748; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #3182ce; }
        .detail-section { font-size: 0.9rem; margin-top: 10px; padding: 10px; background: #1a202c; border-radius: 5px; }
        .text-warning-custom { color: #f6ad55; font-weight: bold; }
        .text-success-custom { color: #68d391; font-weight: bold; }
        .risk-badge-high { background-color: #e53e3e; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8rem; font-weight: bold; }
        .risk-badge-normal { background-color: #38a169; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8rem; font-weight: bold; }
        hr { border-color: #4a5568; }
    </style>
</head>
<body>
<div class="container">
    <h2 class="text-center mb-4 text-info">🩺 AI Medical Diagnosis Assistant</h2>
    <div class="card p-4">
        <h5 class="text-primary">👤 Patient Information & Symptoms</h5><hr>
        
        <div class="row mb-3">
            <div class="col-md-3">
                <label class="form-label text-secondary">Age</label>
                <input type="number" id="age" class="form-control" placeholder="e.g. 25" min="1" max="120">
            </div>
            <div class="col-md-3">
                <label class="form-label text-secondary">Gender</label>
                <select id="gender" class="form-select">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <div class="col-md-6">
                <label class="form-label text-secondary">Model Choice</label>
                <select id="predictModel" class="form-select">
                    <option value="baseline">Baseline (97.98%)</option>
                    <option value="lstm">LSTM (98.79%)</option>
                    <option value="transformer" selected>Transformer (99.99%)</option>
                </select>
            </div>
        </div>

        <div class="row mb-3">
            <div class="col-12">
                <label class="form-label text-secondary">Symptoms</label>
                <textarea id="symptoms" class="form-control" rows="3" placeholder="Enter symptoms in English..."></textarea>
            </div>
        </div>

        <button onclick="predict()" class="btn btn-primary w-100">Analyze & Save Patient Record</button>
        <div id="results" class="mt-4"></div>
    </div>
</div>
<script>
    function predict() {
        const resDiv = document.getElementById('results');
        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;
        const symptoms = document.getElementById('symptoms').value;

        if (!symptoms.trim()) {
            alert('Please enter symptoms!');
            return;
        }

        resDiv.innerHTML = '<div class="text-center text-secondary">Analyzing & Saving Record...</div>';
        
        fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({
                age: age || "N/A",
                gender: gender,
                model_type: document.getElementById('predictModel').value,
                symptoms: symptoms
            })
        }).then(r=>r.json()).then(data => {
            if (data.error) { resDiv.innerHTML=`<div class="alert alert-danger">${data.error}</div>`; return; }
            
            let riskHTML = data.triage_flag === "HIGH RISK" 
                ? `<span class="risk-badge-high">🚨 HIGH RISK TRIAGE</span>` 
                : `<span class="risk-badge-normal">✅ NORMAL TRIAGE</span>`;

            let h = `<div class="d-flex justify-content-between align-items-center mb-3">
                        <h5>Results for <span class="text-info">${data.patient_info.gender}, ${data.patient_info.age} y/o</span> using <span class="text-warning">${data.model_used}</span>:</h5>
                        ${riskHTML}
                     </div>`;
            
            data.results.forEach(res => {
                h += `
                <div class="prediction-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <strong class="text-info" style="font-size:1.2rem">${res.condition}</strong>
                        <span class="badge bg-primary">${(res.probability*100).toFixed(1)}%</span>
                    </div>
                    <div class="detail-section">
                        <p><span class="text-warning-custom">⚠️ Warnings:</span> ${res.details.warnings}</p>
                        <p><span class="text-success-custom">💡 Recommendations:</span> ${res.details.recommendations}</p>
                        <p><small class="text-secondary">🔬 Possible Causes: ${res.details.causes}</small></p>
                    </div>
                </div>`;
            });
            resDiv.innerHTML = h;
        });
    }
</script>
</body>
</html>
""")


@app.route('/predict', methods=['POST'])
def predict_route():
    data     = request.json
    age      = data.get('age', 'N/A')
    gender   = data.get('gender', 'N/A')
    m_type   = data['model_type']
    symptoms = clean_symptoms(data['symptoms'])

    try:
        raw_results = []

        if m_type == "baseline":
            m = BaselineModel()
            if m.load(BASELINE_FILE):
                raw_results = m.predict(symptoms)
            else:
                return jsonify({"error": "Baseline model file not found"}), 400

        elif m_type == "lstm":
            m = MedicalModel()
            m.load(LSTM_PREFIX)
            raw_results = m.predict(symptoms)

        elif m_type == "transformer":
            if transformer_mdl is None or transformer_tok is None:
                return jsonify({"error": "Transformer model is not loaded in memory!"}), 400
            
            inputs = transformer_tok(
                symptoms, 
                return_tensors="pt",
                truncation=True, 
                padding=True,
                max_length=64
            )
            
            with torch.no_grad():
                outputs = transformer_mdl(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                top3 = torch.topk(probs, min(3, len(transformer_id2label)))
            
            raw_results = [
                {
                    "condition": transformer_id2label[int(top3.indices[i])],
                    "probability": float(top3.values[i])
                }
                for i in range(len(top3.indices))
            ]

        final_results = [
            {
                "condition":   res['condition'],
                "probability": res['probability'],
                "details":     get_disease_details(res['condition'])
            }
            for res in raw_results
        ]

        top_condition = final_results[0]['condition'] if final_results else "Unknown"
        top_prob = final_results[0]['probability'] if final_results else 0.0

        # 💡 3. الخيانة المبتكرة: نظام التصنيف الطبي والخطورة (Triage Flag & Clinical Priority)
        critical_conditions = ["heart attack", "stroke", "covid-19", "pneumonia", "appendicitis"]
        is_critical = any(c in top_condition.lower() for c in critical_conditions)
        
        try:
            age_num = int(age)
        except:
            age_num = 0

        triage_flag = "NORMAL"
        if (is_critical and top_prob > 0.70) or age_num > 65:
            triage_flag = "HIGH RISK"

        # 🍃 حفظ السجل الكامل في history_collection
        try:
            history_collection.insert_one({
                "patient_age": age,
                "patient_gender": gender,
                "symptoms_input": symptoms,
                "model_used": m_type,
                "top_prediction": top_condition,
                "top_probability": top_prob,
                "triage_risk": triage_flag,  # الميزة الذكية
                "all_results": final_results,
                "timestamp": datetime.now()
            })
        except Exception as mongo_err:
            print(f"Failed to save history to MongoDB: {mongo_err}")

        # 📊 4. تحديث إحصائيات النظام الشاملة (System Stats)
        try:
            stats_collection.update_one(
                {"system_id": "main_server"},
                {
                    "$inc": {"total_predictions": 1},
                    "$set": {
                        "last_prediction_time": datetime.now(),
                        "server_uptime_start": START_TIME
                    }
                },
                upsert=True
            )
        except Exception as stats_err:
            print(f"Failed to update stats: {stats_err}")

        return jsonify({
            "model_used": m_type, 
            "patient_info": {"age": age, "gender": gender},
            "triage_flag": triage_flag,
            "results": final_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)