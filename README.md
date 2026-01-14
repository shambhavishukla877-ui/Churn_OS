🧠 Churn OS 2.0 – Intelligent Customer Churn & Complaint Insight Engine

Enterprise-Grade Cognitive AI Platform

🔍 Overview

Churn OS is a full-stack enterprise AI platform that predicts customer churn, explains why customers are leaving, and analyzes complaint severity using advanced NLP.

This system is designed as a real-world production architecture, not just a machine-learning model.

It answers four critical business questions:

Question	Answered By
Who is going to leave?	Churn Prediction Engine
Why will they leave?	SHAP Explainable AI
How angry is the customer?	NLP Severity Engine
What is the root cause?	Keyword Extraction


⚙️ Core Capabilities
Layer	Feature
Prediction Brain	Random Forest / Neural Network Champion
Explainable AI	SHAP Local Explanations
NLP Intelligence	TF-IDF + Logistic Regression Severity Classifier


Dashboard	Streamlit Visual Interface
Backend API	FastAPI Secure Brain
Security	API-Key Enterprise Authentication
Monitoring Ready	CloudWatch / Logs Ready
Active Learning	Error Export Module


🏗️ System Architecture
User / Manager
      ↓
Streamlit Dashboard (UI)
      ↓
FastAPI Secure API  (X-API-Key Auth)
      ↓
------------------------------
|  Preprocessor             |
|  Churn Model (AI Brain)   |
|  SHAP Explainer           |
|  NLP Severity Engine      |
------------------------------
      ↓
Insights + Risk + Root Causes



📁 Folder Structure
src/
 ├── main.py                → Secure FastAPI API
 ├── dashboard.py           → Streamlit UI
 ├── data_pipeline.py       → Data Engineering
 ├── train_baseline.py      → Logistic Regression Baseline
 ├── train_advanced.py     → RF / XGBoost Champion Trainer
 ├── train_neural_net.py    → Deep Learning Model
 ├── train_nlp.py           → NLP Severity Engine
 ├── intelligence_engine.py → Explainability + NLP Brain
 ├── error_analyzer.py      → Active Learning Module
 └── config.py              → Central Configuration
data/
 ├── raw/
 └── processed/


🚀 Local Execution
Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Train AI Models
python src/data_pipeline.py
python src/train_advanced.py
python src/train_nlp.py

Step 3: Run Backend
python src/main.py

Step 4: Run Dashboard
streamlit run src/dashboard.py

🔐 Security Layer

All APIs are protected by:

Header:  X-API-Key : sk-proj-churn-secure-2026-v1


Unauthorized requests are blocked.

🌐 Cloud Deployment Strategy (Assignment Requirement)
AWS Production Architecture
User
 ↓
Route53 DNS
 ↓
Application Load Balancer
 ↓
EC2 (FastAPI + Gunicorn)
 ↓
S3 Bucket (Models & Logs)
 ↓
CloudWatch Monitoring
 ↓
IAM Security Policies

🧱 Deployment Flow
Layer	AWS Service
Compute	EC2
Load Balancing	ALB
Storage	S3
DNS	Route53
Monitoring	CloudWatch
Security	IAM + API Key
UI Hosting	Streamlit on EC2
Scaling	Auto Scaling Group


    📈 Enterprise Upgrades
Feature	Status
Deep Learning Champion	✔
Interaction Features	✔
SHAP Explainability	✔
NLP Severity Brain	✔
Security Layer	✔
Error Export Module	✔
Cloud-Ready	✔


    🔮 Future Scope
Phase	   Upgrade
Phase 8	   Dockerization
Phase 9	   PostgreSQL Database
Phase 10   LLM Auto Email Engine


      👨‍💻 Credits
Role	            Name
Project Architect	Shambhavi Shukla
