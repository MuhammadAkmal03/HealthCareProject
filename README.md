# Pulse AI — Multimodal Healthcare Platform

A full-stack AI application I built to explore what it looks like when multiple healthcare models — symptom analysis, medical imaging, RAG-based Q&A, and report summarization — are wired together into a single deployable product.

This isn't just a notebook project. It's containerized, CI/CD automated, and deployed on AWS, with a live frontend on Vercel.

<!-- 🌐 **Live Demo:** [health-care-project-five.vercel.app](https://health-care-project-five.vercel.app) -->
**Project Documentation:** [Document](https://drive.google.com/file/d/1DpXyu6-mMow_ADb0Z35rJE5UWOujLZNo/view?usp=sharing)

---

## Key Features

- **Multimodal AI** — Three independent models (ML, DL, LLM) working together in one platform: symptom classification, chest X-ray analysis, and conversational Q&A
- **End-to-End MLOps** — Automated CI/CD via GitHub Actions: lint → test → Docker build → deploy to AWS EC2, all on every push
- **RAG Pipeline** — AI assistant grounded in a private medical knowledge base (Pinecone vector store + Gemini 2.5 Flash), not just a generic chatbot
- **Report Summarizer** — Upload a PDF or paste medical text; Gemini extracts and explains key clinical findings in plain language
- **Decoupled Architecture** — FastAPI backend on AWS EC2, static frontend on Vercel — independently scalable and deployable
- **Live Analytics Dashboard** — SQLite logs every prediction and AI query; Chart.js visualizes real-time trends and top query topics
- **Model Versioning** — Large model files tracked with Git LFS, keeping the repo fast without losing artifact history
- **Automated Testing** — pytest suite covering all API endpoints with coverage reporting

---

## What It Does

| Module | Model / Stack | Result |
|:---|:---|:---|
| **Symptom Checker** | LightGBM classifier — 5 disease classes | **92% accuracy**  |
| **X-Ray Analyzer** | TensorFlow CNN — Pneumonia vs Normal | **93% overall**  |
| **AI Medical Assistant** | RAG — Gemini 2.5 Flash + LangChain + Pinecone | Grounded answers from a private medical knowledge base |
| **Report Summarizer** | Gemini 2.5 Flash — PDF and raw text input | Extracts key clinical insights in plain language |
| **Analytics Dashboard** | SQLite + Chart.js + Tableau | Real-time prediction trends and top AI query topics |

---

## Model Performance

### Symptom Checker — LightGBM 

Classifies between 5 conditions: **Bronchitis, Cold, Flu, Healthy, Pneumonia**

```
               precision    recall  f1-score   
   Bronchitis       0.90      0.88      0.89        
         Cold       0.92      0.93      0.92        
          Flu       0.97      0.89      0.93        
      Healthy       0.92      0.93      0.92        
    Pneumonia       0.87      0.96      0.91        

     accuracy                           0.92       
    macro avg       0.92      0.92      0.91     
 weighted avg       0.92      0.92      0.92  
```

### X-Ray Analyzer — CNN 

Binary classification: **Pneumonia vs Normal** on chest X-rays (150×150 input)

```
              precision    recall  f1-score   
      NORMAL       0.88      0.87      0.88       
   PNEUMONIA       0.95      0.96      0.95       

    accuracy                           0.93       
   macro avg       0.92      0.91      0.92       
weighted avg       0.93      0.93      0.93       
```

---

## Tech Stack

**Backend** — FastAPI, LightGBM, TensorFlow/Keras, LangChain, Pinecone, HuggingFace (`all-MiniLM-L6-v2` embeddings), Gemini 2.5 Flash, SQLite, Docker, AWS EC2 + ECR

**Frontend** — HTML5, CSS3, JavaScript, Chart.js, Tableau — deployed on Vercel

**MLOps** — GitHub Actions (CI/CD), Git LFS for model versioning, pytest for endpoint testing, `black` + `ruff` for code quality

---

## MLOps Pipeline

Every `git push` to `main` triggers the following:

1. Linting and formatting checks (`black`, `ruff`)
2. Unit tests via `pytest` covering all API endpoints
3. Docker image build (minimal base image)
4. Automated deploy to AWS EC2 via SSH — pulls from ECR, restarts container

Models (`.h5`, `.joblib`) are tracked with **Git LFS** to keep the repo clean. Frontend and backend are fully decoupled — FastAPI on AWS, static frontend on Vercel.

---

## Architecture

```
Frontend (Vercel)
    ├── index.html             → Landing page
    ├── mlprediction.html      → Symptom Checker UI
    ├── dlprediction.html      → X-Ray Analyzer UI
    ├── aiassist.html          → AI Assistant + Report Summarizer
    └── trendchart.html        → Analytics Dashboard

Backend (FastAPI on AWS EC2)
    ├── /predict               → LightGBM symptom classifier
    ├── /analyze               → TensorFlow CNN image classifier
    ├── /assistant/chat        → RAG conversational chain
    ├── /assistant/summarize   → Gemini-powered report summarizer
    └── /assistant/query_topics → Analytics — top AI query topics

Data Layer
    └── SQLite (predictions.db)
        ├── predictions        → Diagnosis logs for trend chart
        └── chatbot_queries    → Query topics for analytics chart
```

---

## Running Locally

**Prerequisites:** Python 3.10+, Docker Desktop, Gemini and Pinecone API keys

```bash
# 1. Clone
git clone https://github.com/MuhammadAkmal03/HealthCareProject
cd HealthCareProject

# 2. Add your API keys to .env 

# 3. Build and run
docker build -t pulse-ai-api .
docker run -d --name pulse-ai-local -p 8000:8000 --env-file .env pulse-ai-api
```

- **API Docs (auto-generated):** `http://localhost:8000/docs`
- **Frontend:** Open `frontend/index.html` in your browser

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# AWS (required for deployment only)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
ECR_REPO=your_ecr_repository_url
EC2_HOST=your_host
EC2_USERNAME=username
EC2_SSH_KEY=ssh_key
```

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/
pytest --cov=app tests/  
```

---

## What I'd Add Next

- User authentication + prediction history
- More imaging models (CT scans, MRI)
- A/B testing for model comparison
- Multi-language support
- Expanded symptom coverage

---

*Built by [Muhammad Akmal](https://github.com/MuhammadAkmal03)*
