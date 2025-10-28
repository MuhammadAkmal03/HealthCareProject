# **Pulse AI: End-to-End Multimodal Healthcare Platform**

A production-ready application demonstrating a complete **MLOps lifecycle** for preliminary medical diagnostics and data analysis.

---

## ** Project Overview**

Pulse AI successfully integrates multiple state-of-the-art AI technologies into a single, cohesive web platform. The core achievement is moving three complex, data-intensive models into a single, efficient, and automated cloud environment.

### **Key Achievements**

| Feature | Technology & Contribution | MLOps Practice Demonstrated |
|:---|:---|:---|
| **Symptom Checker (ML)** | LightGBM classifier with **92% accuracy** . | **Data Preprocessing Pipelines** (`joblib`, `ColumnTransformer`) for clean real-time inference. |
| **Image Analyzer (DL)** | TensorFlow CNN for X-ray analysis | **Model Optimization & Efficiency** |
| **AI Medical Assistant** | **Retrieval-Augmented Generation (RAG)** using **Gemini** and **LangChain** / **Pinecone** vector store. | **LLM Grounding** and using **Vector Databases** for private knowledge. |
| **Report Summarizer** | AI-powered medical report summarization using **Gemini LLM** to extract key insights from user-provided medical documents. | **Natural Language Processing** and **Document Understanding** for clinical text analysis. |
| **Data Analytics** | Live monitoring dashboard visualizing **real-time prediction trends** and **top 5 user queries** (via Chart.js). Integrated Tableau dashboard for comprehensive dataset overview and exploratory data analysis. | **Data Logging & Monitoring** of operational metrics. |

---

## ** MLOps & Deployment Pipeline**

The entire backend deployment is automated to ensure reliability, maintainability, and quick iteration.

### **1. CI/CD Workflow (GitHub Actions)**

The pipeline is triggered on every `git push` to the main branch and performs the following sequence:

1. **Code Quality Check:** Runs `black` and `ruff` checks.
2. **Unit Testing:** Executes **`pytest`** to validate all API endpoints for the Symptom Checker, Image Analyzer, and AI Assistant.
3. **Containerization:** Builds a new, optimized **Docker** image (using a minimal base image).
4. **Deployment:** Securely connects via SSH to the **AWS EC2** instance, pulls the new image from **Amazon ECR**, and restarts the container with zero downtime.

### **2. Security & Artifact Management**

* **Model Versioning:** Large model files (`.h5`, `.joblib`) are stored and managed using **Git LFS** (Large File Storage) to keep the main repository clean and fast.
* **Decoupled Architecture:** The application uses a separated backend (FastAPI/Docker on AWS) and a frontend (HTML/JS on Vercel) for better scalability and flexibility.

---

## ** Getting Started (Local Setup)**

To run the full backend and frontend on your machine:

### **Prerequisites**

* Python 3.10+
* Docker Desktop (Installed and Running)
* AWS CLI (Installed and Configured)
* Gemini and Pinecone API Keys

### **Installation & Execution**

1. **Clone the Repository:**

```bash
git clone https://github.com/MuhammadAkmal03/HealthCareProject
cd HealthCareProject
```

2. **Create/Update `.env`:** Add your secret API keys to a file named `.env` in the root directory.

3. **Build the Docker Image:**

```bash
docker build -t pulse-ai-api .
```

4. **Run the Container:**

```bash
docker run -d --name pulse-ai-local -p 8000:8000 --env-file .env pulse-ai-api
```

* **Access API Docs:** `http://localhost:8000/docs`
* **Access Frontend:** Open the `frontend/index.html` file in your browser.

---

## ** Architecture & Technology Stack**

### **Backend (AWS EC2)**
* **Framework:** FastAPI
* **ML/DL Models:** LightGBM, TensorFlow/TFLite
* **RAG Stack:** LangChain, Gemini, Pinecone
* **Containerization:** Docker
* **Container Registry:** Amazon ECR

### **Frontend (Vercel)**
* **Tech Stack:** HTML5, CSS3, JavaScript
* **Visualization:** Chart.js
* **Deployment:** Vercel (Serverless)

### **DevOps & MLOps**
* **CI/CD:** GitHub Actions
* **Version Control:** Git, Git LFS
* **Cloud Provider:** AWS (EC2, ECR)
* **Monitoring:** Custom logging and analytics dashboard

---

### **Production Features**

*  **Zero-downtime deployments** using Docker container orchestration
*  **Automated testing** with pytest for all API endpoints
*  **Real-time analytics** dashboard for monitoring usage patterns
*  **Secure API key management** using environment variables
*  **CORS configuration** for secure cross-origin requests
*  **Scalable architecture** with decoupled frontend and backend

---

## ** Environment Configuration**

Create a `.env` file in the project root with the following variables:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# AWS Configuration (for deployment)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
ECR_REPO=your_ecr_repository_url
EC2_HOST=your_host
EC2_USERNAME=username
EC2_SSH_KEY=ssh_key

```

---

## ** Testing**

Run the test suite to ensure all components are working correctly:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=app tests/
```

---

## ** Future Enhancements**

- [ ] Add more medical imaging models (CT scans, MRI)
- [ ] Implement user authentication and history tracking
- [ ] Expand symptom checker with more diseases
- [ ] Add multi-language support
- [ ] Integrate real-time model performance monitoring
- [ ] Implement A/B testing for model improvements

---

## ** Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

