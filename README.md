# 🏥 Medical VQA System

**Knowledge-Guided Explainable Vision-Language Transformer for Medical Visual Question Answering**

A production-level Medical VQA system using **Qwen3-VL-8B** with QLoRA fine-tuning, RAG-based knowledge integration, and multi-modal explainability (Grad-CAM + Attention Visualization).

---

## 🏗️ Architecture

```
medical_vqa/
├── config.yaml              # Central configuration
├── requirements.txt          # Dependencies
├── Dockerfile               # GPU-enabled Docker build
├── docker-compose.yml       # Compose deployment
│
├── data/                    # Dataset module
│   ├── dataset.py           # VQA-RAD + PathVQA loaders
│   ├── augmentation.py      # Image augmentation pipeline
│   └── download_datasets.py # Dataset download script
│
├── knowledge/               # RAG module
│   ├── knowledge_base.py    # Curated medical knowledge (UMLS/SNOMED)
│   └── retriever.py         # FAISS vector search + sentence-transformers
│
├── model/                   # Model module
│   ├── model.py             # Qwen3-VL-8B with QLoRA wrapper
│   └── processor.py         # Multimodal input processor
│
├── training/                # Training module
│   ├── train.py             # Main training script
│   ├── trainer.py           # Custom HF Trainer + callbacks
│   └── curriculum.py        # Curriculum learning scheduler
│
├── explainability/          # Explainability module
│   ├── gradcam.py           # Grad-CAM heatmaps
│   ├── attention_viz.py     # Attention map visualization
│   └── explainer.py         # Unified explainability engine
│
├── evaluation/              # Evaluation module
│   ├── metrics.py           # Accuracy, BLEU, ROUGE, clinical checks
│   └── evaluate.py          # Full evaluation pipeline
│
└── api/                     # Inference API
    ├── server.py            # FastAPI server
    └── schemas.py           # Pydantic models
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd medical_vqa
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python -m medical_vqa.data.download_datasets --output_dir ./data
```

### 3. Train the Model

```bash
# Dry run (verify setup without training)
python -m medical_vqa.training.train --config medical_vqa/config.yaml --dry-run

# Full training on GPU
python -m medical_vqa.training.train --config medical_vqa/config.yaml
```

### 4. Evaluate

```bash
python -m medical_vqa.evaluation.evaluate \
    --config medical_vqa/config.yaml \
    --model-path outputs/final_model
```

### 5. Start API Server

```bash
uvicorn medical_vqa.api.server:app --host 0.0.0.0 --port 8000
```

### 6. Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@chest_xray.jpg" \
  -F "question=Is there cardiomegaly?"
```

---

## ⚙️ Configuration

All settings are in `config.yaml`:

| Section | Key Settings |
|---------|-------------|
| **Model** | `Qwen/Qwen3-VL-8B`, Flash Attention 2, BF16 |
| **QLoRA** | 4-bit NF4, double quantization |
| **LoRA** | rank=16, alpha=32, targets: q/k/v/o_proj |
| **Training** | batch=4, grad_accum=8, LR=1e-4, cosine decay |
| **RAG** | top-k=3, all-MiniLM-L6-v2 embeddings |
| **Curriculum** | 3 stages: easy→medium→hard |

---

## 🧠 Key Features

### RAG Knowledge Integration
- **45+ curated medical facts** from UMLS/SNOMED CT
- **FAISS vector search** for fast semantic retrieval
- **Automatic knowledge injection** into model prompts

### Explainability
- **Grad-CAM heatmaps** showing which image regions influenced the answer
- **Attention map visualization** for image-text alignment
- **Text explanations** grounded in medical knowledge

### Training Optimizations (H100)
- **QLoRA** (4-bit quantization + LoRA rank 16)
- **Flash Attention 2** for fast attention computation
- **BF16 mixed precision** training
- **Gradient checkpointing** for memory efficiency
- **Curriculum learning** (easy → hard questions)
- **7-hour time limit** with auto-stop

### Evaluation
- Exact match + fuzzy match accuracy
- BLEU/ROUGE scores for explanations
- Clinical consistency checks
- Confusion matrix + error analysis

---

## 🐳 Docker Deployment

### Build & Run

```bash
docker compose up --build
```

### AWS EC2 Deployment

1. Launch EC2 instance with GPU (e.g., `p4d.24xlarge` for H100, or `g5.xlarge` for A10G)
2. Install NVIDIA drivers + Docker with GPU support
3. Clone the repo and run:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Deploy
docker compose up -d
```

4. Test: `curl http://<EC2_IP>:8000/health`

---

## 📊 API Response Format

```json
{
    "answer": "Yes, there is cardiomegaly.",
    "confidence": 0.92,
    "explanation": "The cardiac silhouette appears enlarged with a cardiothoracic ratio exceeding 0.5, consistent with cardiomegaly. This may be associated with congestive heart failure or valvular disease.",
    "visual_heatmap": "<base64-encoded-PNG>",
    "knowledge_used": "Cardiomegaly is an enlarged heart, typically defined as a cardiothoracic ratio greater than 0.5...",
    "processing_time_ms": 1245.67
}
```

---

## 📋 Requirements

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.2.0 with CUDA
- **GPU**: NVIDIA H100 (80GB) recommended; A100/A10G also supported
- **VRAM**: ~20GB (4-bit quantized inference), ~40GB (training)

---

## 📄 License

For research and educational purposes. Medical AI systems require proper clinical validation before deployment in healthcare settings.
