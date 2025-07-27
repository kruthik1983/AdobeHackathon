# PDF Outline Extractor using Machine Learning

A containerized ML-powered tool to extract structured outlines (titles, headings, body text) from PDF documents. It outputs clean, hierarchical JSON files and detailed CSV logs for analysis and retraining.

---

## 🔧 Features

- 🧠 **ML-Based Classification** — Uses Random Forest and Gradient Boosting to detect document structure.
- 🔁 **Iterative Learning** — Improves over time with manual labeling and retraining.
- 🧩 **Feature-Rich Extraction** — Extracts 20+ features like font size, style, position, and spacing.
- 🐳 **Dockerized** — Easily deployable and runs consistently across environments.
- 📄 **Structured Output** — Outputs clean JSON and detailed CSV reports.

---

## 🚀 Quickstart

### ✅ Prerequisites

- Python 3.9+
- Docker (for containerized use)

### 🛠️ Local Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### 📌 Run the Pipeline

1. **Add PDFs**: Place files into `input/Pdf/`.

2. **Execute**:
   ```bash
   python -m src.main
   ```

3. **Label New Data**: Open `dataset/pdf_analyze.csv`, fill the `level` column (e.g., `Title`, `H1`, `Body`), and save.

4. **Retrain & Re-run**:
   ```bash
   python -m src.main
   ```

---

## 🐳 Docker Usage

### 🔨 Build Image

```bash
docker build -t pdf-outline-extractor .
```

### ▶️ Run Container

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-outline-extractor
```

---

## 📦 Output

- `output/example.json`: Final structured JSON
- `output/example_font_report.csv`: Font style diagnostics
