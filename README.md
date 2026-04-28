# 🚨 Deep Learning for Comment Toxicity Detection with Streamlit

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Domain](https://img.shields.io/badge/Domain-NLP%20%7C%20Content%20Moderation-green)

---

## 📌 Project Overview

Online communities face significant challenges due to toxic comments including harassment, hate speech, and offensive language. This project builds a **real-time comment toxicity detection system** using Deep Learning (Bidirectional LSTM) and deploys it as an interactive **Streamlit web application**.

Users can type a comment or upload a CSV file and instantly receive toxicity predictions across 6 categories.

---

## 🎯 Problem Statement

Automated systems are needed to detect and flag toxic comments in real-time to assist platform moderators. This model analyzes text input from online comments and predicts the likelihood of each comment being toxic across multiple categories.

---

## 💼 Business Use Cases

- **Social Media Platforms** — Auto-detect and filter toxic comments in real-time
- **Online Forums & Communities** — Moderate user-generated content efficiently
- **Content Moderation Services** — Enhance moderation capabilities at scale
- **Brand Safety** — Ensure ads appear in safe, appropriate environments
- **E-learning Platforms** — Create safer online learning environments
- **News Websites** — Moderate user comments on articles and posts

---

## 🗂️ Project Structure

```
Comment_Toxicity_Detection_with_Streamlit/
│
├── data/
│   └── train.csv                  # Training dataset
│
├── models/
│   ├── toxicity_model.h5          # Trained Bi-LSTM model
│   └── tokenizer.pkl              # Fitted tokenizer
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py              # Text cleaning functions
│   └── predict.py                 # Single & batch prediction logic
│
├── c_t_app.py                     # Streamlit web application
├── toxicity_project.ipynb         # Model training notebook
├── eda_report.png                 # EDA visualization
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🧠 Model Architecture

- **Type:** Bidirectional LSTM (Deep Learning)
- **Embedding Dim:** 128
- **Max Vocab Size:** 50,000 tokens
- **Max Sequence Length:** 200
- **Architecture:**
  - Input → Embedding → BiLSTM (64 units) → GlobalMaxPooling → Dense(64, ReLU) → Dropout(0.3) → Dense(6, Sigmoid)
- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Output:** 6 toxicity category probabilities

### 🏷️ Toxicity Labels
| Label | Description |
|-------|-------------|
| `toxic` | General toxic content |
| `severe_toxic` | Severely toxic content |
| `obscene` | Obscene language |
| `threat` | Threatening language |
| `insult` | Insulting content |
| `identity_hate` | Hate speech targeting identity |

---

## 📊 Approach

### 1. Data Exploration & Preparation
- Loaded and explored the Kaggle Jigsaw toxicity dataset
- Performed EDA: label distribution, comment length analysis, clean vs toxic ratio
- Text preprocessing: lowercasing, URL removal, HTML tag removal, special character cleaning
- Tokenization and padding (maxlen=200)
- 90/10 Train-Validation split

### 2. Model Development
- Built a Bidirectional LSTM model using TensorFlow/Keras
- Used `EarlyStopping` and `ModelCheckpoint` callbacks
- Trained for up to 5 epochs with batch size 256
- Evaluated using AUC-ROC per label and classification report
- Saved model as `.h5` and tokenizer as `.pkl`

### 3. Streamlit Application
- Interactive UI for single comment prediction
- Bulk prediction via CSV file upload
- Displays toxicity scores, dominant label, and visual charts
- Real-time predictions using the saved model

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Manikandan-08J/Deep_Learning_for_Comment_Toxicity_Detection_with_Streamlit.git
cd Deep_Learning_for_Comment_Toxicity_Detection_with_Streamlit
```

### Step 2 — Create Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Streamlit App
```bash
streamlit run c_t_app.py
```

The app will open at: **http://localhost:8501**

---

## 📦 Requirements

```
tensorflow
streamlit
pandas
numpy
scikit-learn
plotly
matplotlib
pickle5
```

---

## 🚀 Usage

### Single Comment Prediction
1. Open the app at `http://localhost:8501`
2. Type or paste a comment in the text box
3. Click **Analyze** to get real-time toxicity predictions
4. View toxicity scores per category and the dominant label

### Bulk CSV Prediction
1. Prepare a CSV file with a column named `comment_text`
2. Upload it via the **Upload CSV** section
3. Download the results with toxicity scores for each comment

---

## 📈 Results

- ✅ Real-time interactive toxicity detection web application
- ✅ Multi-label classification across 6 toxicity categories
- ✅ Bulk prediction support via CSV upload
- ✅ EDA dashboard with label distribution and comment length analysis

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model |
| Bidirectional LSTM | Sequence modeling |
| Streamlit | Web app framework |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Plotly | Visualization |
| Scikit-learn | Evaluation metrics |

---

## 📁 Dataset

The dataset used is the **Jigsaw Toxic Comment Classification** dataset.

📥 [Download Dataset](https://drive.google.com/drive/folders/1WXLTp57_TYa61rcPfQIzRUcE1Rz76Emk?usp=sharing)

Place `train.csv` inside the `data/` folder before training.

---

## 👨‍💻 Author

**Manikandan**
- GitHub: [Manikandan-08J](https://github.com/Manikandan-08J)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
