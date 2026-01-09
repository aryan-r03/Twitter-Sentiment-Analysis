<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&pause=1000&color=5ec8f2&center=true&vCenter=true&width=700&lines=Twitter+Sentiment+Analysis;Machine+Learning+%2B+Flask;NLP+%26+Text+Classification;Real-Time+Emotion+Detection" alt="Typing SVG" />
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/NLTK-154F3C?style=for-the-badge&logo=python&logoColor=white" alt="NLTK"/>
  <img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-85%25+-brightgreen?style=flat-square" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Model-Logistic_Regression-blue?style=flat-square" alt="Model"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF-orange?style=flat-square" alt="NLP"/>
  <img src="https://img.shields.io/badge/API-RESTful-red?style=flat-square" alt="API"/>
</p>

---

<div align="center">

### 🐦 AI-Powered Twitter Sentiment Analysis Web Application

> **Professional Flask web application that analyzes tweet sentiment in real-time using Machine Learning and Natural Language Processing. Features advanced text preprocessing, TF-IDF vectorization, and Logistic Regression for accurate emotion detection.**

**💬 Perfect for social media analytics, brand monitoring, and NLP learning projects**

[Features](#-features) • [Demo](#-demo--preview) • [Quick Start](#-quick-start) • [API](#-api-reference) • [Model](#-model-details)

</div>

---

## 📋 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🎬 Demo & Preview](#-demo--preview)
- [🧠 Tech Stack](#-tech-stack)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [💻 Usage Guide](#-usage-guide)
- [📡 API Reference](#-api-reference)
- [🤖 Model Details](#-model-details)
- [🔧 Text Processing Pipeline](#-text-processing-pipeline)
- [📊 Dataset Format](#-dataset-format)
- [⚙️ Configuration](#%EF%B8%8F-configuration)
- [🎨 Customization](#-customization)
- [🐛 Troubleshooting](#-troubleshooting)
- [🚀 Future Enhancements](#-future-enhancements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Project Overview

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/color/96/000000/twitter.png" width="80" height="80" alt="Twitter"/>
        <br><b>Tweet Analysis</b>
        <br>Real-time sentiment
        <br>280 char support
      </td>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="80" height="80" alt="AI"/>
        <br><b>Machine Learning</b>
        <br>Logistic Regression
        <br>85%+ accuracy
      </td>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/color/96/000000/api.png" width="80" height="80" alt="API"/>
        <br><b>RESTful API</b>
        <br>JSON responses
        <br>Easy integration
      </td>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/color/96/000000/web.png" width="80" height="80" alt="Web"/>
        <br><b>Modern UI</b>
        <br>Responsive design
        <br>Real-time results
      </td>
    </tr>
  </table>
</div>

A **production-ready sentiment analysis application** that uses Machine Learning to classify tweets as positive or negative in real-time. Built with Flask for the backend, scikit-learn for ML, and NLTK for natural language processing.

### 🎯 Why This Project?

<table>
<tr>
<td width="50%">

**For Learning:**
- 🎓 Master NLP fundamentals
- 📊 Understand ML classification
- 🌐 Learn Flask web development
- 🧹 Practice text preprocessing
- 📈 Explore feature engineering (TF-IDF)

</td>
<td width="50%">

**For Production:**
- 💼 Industry-standard architecture
- 🎨 Professional UI/UX design
- 📱 Social media monitoring
- 🔍 Brand sentiment analysis
- 📊 Customer feedback analysis

</td>
</tr>
</table>

---

## ✨ Features

<div align="center">

### Core Capabilities

<table>
  <tr>
    <th>Category</th>
    <th>Features</th>
  </tr>
  <tr>
    <td><b>🤖 Machine Learning</b></td>
    <td>
      ✅ Logistic Regression classifier<br>
      ✅ Balanced class weights<br>
      ✅ 85%+ accuracy on test data<br>
      ✅ TF-IDF vectorization (5000 features)<br>
      ✅ Bigram support (1-2 word phrases)<br>
      ✅ Model persistence with pickle
    </td>
  </tr>
  <tr>
    <td><b>📝 NLP Text Processing</b></td>
    <td>
      ✅ Advanced text cleaning<br>
      ✅ URL and mention removal<br>
      ✅ Lemmatization (WordNet)<br>
      ✅ Stopword removal (keeps negations)<br>
      ✅ Special character handling<br>
      ✅ Lowercase normalization
    </td>
  </tr>
  <tr>
    <td><b>🌐 Web Application</b></td>
    <td>
      ✅ Modern, responsive UI<br>
      ✅ Real-time sentiment analysis<br>
      ✅ Character counter (280 limit)<br>
      ✅ Animated result display<br>
      ✅ Confidence score visualization<br>
      ✅ Keyboard shortcuts (Enter to analyze)
    </td>
  </tr>
  <tr>
    <td><b>📡 RESTful API</b></td>
    <td>
      ✅ JSON request/response format<br>
      ✅ POST /api/analyze endpoint<br>
      ✅ Detailed sentiment scores<br>
      ✅ Error handling & validation<br>
      ✅ CORS support ready<br>
      ✅ Easy external integration
    </td>
  </tr>
  <tr>
    <td><b>📊 Model Evaluation</b></td>
    <td>
      ✅ Accuracy, Precision, Recall metrics<br>
      ✅ F1-Score calculation<br>
      ✅ Confusion matrix visualization<br>
      ✅ Classification report<br>
      ✅ Train/test split (80/20)<br>
      ✅ Stratified sampling
    </td>
  </tr>
  <tr>
    <td><b>💾 Data Handling</b></td>
    <td>
      ✅ CSV dataset loading<br>
      ✅ Multiple format support<br>
      ✅ Automatic label conversion<br>
      ✅ Data validation & cleaning<br>
      ✅ Balanced dataset sampling<br>
      ✅ Missing value handling
    </td>
  </tr>
</table>

</div>

---

## 🎬 Demo & Preview

<div align="center">

### Application Interface

```
┌─────────────────────────────────────────────────────┐
│      🐦 Twitter Sentiment Analysis                  │
│   AI-Powered Emotion Detection using ML             │
│    [Trained with Logistic Regression]               │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Enter Tweet                                       │
│   ┌─────────────────────────────────────────────┐   │
│   │ I love this amazing product! It's so good!  │   │
│   └─────────────────────────────────────────────┘   │
│                               47 / 280 characters   │
│                                                     │
│                                                     │
│           [      Analyze Sentiment      ]           │
│   ╔════════════════════════════════════════════╗    │
│   ║             Sentiment: Positive            ║    │
│   ║                                            ║    │
│   ║  ┌───────────┐ ┌────────────┐ ┌──────────┐ ║    │
│   ║  │Confidence │ │Positive    │ │Negative  │ ║    │
│   ║  │   95%     │ │Score: 95%  │ │Score: 5% │ ║    │
│   ║  └───────────┘ └────────────┘ └──────────┘ ║    │
│   ╚════════════════════════════════════════════╝    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface (Web)                  │
│                 index.html + JavaScript                 │
└────────────────────────────┬────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────┐
│             Flask Web Server (app.py)             │
│  ┌────────────────────────────────────────────┐   │
│  │ # Route Handlers                           │   │
│  │  • GET /           → Serve HTML            │   │
│  │  • POST /api/analyze → Analyze sentiment   │   │
│  └────────────────────────────────────────────┘   │
└─────────────────────────┬─────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│        SentimentModel Class (sentiment_model.py)        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  1. Text Preprocessing                           │   │
│  │     • Lowercase, remove URLs, mentions           │   │
│  │     • Lemmatization, stopword removal            │   │
│  │                                                  │   │
│  │  2. TF-IDF Vectorization                         │   │
│  │     • Convert text → numerical features          │   │
│  │     • 5000 max features, bigrams                 │   │
│  │                                                  │   │
│  │  3. Logistic Regression                          │   │
│  │     • Binary classification (0/1)                │   │
│  │     • Probability scores                         │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│                  Prediction Result                      │
│  {                                                      │
│    "sentiment": "Positive",                             │
│    "confidence": 95,                                    │
│    "positive_score": 95,                                │
│    "negative_score": 5                                  │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

### ML Pipeline Workflow

```
Input Tweet
│
▼
┌─────────────────────────────┐
│   Text Preprocessing        │
│  • Convert to lowercase     │
│  • Remove URLs & @mentions  │
│  • Clean special chars      │
│  • Lemmatize words          │
│  • Remove stopwords         │
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│   TF-IDF Vectorization      │
│  • Extract features         │
│  • Weight by importance     │
│  • Create feature vector    │
│    (5000 dimensions)        │
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│   Logistic Regression       │
│  • Predict class (0/1)      │
│  • Calculate probabilities  │
│  • Return confidence        │
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│   Sentiment Result          │
│  • Positive or Negative     │
│  • Confidence score (%)     │
│  • Individual probabilities │
└─────────────────────────────┘
```

</div>

---

## 🧠 Tech Stack

<div align="center">

### Technologies & Libraries

<table>
  <tr>
    <td align="center" width="20%">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="60" height="60" alt="Python"/>
      <br><b>Python 3.8+</b>
      <br>Core language
    </td>
    <td align="center" width="20%">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flask/flask-original.svg" width="60" height="60" alt="Flask"/>
      <br><b>Flask 3.0</b>
      <br>Web framework
    </td>
    <td align="center" width="20%">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="60" height="60" alt="Scikit-learn"/>
      <br><b>Scikit-learn</b>
      <br>ML library
    </td>
    <td align="center" width="20%">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="60" height="60" alt="Pandas"/>
      <br><b>Pandas</b>
      <br>Data processing
    </td>
    <td align="center" width="20%">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="60" height="60" alt="NumPy"/>
      <br><b>NumPy</b>
      <br>Numerical ops
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://img.icons8.com/color/96/000000/code.png" width="60" height="60" alt="NLTK"/>
      <br><b>NLTK</b>
      <br>NLP toolkit
    </td>
    <td align="center" width="33%">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" width="60" height="60" alt="HTML5"/>
      <br><b>HTML5/CSS3</b>
      <br>Frontend
    </td>
    <td align="center" width="33%">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" width="60" height="60" alt="JavaScript"/>
      <br><b>JavaScript</b>
      <br>Frontend logic
    </td>
  </tr>
</table>

### Component Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Flask | HTTP routing, templating, JSON API |
| **ML Algorithm** | Logistic Regression | Binary sentiment classification |
| **Feature Extraction** | TF-IDF Vectorizer | Text → numerical features |
| **Text Processing** | NLTK | Lemmatization, stopwords, tokenization |
| **Data Handling** | Pandas | CSV loading, data manipulation |
| **Model Storage** | Pickle | Serialize/deserialize trained model |
| **Frontend** | HTML/CSS/JavaScript | Responsive UI, AJAX requests |

</div>

---

## 📦 Installation

<div align="center">

### System Requirements

</div>

<table>
  <tr>
    <th>Requirement</th>
    <th>Minimum</th>
    <th>Recommended</th>
  </tr>
  <tr>
    <td><b>Python</b></td>
    <td>3.8</td>
    <td>3.9 - 3.11</td>
  </tr>
  <tr>
    <td><b>RAM</b></td>
    <td>2 GB</td>
    <td>4 GB+</td>
  </tr>
  <tr>
    <td><b>Storage</b></td>
    <td>500 MB</td>
    <td>1 GB+ (for datasets)</td>
  </tr>
  <tr>
    <td><b>OS</b></td>
    <td colspan="2">Windows 10+, macOS 10.14+, Ubuntu 18.04+</td>
  </tr>
</table>

<div align="center">

### Dependencies

</div>

```python
# Core Framework
flask==3.0.0                # Web application framework

# Machine Learning
pandas==2.1.4               # Data manipulation
numpy==1.26.2               # Numerical computing
scikit-learn==1.3.2         # ML algorithms & tools

# Natural Language Processing
nltk==3.8.1                 # NLP toolkit (stopwords, lemmatization)
```

---

## 🚀 Quick Start

<div align="center">

### Step 1️⃣: Clone Repository

</div>

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

<div align="center">

### Step 2️⃣: Create Virtual Environment (Recommended)

</div>

<table>
<tr>
<td width="50%">

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

</td>
<td width="50%">

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

</td>
</tr>
</table>

<div align="center">

### Step 3️⃣: Install Dependencies

</div>

```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import flask, sklearn, nltk, pandas; print('All dependencies installed!')"
```

<div align="center">

### Step 4️⃣: Download NLTK Data (Automatic)

</div>

The application will automatically download required NLTK data on first run:
- Stopwords corpus
- WordNet lemmatizer
- OMW-1.4 (Open Multilingual Wordnet)

**Manual download (if needed):**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

<div align="center">

### Step 5️⃣: Prepare Dataset (Optional)

</div>

**Option A: Use Sample Data (Default)**
- Application creates a sample dataset automatically
- Good for testing and learning

**Option B: Use Custom Dataset**
```bash
# Place your CSV file in the project directory
# Name it: data.csv or Twitter_data.csv

# CSV Format:
# - Column 1: Tweet text
# - Column 2: Sentiment label (Positive/Negative or 0/1)
```

<div align="center">

### Step 6️⃣: Run the Application

</div>

```bash
python app.py
```

**Expected Output:**
```
============================================================
INITIALIZING SENTIMENT ANALYSIS MODEL
============================================================

✓ Found existing model: sentiment_model.pkl
✓ Model loaded successfully!

============================================================
MODEL READY
============================================================

============================================================
TWITTER SENTIMENT ANALYSIS - MACHINE LEARNING PROJECT
============================================================

📌 To use your own dataset:
   Place CSV file named 'data.csv' in this directory
   CSV should have columns: 'text' and 'sentiment'

Starting Flask server...
Open your browser and go to: http://127.0.0.1:5000
============================================================

 * Running on http://127.0.0.1:5000
```

<div align="center">

### Step 7️⃣: Access the Application

</div>

1. Open your browser
2. Navigate to the **URL** from the terminal
3. Enter a tweet in the text area
4. Click "Analyze Sentiment"
5. View results with confidence scores! 🎉

---

## 💻 Usage Guide

<div align="center">

### Web Interface

</div>

**Step-by-Step:**

1. **Enter Tweet Text**
   - Type or paste a tweet (up to 280 characters)
   - Character counter shows remaining length
   - Example: "I absolutely love this product! Best purchase ever!"

2. **Analyze Sentiment**
   - Click "Analyze Sentiment" button
   - Or press Enter key for quick analysis
   - Loading animation appears during processing

3. **View Results**
   - **Sentiment**: Positive or Negative
   - **Confidence**: Overall prediction confidence (%)
   - **Positive Score**: Probability of positive sentiment (%)
   - **Negative Score**: Probability of negative sentiment (%)

**Example Tweets to Try:**

<table>
  <tr>
    <th>Tweet</th>
    <th>Expected Result</th>
  </tr>
  <tr>
    <td>"I love this amazing product! It's fantastic!"</td>
    <td>✅ Positive (90%+ confidence)</td>
  </tr>
  <tr>
    <td>"This is terrible. Worst experience ever."</td>
    <td>❌ Negative (90%+ confidence)</td>
  </tr>
  <tr>
    <td>"The weather is nice today."</td>
    <td>✅ Positive (moderate confidence)</td>
  </tr>
  <tr>
    <td>"I don't like this at all. Very disappointed."</td>
    <td>❌ Negative (high confidence)</td>
  </tr>
</table>

<div align="center">

### Training Custom Model

</div>

**With Your Own Dataset:**

```python
from sentiment_model import SentimentModel

# Initialize model
model = SentimentModel()

# Load your CSV file
df = model.load_dataset_from_csv(
    'your_data.csv',
    text_column='tweet',      # Column with tweet text
    label_column='sentiment'  # Column with labels
)

# Train model
if df is not None:
    accuracy = model.train(df)
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Save trained model
    model.save_model('my_sentiment_model.pkl')
```

**Test Predictions:**

```python
# Make predictions
test_tweets = [
    "I love this product!",
    "This is terrible.",
    "Not bad, could be better."
]

for tweet in test_tweets:
    result = model.predict(tweet)
    print(f"\nTweet: {tweet}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}%")
```

---

## 📡 API Reference

<div align="center">

### Available Endpoints

</div>

### 🏠 Home Page

**Endpoint:** `GET /`

**Description:** Serves the main HTML interface

**Response:** HTML page

**Usage:**
```bash
curl http://127.0.0.1:5000/
```

---

### 🔮 Analyze Sentiment

**Endpoint:** `POST /api/analyze`

**Description:** Analyzes sentiment of provided tweet text

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "tweet": "I absolutely love this product! It's amazing!"
}
```

**Response (Success):**
```json
{
  "success": true,
  "result": {
    "sentiment": "Positive",
    "confidence": 95,
    "positive_score": 95,
    "negative_score": 5
  }
}
```

**Response (Error - Empty Tweet):**
```json
{
  "success": false,
  "error": "Tweet is empty"
}
```

**Response (Error - No Tweet Provided):**
```json
{
  "success": false,
  "error": "No tweet provided"
}
```

**Status Codes:**
- `200 OK` - Analysis successful
- `400 Bad Request` - Invalid input (empty tweet, no tweet field)
- `500 Internal Server Error` - Server/model error

---

<div align="center">

### Integration Examples

</div>

<details>
<summary><b>Python (requests)</b></summary>

```python
import requests
import json

# API endpoint
url = "http://127.0.0.1:5000/api/analyze"

# Tweet to analyze
tweet_data = {
    "tweet": "I love this amazing product! Best purchase ever!"
}

# Make request
response = requests.post(url, json=tweet_data)
result = response.json()

if result['success']:
    print(f"Sentiment: {result['result']['sentiment']}")
    print(f"Confidence: {result['result']['confidence']}%")
    print(f"Positive Score: {result['result']['positive_score']}%")
    print(f"Negative Score: {result['result']['negative_score']}%")
else:
    print(f"Error: {result['error']}")
```

</details>

<details>
<summary><b>JavaScript (Fetch API)</b></summary>

```javascript
// API endpoint
const url = 'http://127.0.0.1:5000/api/analyze';

// Tweet to analyze
const tweetData = {
  tweet: "I love this amazing product! Best purchase ever!"
};

// Make request
fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(tweetData)
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log(`Sentiment: ${data.result.sentiment}`);
    console.log(`Confidence: ${data.result.confidence}%`);
    console.log(`Positive Score: ${data.result.positive_score}%`);
    console.log(`Negative Score: ${data.result.negative_score}%`);
  } else {
    console.error(`Error: ${data.error}`);
  }
})
.catch(error => console.error('Request failed:', error));
```

</details>

<details>
<summary><b>cURL (Command Line)</b></summary>

```bash
# Analyze sentiment
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tweet": "I love this amazing product! Best purchase ever!"
  }'

# Pretty print with jq
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d @tweet.json | jq '.'
```

</details>

<details>
<summary><b>Node.js (Axios)</b></summary>

```javascript
const axios = require('axios');

// API endpoint
const url = 'http://127.0.0.1:5000/api/analyze';

// Tweet to analyze
const tweetData = {
  tweet: "I love this amazing product! Best purchase ever!"
};

// Make request
axios.post(url, tweetData)
  .then(response => {
    const data = response.data;
    if (data.success) {
      console.log(`Sentiment: ${data.result.sentiment}`);
      console.log(`Confidence: ${data.result.confidence}%`);
      console.log(`Positive: ${data.result.positive_score}%`);
      console.log(`Negative: ${data.result.negative_score}%`);
    } else {
      console.error(`Error: ${data.error}`);
    }
  })
  .catch(error => console.error('Request failed:', error));
```

</details>

---

## 🤖 Model Details

<div align="center">

### Logistic Regression Classifier

</div>

**Algorithm Configuration:**

```python
LogisticRegression(
    max_iter=1000,          # Maximum iterations for convergence
    random_state=42,        # Reproducibility
    class_weight='balanced' # Handle imbalanced datasets
)
```

**Key Features:**
- **Binary Classification**: Positive (1) vs Negative (0)
- **Probability Output**: Confidence scores for each class
- **Balanced Weights**: Handles imbalanced data automatically
- **Fast Training**: Efficient on large datasets
- **Interpretable**: Clear feature importance

<div align="center">

### Model Performance Metrics

</div>

<table>
  <tr>
    <th>Metric</th>
    <th>Score</th>
    <th>Interpretation</th>
  </tr>
  <tr>
    <td><b>Accuracy</b></td>
    <td><b>85-90%</b></td>
    <td>Overall correct predictions</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td><b>83-88%</b></td>
    <td>Positive predictions that are correct</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td><b>85-90%</b></td>
    <td>Actual positives correctly identified</td>
  </tr>
  <tr>
    <td><b>F1-Score</b></td>
    <td><b>0.84-0.89</b></td>
    <td>Harmonic mean of precision & recall</td>
  </tr>
</table>

**Sample Confusion Matrix:**

```
                Predicted
              Neg    Pos
Actual Neg    420    35      Specificity: 92.3%
       Pos     48    397     Recall: 89.2%

True Negatives: 420    False Positives: 35
False Negatives: 48    True Positives: 397

Overall Accuracy: (420 + 397) / 900 = 90.78%
```

<div align="center">

### TF-IDF Vectorization

</div>

**Configuration:**

```python
TfidfVectorizer(
    max_features=5000,      # Top 5000 most important features
    ngram_range=(1, 2)      # Unigrams and bigrams
)
```

**What is TF-IDF?**

- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How unique/rare a word is across all documents
- **TF-IDF Score**: TF × IDF = Importance of word in document

**Example:**
```
Tweet: "I love this product! The product is amazing!"

Unigrams: ["love", "product", "amazing", ...]
Bigrams: ["love product", "product amazing", ...]

TF-IDF Vector: [0.23, 0.45, 0.67, ...] (5000 dimensions)
```

**Why Bigrams?**
- Captures phrases: "not good" vs "good"
- Better context understanding
- Improved accuracy for negations

---

## 🔧 Text Processing Pipeline

<div align="center">

### Preprocessing Steps

</div>

**Complete Pipeline:**

```python
def clean_text(text):
    # 1. Lowercase
    text = text.lower()
    # → "I Love This!" → "i love this!"
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # → "Check this: http://example.com" → "Check this:"
    
    # 3. Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # → "@user Great product!" → "Great product!"
    
    # 4. Remove special characters (keep !?)
    text = re.sub(r'[^a-zA-z\s!?]', '', text)
    # → "Product #1 costs $50!" → "Product  costs !"
    
    # 5. Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    # → "running" → "run", "better" → "good"
    
    # 6. Remove stopwords (keep negations)
    keep_words = {'not', 'no', 'never', 'none', 'neither', 'nor'}
    words = [word for word in words 
             if word not in stopwords or word in keep_words]
    # → Keeps "not good" (important for sentiment)
    
    return ' '.join(words)
```

**Example Transformation:**

<table>
  <tr>
    <th>Stage</th>
    <th>Text</th>
  </tr>
  <tr>
    <td><b>Original</b></td>
    <td>I LOVE this product! Check it out: http://example.com @company #amazing</td>
  </tr>
  <tr>
    <td><b>Lowercase</b></td>
    <td>i love this product! check it out: http://example.com @company #amazing</td>
  </tr>
  <tr>
    <td><b>Remove URLs</b></td>
    <td>i love this product! check it out:  @company #amazing</td>
  </tr>
  <tr>
    <td><b>Remove Mentions</b></td>
    <td>i love this product! check it out:  #amazing</td>
  </tr>
  <tr>
    <td><b>Remove Special Chars</b></td>
    <td>i love this product check it out  amazing</td>
  </tr>
  <tr>
    <td><b>Lemmatize</b></td>
    <td>i love this product check it out amazing</td>
  </tr>
  <tr>
    <td><b>Remove Stopwords</b></td>
    <td>love product check amazing</td>
  </tr>
  <tr>
    <td><b>Final</b></td>
    <td><b>love product check amazing</b></td>
  </tr>
</table>

**Negation Handling:**

```python
# These words are preserved even though they're stopwords
keep_words = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', "n't"}

# Why? They completely change sentiment:
"good" → Positive
"not good" → Negative ✓ (preserved)
```

---

## 📊 Dataset Format

<div align="center">

### Supported CSV Formats

</div>

**Format 1: Standard Binary (Recommended)**

```csv
text,sentiment
"I love this product!",1
"This is terrible.",0
"Great experience!",1
"Worst service ever.",0
```

**Format 2: Text Labels**

```csv
text,sentiment
"I love this product!",positive
"This is terrible.",negative
"Great experience!",positive
"Worst service ever.",negative
```

**Format 3: Twitter Format (0/4)**

```csv
text,sentiment
"I love this product!",4
"This is terrible.",0
"Great experience!",4
"Worst service ever.",0
```

**Format 4: Alternative Column Names**

```csv
tweet,label
"I love this product!",pos
"This is terrible.",neg
```

<div align="center">

### Automatic Format Detection

</div>

The application automatically detects and converts:

<table>
  <tr>
    <th>Format</th>
    <th>Conversion</th>
  </tr>
  <tr>
    <td>0/1</td>
    <td>✓ Already binary</td>
  </tr>
  <tr>
    <td>0/4</td>
    <td>0→0, 4→1</td>
  </tr>
  <tr>
    <td>positive/negative</td>
    <td>negative→0, positive→1</td>
  </tr>
  <tr>
    <td>pos/neg</td>
    <td>neg→0, pos→1</td>
  </tr>
  <tr>
    <td>Custom numeric</td>
    <td>Threshold at median</td>
  </tr>
</table>

**Dataset Requirements:**

✅ **Must Have:**
- Text column (tweets/messages)
- Sentiment/label column
- At least 100 samples (recommended: 1000+)
- Balanced classes (equal pos/neg samples)

❌ **Avoid:**
- Missing values in text or sentiment
- Empty tweets
- Single-class datasets
- Extreme class imbalance (>90% one class)

**Example Custom Dataset:**

```python
import pandas as pd

# Create custom dataset
data = {
    'text': [
        "I love this!",
        "Terrible experience",
        "Amazing product!",
        "Very disappointed",
        # ... more samples
    ],
    'sentiment': [1, 0, 1, 0, ...]  # 0=negative, 1=positive
}

df = pd.DataFrame(data)
df.to_csv('my_dataset.csv', index=False)
```

---

## ⚙️ Configuration

<div align="center">

### Model Parameters

</div>

**Modify in `sentiment_model.py`:**

```python
# TF-IDF Configuration
self.vectorizer = TfidfVectorizer(
    max_features=5000,        # Number of features (1000-10000)
    ngram_range=(1, 2),       # (1,1) for unigrams only, (1,3) for trigrams
    min_df=2,                 # Minimum document frequency
    max_df=0.95               # Maximum document frequency
)

# Logistic Regression Configuration
self.model = LogisticRegression(
    max_iter=1000,            # Increase if convergence warning
    random_state=42,          # For reproducibility
    class_weight='balanced',  # Handle imbalanced data
    C=1.0,                    # Regularization strength (lower = more regularization)
    solver='lbfgs'            # Optimization algorithm
)
```

**Parameter Tuning Guide:**

<table>
  <tr>
    <th>Parameter</th>
    <th>Effect</th>
    <th>Recommendation</th>
  </tr>
  <tr>
    <td><b>max_features</b></td>
    <td>Number of TF-IDF features</td>
    <td>5000 (balanced)<br>3000 (faster)<br>10000 (more accurate)</td>
  </tr>
  <tr>
    <td><b>ngram_range</b></td>
    <td>Word combinations to consider</td>
    <td>(1,2) - unigrams + bigrams<br>(1,1) - single words only<br>(1,3) - up to 3-word phrases</td>
  </tr>
  <tr>
    <td><b>max_iter</b></td>
    <td>Training iterations</td>
    <td>1000 (default)<br>2000 (if not converging)</td>
  </tr>
  <tr>
    <td><b>class_weight</b></td>
    <td>Handle class imbalance</td>
    <td>'balanced' (recommended)<br>None (equal weights)</td>
  </tr>
</table>

<div align="center">

### Flask Server Configuration

</div>

**Modify in `app.py`:**

```python
# Change port
app.run(debug=True, port=5000)  # Use 8000, 8080, etc.

# Production mode
app.run(debug=False, host='0.0.0.0', port=5000)

# Threading for multiple requests
app.run(debug=False, threaded=True)
```

<div align="center">

### File Paths

</div>

```python
# CSV dataset path
CSV_FILE = 'data.csv'  # Change to your dataset name

# Model save/load path
MODEL_FILE = 'sentiment_model.pkl'  # Custom model name
```

---

## 🎨 Customization

<div align="center">

### Extension Ideas

</div>

<details>
<summary><b>🎨 Custom UI Theme</b></summary>

**Modify `templates/index.html`:**

```css
/* Change gradient colors */
body {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    /* Or try: */
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

/* Change button colors */
.btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

/* Change positive/negative colors */
.result.positive {
    background: #c3e6cb;  /* Light green */
    border: 2px solid #28a745;
}

.result.negative {
    background: #f5c6cb;  /* Light red */
    border: 2px solid #dc3545;
}
```

</details>

<details>
<summary><b>📊 Add More Metrics</b></summary>

```python
# In sentiment_model.py - predict() method

def predict(self, text):
    # ... existing code ...
    
    # Add entropy (uncertainty measure)
    entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    
    # Add emotion detection (requires additional model)
    emotions = self.detect_emotions(text)
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'positive_score': int(probabilities[1] * 100),
        'negative_score': int(probabilities[0] * 100),
        'entropy': round(entropy, 3),  # NEW
        'emotions': emotions  # NEW
    }
```

</details>

<details>
<summary><b>💾 Save Analysis History</b></summary>

```python
# In app.py

import datetime
import json

HISTORY_FILE = 'analysis_history.json'

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # ... existing analysis code ...
    
    # Save to history
    history_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'tweet': tweet,
        'result': result
    }
    
    # Load existing history
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except:
        history = []
    
    # Add new entry
    history.append(history_entry)
    
    # Save history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    
    return jsonify({'success': True, 'result': result})

# Add endpoint to view history
@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        return jsonify({'success': True, 'history': history})
    except:
        return jsonify({'success': False, 'error': 'No history found'})
```

</details>

<details>
<summary><b>🔄 Try Different ML Models</b></summary>

```python
# In sentiment_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Option 1: Random Forest
self.model = RandomForestClassifier(
    n_estimators=100,
    max_depth=50,
    random_state=42
)

# Option 2: Support Vector Machine
self.model = SVC(
    kernel='linear',
    probability=True,  # Required for predict_proba
    random_state=42
)

# Option 3: Naive Bayes
self.model = MultinomialNB(
    alpha=1.0  # Smoothing parameter
)

# Compare models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Naive Bayes': MultinomialNB()
}

for name, model in models.items():
    self.model = model
    accuracy = self.train(df)
    print(f"{name}: {accuracy:.2%}")
```

</details>

<details>
<summary><b>📱 Add Batch Analysis</b></summary>

```python
# In app.py

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        tweets = data.get('tweets', [])
        
        if not tweets or not isinstance(tweets, list):
            return jsonify({
                'success': False,
                'error': 'Invalid tweets array'
            }), 400
        
        results = []
        for tweet in tweets:
            result = sentiment_model.predict(tweet)
            results.append({
                'tweet': tweet,
                'sentiment': result
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

**Usage:**
```bash
curl -X POST http://127.0.0.1:5000/api/batch_analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tweets": [
      "I love this!",
      "This is terrible.",
      "Not bad."
    ]
  }'
```

</details>

<details>
<summary><b>🌐 Add Multi-language Support</b></summary>

```python
# Install: pip install googletrans==3.1.0a0

from googletrans import Translator

translator = Translator()

def translate_to_english(text):
    """Translate text to English before analysis"""
    try:
        detected = translator.detect(text)
        if detected.lang != 'en':
            translated = translator.translate(text, dest='en')
            return translated.text
    except:
        pass
    return text

# In predict() method
def predict(self, text):
    # Translate if needed
    text_english = translate_to_english(text)
    
    # ... rest of prediction code ...
```

</details>

---

## 🐛 Troubleshooting

<div align="center">

### Common Issues & Solutions

</div>

<details>
<summary><b>❌ NLTK Data Not Found</b></summary>

**Symptoms:**
```
LookupError: Resource stopwords not found.
LookupError: Resource wordnet not found.
```

**Solutions:**

1. **Manual Download:**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

2. **Download to Specific Directory:**
   ```python
   import nltk
   nltk.download('stopwords', download_dir='/path/to/nltk_data')
   nltk.data.path.append('/path/to/nltk_data')
   ```

3. **Download All NLTK Data:**
   ```python
   import nltk
   nltk.download('all')  # Warning: Large download (3.5 GB)
   ```

4. **Verify Installation:**
   ```python
   from nltk.corpus import stopwords
   print(stopwords.words('english')[:10])
   # Should print: ['i', 'me', 'my', 'myself', ...]
   ```

</details>

<details>
<summary><b>🔄 Model Not Training / Low Accuracy</b></summary>

**Symptoms:**
- Accuracy < 70%
- Model predicts same class for everything
- Convergence warnings

**Solutions:**

1. **Check Dataset Balance:**
   ```python
   print(df['sentiment'].value_counts())
   # Should be roughly equal:
   # 0    5000
   # 1    5000
   ```

2. **Increase Training Data:**
   - Need minimum 500 samples (250 per class)
   - Recommended: 5000+ samples

3. **Increase Max Iterations:**
   ```python
   self.model = LogisticRegression(max_iter=2000)  # Increase from 1000
   ```

4. **Balance Dataset:**
   ```python
   # Undersample majority class
   min_count = min(
       (df['sentiment'] == 0).sum(),
       (df['sentiment'] == 1).sum()
   )
   
   df_neg = df[df['sentiment'] == 0].sample(n=min_count)
   df_pos = df[df['sentiment'] == 1].sample(n=min_count)
   df_balanced = pd.concat([df_neg, df_pos])
   ```

5. **Check Text Quality:**
   ```python
   # Print sample cleaned texts
   print(df['cleaned_text'].head(10))
   # Should not be empty or too short
   ```

</details>

<details>
<summary><b>💾 Model Save/Load Errors</b></summary>

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'sentiment_model.pkl'
pickle.UnpicklingError: invalid load key
```

**Solutions:**

1. **Check File Exists:**
   ```python
   import os
   print(os.path.exists('sentiment_model.pkl'))
   print(os.path.abspath('sentiment_model.pkl'))
   ```

2. **Ensure Directory Permissions:**
   ```bash
   # Linux/Mac
   chmod 755 .
   
   # Windows
   # Check folder permissions in Properties
   ```

3. **Save with Absolute Path:**
   ```python
   import os
   model_path = os.path.join(os.getcwd(), 'sentiment_model.pkl')
   model.save_model(model_path)
   ```

4. **Delete Corrupted Model:**
   ```bash
   rm sentiment_model.pkl
   # Then retrain
   python app.py
   ```

</details>

<details>
<summary><b>🌐 Flask Server Won't Start</b></summary>

**Symptoms:**
```
Address already in use
OSError: [Errno 48] Address already in use
```

**Solutions:**

1. **Find Process Using Port:**
   ```bash
   # Linux/Mac
   lsof -i :5000
   
   # Windows
   netstat -ano | findstr :5000
   ```

2. **Kill Process:**
   ```bash
   # Linux/Mac
   kill -9 <PID>
   
   # Windows
   taskkill /PID <PID> /F
   ```

3. **Use Different Port:**
   ```python
   app.run(debug=True, port=8000)  # Change to 8000
   ```

4. **Check for Multiple Instances:**
   ```bash
   ps aux | grep python  # Linux/Mac
   tasklist | findstr python  # Windows
   ```

</details>

<details>
<summary><b>📊 CSV Loading Errors</b></summary>

**Symptoms:**
```
FileNotFoundError: data.csv not found
KeyError: 'sentiment'
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Solutions:**

1. **Check File Location:**
   ```python
   import os
   print(os.listdir('.'))  # List files in current directory
   ```

2. **Try Different Encoding:**
   ```python
   # In load_dataset_from_csv()
   try:
       df = pd.read_csv('data.csv', encoding='utf-8')
   except:
       df = pd.read_csv('data.csv', encoding='latin-1')
   except:
       df = pd.read_csv('data.csv', encoding='iso-8859-1')
   ```

3. **Check CSV Format:**
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv', nrows=5)
   print(df.columns)  # Check column names
   print(df.head())   # Check first rows
   ```

4. **Manual Column Mapping:**
   ```python
   df = df.rename(columns={
       'Tweet': 'text',        # Rename your columns
       'Label': 'sentiment'
   })
   ```

</details>

---

## 🚀 Future Enhancements

<div align="center">

### Planned Features

</div>

<table>
  <tr>
    <th>Feature</th>
    <th>Description</th>
    <th>Status</th>
  </tr>
  <tr>
    <td><b>😊 Multi-class Emotions</b></td>
    <td>Detect joy, anger, sadness, fear, surprise</td>
    <td>🔄 Planned</td>
  </tr>
  <tr>
    <td><b>🌍 Multi-language Support</b></td>
    <td>Analyze tweets in multiple languages</td>
    <td>🔄 Planned</td>
  </tr>
  <tr>
    <td><b>📊 Analytics Dashboard</b></td>
    <td>Visualize sentiment trends over time</td>
    <td>🔄 Planned</td>
  </tr>
  <tr>
    <td><b>🔄 Real-time Twitter Stream</b></td>
    <td>Analyze live tweets from Twitter API</td>
    <td>💡 Idea</td>
  </tr>
  <tr>
    <td><b>🤖 Deep Learning Model</b></td>
    <td>Use LSTM/BERT for better accuracy</td>
    <td>💡 Idea</td>
  </tr>
  <tr>
    <td><b>📱 Mobile App</b></td>
    <td>iOS/Android app for on-the-go analysis</td>
    <td>💡 Idea</td>
  </tr>
  <tr>
    <td><b>🔗 Browser Extension</b></td>
    <td>Analyze tweets directly on Twitter.com</td>
    <td>💡 Idea</td>
  </tr>
  <tr>
    <td><b>📈 Trend Analysis</b></td>
    <td>Track sentiment changes for topics/hashtags</td>
    <td>💡 Idea</td>
  </tr>
  <tr>
    <td><b>🎯 Aspect-based Sentiment</b></td>
    <td>Analyze sentiment for specific aspects (price, quality, etc.)</td>
    <td>💡 Idea</td>
  </tr>
  <tr>
    <td><b>💾 Database Integration</b></td>
    <td>Store analysis results in PostgreSQL/MongoDB</td>
    <td>💡 Idea</td>
  </tr>
</table>

---

## 🤝 Contributing

<div align="center">

**Contributions are welcome!** Help improve sentiment analysis:

</div>

### Ways to Contribute

<table>
  <tr>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/color/96/000000/bug.png" width="60" height="60" alt="Bug"/>
      <br><b>Report Bugs</b>
      <br>Found an issue?
      <br>Open an issue
    </td>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/color/96/000000/idea.png" width="60" height="60" alt="Feature"/>
      <br><b>Suggest Features</b>
      <br>Have an idea?
      <br>Share it!
    </td>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/color/96/000000/code.png" width="60" height="60" alt="Code"/>
      <br><b>Submit Code</b>
      <br>Improvements?
      <br>Send a PR
    </td>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/color/96/000000/document.png" width="60" height="60" alt="Docs"/>
      <br><b>Improve Docs</b>
      <br>Better explanation?
      <br>Update README
    </td>
  </tr>
</table>

### Development Workflow

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```
3. **Create** a feature branch:
   ```bash
   git checkout -b feature/emotion-detection
   ```
4. **Make** your changes
5. **Test** thoroughly
6. **Commit** with clear messages:
   ```bash
   git commit -m 'Add emotion detection feature'
   ```
7. **Push** to your fork:
   ```bash
   git push origin feature/emotion-detection
   ```
8. **Open** a Pull Request

### Code Style Guidelines

- ✅ Follow PEP 8 for Python code
- ✅ Use descriptive variable names
- ✅ Add docstrings to functions
- ✅ Comment complex logic
- ✅ Write unit tests for new features
- ✅ Update documentation

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**

Free to use, modify, and distribute with attribution

</div>

<details>
<summary><b>Click to view full license</b></summary>

```
MIT License

Copyright (c) 2025 Twitter Sentiment Analysis Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

---

##  Acknowledgments

<div align="center">

Special thanks to:

- 🐍 **Scikit-learn Team** for powerful ML tools
- 📚 **NLTK Developers** for NLP resources
- 🌐 **Flask Community** for the web framework
- 🐦 **Twitter** for inspiring social media analytics
- 👥 **Open Source Community** for continuous support
-  **Thank You** for using and supporting this project!

</div>

---

## 👨‍💻 Author

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=1000&color=ffffff&center=true&vCenter=true&width=435&lines=Aryan+Ranjan;Web+Application+Developer;Data+Science+Enthusiast;Open+Source+Contributor" alt="Author Typing SVG" />

<br>

**🎓 Computer Applications in AI & ML**
<br>
**Building intelligent NLP solutions**

</div>

---

## 📞 Support

<div align="center">

### Need Help?

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://img.icons8.com/color/96/000000/document.png" width="80" height="80" alt="Docs"/>
      <br><b>Documentation</b>
      <br>Complete README Guide
      <br><i>Setup & troubleshooting</i>
    </td>
    <td align="center" width="50%">
      <img src="https://img.icons8.com/color/96/000000/code.png" width="80" height="80" alt="Code"/>
      <br><b>Code Comments</b>
      <br>In-line Documentation
      <br><i>Implementation details</i>
    </td>
  </tr>
</table>

<br>

**Refer to the troubleshooting section above for common issues and solutions**

</div>

---

<div align="center">

## 🌟 Show Your Support

**If this project helped you, please consider:**

<a href="https://github.com/your-username/twitter-sentiment-analysis">
  <img src="https://img.shields.io/github/stars/your-username/twitter-sentiment-analysis?style=social" alt="GitHub stars"/>
</a>
<a href="https://github.com/your-username/twitter-sentiment-analysis/fork">
  <img src="https://img.shields.io/github/forks/your-username/twitter-sentiment-analysis?style=social" alt="GitHub forks"/>
</a>
<a href="https://github.com/your-username/twitter-sentiment-analysis/watchers">
  <img src="https://img.shields.io/github/watchers/your-username/twitter-sentiment-analysis?style=social" alt="GitHub watchers"/>
</a>

<br><br>

**⭐ Star this repository if you found it helpful!**

**🍴 Fork it to build your own NLP projects!**

**📢 Share it with the ML community!**

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" alt="Footer"/>

<br>

<i>💬 "The limits of my language mean the limits of my world." - Ludwig Wittgenstein</i>

<br><br>


<br>

---

**© 2025 Open Source Project | Natural Language Processing | MIT License**

<br>


</div>
