# Twitter Sentiment Analysis

A Flask web application that performs sentiment analysis on tweets using machine learning (Logistic Regression).

## Project Structure

```
├── app.py                    # Main Flask application with routes
├── sentiment_model.py        # SentimentModel class with ML logic
├── templates/
│   └── index.html           # Frontend HTML template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Features

- **Text Preprocessing**: Cleans and lemmatizes text, removes URLs, mentions, and stopwords
- **TF-IDF Vectorization**: Converts text to numerical features
- **Machine Learning**: Uses Logistic Regression for binary sentiment classification
- **Web Interface**: Beautiful, responsive UI for real-time sentiment analysis
- **Model Persistence**: Saves and loads trained models

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (automatically done on first run):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Usage

1. Place your training data CSV file named `Twitter_data.csv` in the project directory
   - CSV should have columns for text and sentiment
   - Supported formats: text/sentiment, text/label, etc.

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

4. Enter a tweet and click "Analyze Sentiment" to see the results!

## Dataset Format

The CSV file should contain:
- A text column (tweet, text, message, etc.)
- A sentiment/label column with values:
  - Binary: 0/1, negative/positive, neg/pos
  - Or: 0/4 (will be converted to 0/1)

## Model Details

- **Algorithm**: Logistic Regression with balanced class weights
- **Features**: TF-IDF with bigrams (5000 max features)
- **Preprocessing**: 
  - Lowercasing
  - URL and mention removal
  - Lemmatization
  - Stopword removal (keeping negation words)

## API Endpoint

### POST /api/analyze
Analyzes sentiment of provided text.

**Request:**
```json
{
  "tweet": "I love this amazing product!"
}
```

**Response:**
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

## Files Description

### sentiment_model.py
Contains the `SentimentModel` class with methods:
- `clean_text()`: Text preprocessing
- `load_dataset_from_csv()`: Load and validate dataset
- `create_sample_dataset()`: Load Twitter_data.csv specifically
- `train()`: Train the model and evaluate performance
- `predict()`: Predict sentiment for new text
- `save_model()` / `load_model()`: Model persistence

### app.py
Flask application with:
- Model initialization and training
- Route handlers for web interface and API
- Request/response handling

### templates/index.html
Frontend with:
- Beautiful gradient UI
- Real-time character counter
- Animated results display
- Responsive design

## License

MIT License
