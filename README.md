# 🐦 Twitter Sentiment Analysis Web App

This is a complete end-to-end Machine Learning project for analyzing the sentiment (Positive or Negative) of Twitter data. It uses a **Logistic Regression** model trained on text features extracted via **TF-IDF** and is deployed as a live web application using **Flask**.

The project features a full ML pipeline, from data cleaning and training to model persistence and a simple, interactive frontend.

## ✨ Features

* **Machine Learning Pipeline:** Trains a Logistic Regression classifier using TF-IDF vectorization.
* **Robust Preprocessing:** Includes custom text cleaning (removing URLs, mentions, and punctuation), stop-word removal, and **Lemmatization** via NLTK.
* **Model Persistence:** Saves the trained model and vectorizer using `pickle` for fast deployment.
* **Flask Web Application:** A simple, single-page web interface to analyze tweets in real-time.
* **API Endpoints:** Includes REST endpoints for single-tweet analysis (`/api/analyze`), batch analysis (`/api/batch_analyze`), and dynamic retraining (`/api/retrain`).
* **Flexible Data Loading:** Can load custom CSV data or fall back to a built-in sample dataset (`Twitter_data.csv`).

## 🛠️ Installation and Setup

### Prerequisites

* Python 3.8+
* The `Twitter_data.csv` file (or your own dataset) in the project directory.

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install Dependencies:**
    The project uses `pandas`, `numpy`, `scikit-learn`, `nltk`, and `Flask`.

    ```bash
    pip install pandas numpy scikit-learn nltk flask flask-cors
    ```
    *(Note: You might need to run `pip install scikit-learn` instead of `sklearn`.)*

4.  **Download NLTK Data:**
    The `App.py` script attempts to download necessary NLTK packages (`stopwords`, `wordnet`) automatically, but you can run this manually if needed:
    ```bash
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
    ```

## ▶️ How to Run the App

1.  **Start the Flask Server:**
    ```bash
    python App.py
    ```
    The console will print status messages about model loading or training.

2.  **Access the Web Interface:**
    Open your web browser and navigate to:
    **`http://127.0.0.1:5000`**

3.  **Analyze Tweets:**
    Enter a tweet into the text area and click "Analyze Sentiment" to see the prediction, confidence, and positive/negative scores.

## 📊 Model Details

The model training process is defined in the `train_and_save_model` function and leverages the `SentimentModel` class:

* **Model:** `LogisticRegression` with `class_weight='balanced'` to handle potential data imbalance.
* **Vectorization:** `TfidfVectorizer` with `max_features=5000` and `ngram_range=(1, 2)` (Unigrams and Bigrams).
* **Evaluation Metrics:** Reports Accuracy, Precision, Recall, and F1-Score on the test set.

## ⚙️ API Endpoints

The Flask application exposes several API endpoints for integration:

| Endpoint | Method | Description | Request Body Example (JSON) |
| :--- | :--- | :--- | :--- |
| `/api/analyze` | `POST` | Analyze sentiment of a single tweet. | `{"tweet": "This is amazing!"}` |
| `/api/batch_analyze` | `POST` | Analyze sentiment for a list of tweets. | `{"tweets": ["Good day", "Bad day"]}` |
| `/api/retrain` | `POST` | Retrain the model using a new CSV file. | `{"csv_path": "new_data.csv"}` |
