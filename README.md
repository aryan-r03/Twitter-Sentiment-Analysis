🐦 Twitter Sentiment Analysis Web App
This is a complete end-to-end Machine Learning project for analyzing the sentiment (Positive or Negative) of Twitter data. It uses a Logistic Regression model trained on text features extracted via TF-IDF and is deployed as a live web application using Flask.

The project features a full ML pipeline, from data cleaning and training to model persistence and a simple, interactive frontend.

✨ Features
Machine Learning Pipeline: Trains a Logistic Regression classifier using TF-IDF vectorization.

Robust Preprocessing: Includes custom text cleaning (removing URLs, mentions, and punctuation), stop-word removal, and Lemmatization via NLTK.

Model Persistence: Saves the trained model and vectorizer using pickle for fast deployment.

Flask Web Application: A simple, single-page web interface to analyze tweets in real-time.

API Endpoints: Includes REST endpoints for single-tweet analysis (/api/analyze), batch analysis (/api/batch_analyze), and dynamic retraining (/api/retrain).

Flexible Data Loading: Can load custom CSV data or fall back to a built-in sample dataset.

🛠️ Installation and Setup
Prerequisites

Python 3.8+

The Twitter_data.csv file (or your own dataset) in the project directory.
