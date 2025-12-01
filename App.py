import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass


class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def clean_text(self, text):
        text = str(text)
        text = text.lower()

        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        text = re.sub(r'@\w+', '', text)

        text = re.sub(r'[^a-zA-z\s!?]', '', text)

        words = text.split()

        if self.stop_words:
            keep_words = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', "n't"}
            words = [self.lemmatizer.lemmatize(word) for word in words
                     if (word not in self.stop_words or word in keep_words) and len(word) > 2]

        return ' '.join(words)


    def load_dataset_from_csv(self, csv_path, text_column='text', label_column='sentiment'):
        print(f"\nLoading dataset from: {csv_path}")

        try:

            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except:
                df = pd.read_csv(csv_path, encoding='latin-1')

            print(f"Dataset loaded successfully! Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            # Check if required columns exist
            if text_column not in df.columns:
                print(f"\nWarning: Column '{text_column}' not found!")
                print("Available columns:", df.columns.tolist())
                possible_text_cols = ['text', 'tweet', 'message', 'content', 'SentimentText']
                for col in possible_text_cols:
                    if col in df.columns:
                        text_column = col
                        print(f"Using '{text_column}' as text column")
                        break

            if label_column not in df.columns:
                print(f"\nWarning: Column '{label_column}' not found!")
                print("Available columns:", df.columns.tolist())

                possible_label_cols = ['sentiment', 'label', 'target', 'polarity', 'Sentiment']
                for col in possible_label_cols:
                    if col in df.columns:
                        label_column = col
                        print(f"Using '{label_column}' as label column")
                        break

            df = df.rename(columns={text_column: 'text', label_column: 'sentiment'})
            df = df.dropna(subset=['text', 'sentiment'])

            unique_labels = df['sentiment'].unique()
            print(f"\nUnique sentiment labels: {unique_labels}")

            if set(unique_labels).issubset({0, 1}):
                print("Labels are already binary (0, 1)")
            elif set(unique_labels).issubset({0, 4}):
                df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
                print("Converted labels from (0, 4) to (0, 1)")
            elif set(unique_labels).issubset({'negative', 'positive'}):
                df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
                print("Converted labels from (negative, positive) to (0, 1)")
            elif set(unique_labels).issubset({'neg', 'pos'}):
                df['sentiment'] = df['sentiment'].map({'neg': 0, 'pos': 1})
                print("Converted labels from (neg, pos) to (0, 1)")
            else:
                print(f"Warning: Unexpected label format: {unique_labels}")
                print("Attempting automatic conversion...")
                df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
                df = df.dropna(subset=['sentiment'])
                if df['sentiment'].min() != 0 or df['sentiment'].max() != 1:
                    df['sentiment'] = (df['sentiment'] > df['sentiment'].median()).astype(int)

            print(f"\nDataset distribution:")
            print(f"Negative tweets: {(df['sentiment'] == 0).sum()}")
            print(f"Positive tweets: {(df['sentiment'] == 1).sum()}")
            print(f"Total tweets: {len(df)}")

            print("\n--- Sample Tweets ---")
            print("\nPositive examples:")
            print(df[df['sentiment'] == 1]['text'].head(2).values)
            print("\nNegative examples:")
            print(df[df['sentiment'] == 0]['text'].head(2).values)

            return df

        except FileNotFoundError:
            print(f"Error: File '{csv_path}' not found!")
            print("Please make sure the CSV file is in the same directory as this script.")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None

    def create_sample_dataset(self):
        print("\nLoading Twitter_data.csv...")

        try:
            df = pd.read_csv(
                'Twitter_data.csv',
                encoding='latin-1',
                header=None,  # ← Tell pandas there are NO headers
                names=['ids', 'game', 'sentiment_text', 'text']  # ← Name the columns
            )

            print(f"✓ Dataset loaded: {len(df)} tweets")
            print(f"Columns identified: {df.columns.tolist()}")

            df['sentiment'] = df['sentiment_text'].map({
                'Positive': 1,
                'Negative': 0,
                'Neutral': 0
            })

            df = df.dropna(subset=['text', 'sentiment'])

            neg_count = (df['sentiment'] == 0).sum()
            pos_count = (df['sentiment'] == 1).sum()

            print(f"\nDataset distribution:")
            print(f"  Negative: {neg_count}")
            print(f"  Positive: {pos_count}")

            if neg_count > 0 and pos_count > 0:
                min_count = min(neg_count, pos_count)
                df_neg = df[df['sentiment'] == 0].sample(n=min_count, random_state=42)
                df_pos = df[df['sentiment'] == 1].sample(n=min_count, random_state=42)
                df = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=42)

            print(f"✓ Final dataset: {len(df)} tweets")
            print(f"  Negative: {(df['sentiment'] == 0).sum()}")
            print(f"  Positive: {(df['sentiment'] == 1).sum()}")

            if len(df) > 0:
                print(f"\n📊 Sample tweets:")
                if (df['sentiment'] == 1).any():
                    print(f"  Positive: {df[df['sentiment'] == 1]['text'].iloc[0][:80]}...")
                if (df['sentiment'] == 0).any():
                    print(f"  Negative: {df[df['sentiment'] == 0]['text'].iloc[0][:80]}...")

            return df

        except FileNotFoundError:
            print("❌ Error: Twitter_data.csv not found!")
            return None
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            import traceback
            traceback.print_exc()
            return None




    def train(self, df, test_size=0.2):
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)

        print("\nStep 1: Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)

        df = df[df['cleaned_text'].str.len() > 0]
        print(f"Valid samples after cleaning: {len(df)}")

        print("\nStep 2: Vectorizing text using TF-IDF...")
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['sentiment']
        print(f"Feature matrix shape: {X.shape}")

        print("\nStep 3: Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")

        print("\nStep 4: Training Logistic Regression model...")
        self.model.fit(X_train, y_train)

        print("\nStep 5: Evaluating model performance...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1 Score:  {f1 * 100:.2f}%")

        print("\n--- Detailed Classification Report ---")
        print(classification_report(y_test, y_pred,
                                    target_names=['Negative', 'Positive']))

        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives:  {cm[0][0]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}")
        print(f"True Positives:  {cm[1][1]}")

        return accuracy

    def predict(self, text):
        cleaned = self.clean_text(text)

        if not cleaned:
            return {
                'sentiment': 'Neutral',
                'confidence': 50.0,
                'positive_score': 50.0,
                'negative_score': 50.0
            }

        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        probability = self.model.predict_proba(vectorized)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability) * 100

        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'positive_score': round(probability[1] * 100, 2),
            'negative_score': round(probability[0] * 100, 2)
        }

    def save_model(self, filepath='sentiment_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
        print(f"\n✓ Model saved to {filepath}")

    def load_model(self, filepath='sentiment_model.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
        print(f"✓ Model loaded from {filepath}")

def train_and_save_model(csv_path=None):
    print("\n" + "=" * 60)
    print("TWITTER SENTIMENT ANALYSIS - MODEL TRAINING")
    print("=" * 60)

    sentiment_model = SentimentModel()

    if csv_path:
        df = sentiment_model.load_dataset_from_csv(csv_path)
        if df is None:
            print("\nFalling back to sample dataset...")
            df = sentiment_model.create_sample_dataset()
    else:
        df = sentiment_model.create_sample_dataset()
        print(f"Sample dataset size: {len(df)} tweets")

    accuracy = sentiment_model.train(df)

    sentiment_model.save_model('sentiment_model.pkl')

    print("\n" + "=" * 60)
    print("✓ MODEL TRAINING COMPLETE!")
    print("=" * 60)

    return sentiment_model



from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

CSV_FILE = 'Twitter_data.csv'

if os.path.exists('sentiment_model.pkl'):
    print("\n✓ Loading existing model...")
    sentiment_analyzer = SentimentModel()
    sentiment_analyzer.load_model('sentiment_model.pkl')
else:
    print("\n⚠ No existing model found. Training new model...")
    if os.path.exists(CSV_FILE):
        print(f"Found CSV file: {CSV_FILE}")
        sentiment_analyzer = train_and_save_model(csv_path=CSV_FILE)
    else:
        print(f"CSV file '{CSV_FILE}' not found. Using sample dataset.")
        sentiment_analyzer = train_and_save_model()


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        tweet = data.get('tweet', '')

        if not tweet:
            return jsonify({'error': 'No tweet provided'}), 400

        result = sentiment_analyzer.predict(tweet)

        return jsonify({
            'success': True,
            'tweet': tweet,
            'result': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        tweets = data.get('tweets', [])

        if not tweets:
            return jsonify({'error': 'No tweets provided'}), 400

        results = []
        for tweet in tweets:
            result = sentiment_analyzer.predict(tweet)
            results.append({
                'tweet': tweet,
                'result': result
            })

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    try:
        data = request.get_json()
        csv_path = data.get('csv_path', CSV_FILE)

        if not os.path.exists(csv_path):
            return jsonify({'error': f'CSV file not found: {csv_path}'}), 404

        global sentiment_analyzer
        sentiment_analyzer = train_and_save_model(csv_path=csv_path)

        return jsonify({
            'success': True,
            'message': 'Model retrained successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Twitter Sentiment Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            margin-top: 10px;
            font-size: 14px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }

        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .char-count {
            text-align: right;
            color: #999;
            margin-top: 5px;
            font-size: 14px;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin-top: 15px;
            transition: transform 0.2s;
        }

        .btn:hover {
            transform: translateY(-2px);
        }

        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            animation: fadeIn 0.5s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result.positive {
            background: #d4edda;
            border: 2px solid #28a745;
        }

        .result.negative {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }

        .result h3 {
            font-size: 24px;
            margin-bottom: 15px;
        }

        .result.positive h3 {
            color: #28a745;
        }

        .result.negative h3 {
            color: #dc3545;
        }

        .scores {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }

        .score-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .score-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }

        .score-value {
            font-size: 28px;
            font-weight: bold;
        }

        .score-value.positive {
            color: #28a745;
        }

        .score-value.negative {
            color: #dc3545;
        }

        .loading {
            text-align: center;
            color: #667eea;
            margin-top: 20px;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }

        .info-box h4 {
            color: #1976d2;
            margin-bottom: 5px;
        }

        .info-box p {
            color: #555;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐦 Twitter Sentiment Analysis</h1>
            <p>AI-Powered Emotion Detection using Machine Learning</p>
            <div class="badge">Trained with Logistic Regression</div>
        </div>

        
</div>

        <div class="card">
            <h2>Enter Tweet</h2>
            <textarea 
                id="tweetInput" 
                placeholder="Type or paste a tweet here... (e.g., 'I love this amazing product!')"
                maxlength="280"
            ></textarea>
            <div class="char-count">
                <span id="charCount">0</span> / 280 characters
            </div>
            <button class="btn" id="analyzeBtn" onclick="analyzeSentiment()">
                Analyze Sentiment
            </button>

            <div id="loading" class="loading" style="display: none;">
                <div class="spinner"></div>
                <p>Analyzing sentiment...</p>
            </div>

            <div id="result" style="display: none;"></div>
        </div>
    </div>

    <script>
        const tweetInput = document.getElementById('tweetInput');
        const charCount = document.getElementById('charCount');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const loading = document.getElementById('loading');
        const resultDiv = document.getElementById('result');

        tweetInput.addEventListener('input', function() {
            charCount.textContent = this.value.length;
        });

        async function analyzeSentiment() {
            const tweet = tweetInput.value.trim();

            if (!tweet) {
                alert('Please enter a tweet to analyze!');
                return;
            }

            analyzeBtn.disabled = true;
            loading.style.display = 'block';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ tweet: tweet })
                });

                const data = await response.json();

                if (data.success) {
                    displayResult(data.result);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error analyzing sentiment: ' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                loading.style.display = 'none';
            }
        }

        function displayResult(result) {
            const sentimentClass = result.sentiment.toLowerCase();

            resultDiv.innerHTML = `
                <h3>Sentiment: ${result.sentiment}</h3>
                <div class="scores">
                    <div class="score-item">
                        <div class="score-label">Confidence</div>
                        <div class="score-value">${result.confidence}%</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Positive Score</div>
                        <div class="score-value positive">${result.positive_score}%</div>
                    </div>
                    <div class="score-item">
                        <div class="score-label">Negative Score</div>
                        <div class="score-value negative">${result.negative_score}%</div>
                    </div>
                </div>
            `;

            resultDiv.className = 'result ' + sentimentClass;
            resultDiv.style.display = 'block';
        }

        tweetInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.ctrlKey) {
                e.preventDefault();
                analyzeSentiment();
            }
        });
    </script>
</body>
</html>
"""


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TWITTER SENTIMENT ANALYSIS - MACHINE LEARNING PROJECT")
    print("=" * 60)
    print("\n📌 To use your own dataset:")
    print(f"   Place CSV file named '{CSV_FILE}' in this directory")
    print("   CSV should have columns: 'text' and 'sentiment'")
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000)