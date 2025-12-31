from flask import Flask, render_template, request, jsonify
import os
from sentiment_model import SentimentModel

app = Flask(__name__)

CSV_FILE = 'data.csv'      #csv path
MODEL_FILE = 'sentiment_model.pkl'

sentiment_model = SentimentModel()

print("\n" + "=" * 60)
print("INITIALIZING SENTIMENT ANALYSIS MODEL")
print("=" * 60)

if os.path.exists(MODEL_FILE):
    print(f"\n✓ Found existing model: {MODEL_FILE}")
    if sentiment_model.load_model(MODEL_FILE):
        print("✓ Model loaded successfully!")
    else:
        print("⚠ Failed to load model. Will train a new one...")
        df = sentiment_model.create_sample_dataset()
        if df is not None and len(df) > 0:
            sentiment_model.train(df)
            sentiment_model.save_model(MODEL_FILE)
else:
    print(f"\n⚠ No saved model found at {MODEL_FILE}")
    print("Training new model...")

    df = sentiment_model.create_sample_dataset()

    if df is not None and len(df) > 0:
        sentiment_model.train(df)
        sentiment_model.save_model(MODEL_FILE)
    else:
        print("\n❌ ERROR: Could not load dataset!")
        print("Please ensure Twitter_data.csv is in the directory")

print("\n" + "=" * 60)
print("MODEL READY")
print("=" * 60 + "\n")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        if not data or 'tweet' not in data:
            return jsonify({
                'success': False,
                'error': 'No tweet provided'
            }), 400

        tweet = data['tweet']

        if not tweet.strip():
            return jsonify({
                'success': False,
                'error': 'Tweet is empty'
            }), 400

        result = sentiment_model.predict(tweet)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
