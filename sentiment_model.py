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
                'data.csv', # ← Here place the csv path
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
        print(f"F1-Score:  {f1 * 100:.2f}%")

        print("\n" + "-" * 60)
        print("CONFUSION MATRIX")
        print("-" * 60)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n                Predicted")
        print(f"              Neg    Pos")
        print(f"Actual Neg   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Pos   {cm[1][0]:4d}  {cm[1][1]:4d}")

        print("\n" + "-" * 60)
        print("CLASSIFICATION REPORT")
        print("-" * 60)
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

        return accuracy

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'Unknown',
                'confidence': 0,
                'positive_score': 0,
                'negative_score': 0
            }

        cleaned = self.clean_text(text)

        if not cleaned:
            return {
                'sentiment': 'Unknown',
                'confidence': 0,
                'positive_score': 0,
                'negative_score': 0
            }

        X = self.vectorizer.transform([cleaned])

        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        sentiment = 'Positive' if prediction == 1 else 'Negative'
        confidence = int(max(probabilities) * 100)

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': int(probabilities[1] * 100),
            'negative_score': int(probabilities[0] * 100)
        }

    def save_model(self, model_path='sentiment_model.pkl'):
        print(f"\nSaving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'lemmatizer': self.lemmatizer,
                'stop_words': self.stop_words
            }, f)
        print(f"✓ Model saved successfully!")

    def load_model(self, model_path='sentiment_model.pkl'):
        print(f"\nLoading model from {model_path}...")
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.lemmatizer = data['lemmatizer']
                self.stop_words = data['stop_words']
            print(f"✓ Model loaded successfully!")
            return True
        except FileNotFoundError:
            print(f"Model file not found at {model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
