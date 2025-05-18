import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load Dataset
df = pd.read_csv('emotions.csv')  # Make sure emotions.csv is in the same folder

# Step 2: Clean Text
df['clean_text'] = df['text'].apply(nfx.remove_stopwords)
df['clean_text'] = df['clean_text'].apply(nfx.remove_punctuations)

# Step 3: Train-Test Split
X = df['clean_text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save Model
joblib.dump(model, 'emotion_model.pkl')
print("Model saved as emotion_model.pkl")

# Load the model again for prediction (to simulate runtime usage)
model = joblib.load('emotion_model.pkl')

# Step 8: Runtime input loop for emotion detection
def predict_emotion(text):
    return model.predict([text])[0]

if __name__ == "__main__":
    print("\nEmotion Detection (type 'exit' to quit)")
    while True:
        user_input = input("Enter text: ")
        if user_input.strip().lower() == 'exit':
            print("Goodbye!")
            break
        prediction = predict_emotion(user_input)
        print(f"Detected Emotion: {prediction}\n")
