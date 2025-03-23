import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📊 Load the dataset
file_path = r"C:/Users/nhymavathi40gmail.c/OneDrive/Desktop/OneDrive/Documents/spam message detection/spam/spam_sms.csv"
data = pd.read_csv(file_path)

# 🌟 Preprocessing
data = data.iloc[:, :2]  # Keep only the first two columns
data.columns = ['label', 'message']

# 🔥 Inspect unique labels to avoid mapping issues
print("\n✅ Unique labels in the dataset:")
print(data['label'].unique())

# 🔥 Fix label mapping
data['label'] = data['label'].str.strip().str.lower()
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# ✅ Remove NaN labels
data = data.dropna(subset=['label', 'message'])

# ✅ Check class distribution
print("\n✅ Label distribution after cleaning:")
print(data['label'].value_counts())

# 🔥 TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['message'])
y = data['label'].values

# ✅ Stratified splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🚀 Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# 🛠️ Model evaluation
y_pred = model.predict(X_test)

# ✅ Display results
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save the model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\n🔥 Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
