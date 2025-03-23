from flask import Flask, request, render_template, redirect, url_for
import pickle
import os

# ✅ Initialize the Flask app
app = Flask(__name__)

# Load the model and vectorizer
model_path = os.path.join(os.getcwd(), "model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# 🌐 Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# 🚀 Route to handle spam detection
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']  # Get message from the form
    actual_label = request.form.get('actual_label')  # Get the actual label (optional)

    # Transform the message using the same vectorizer
    input_features = vectorizer.transform([message])

    # Predict spam or not spam
    prediction = model.predict(input_features)[0]

    # Prepare the result message
    prediction_label = "Spam" if prediction == 1 else "Not Spam"

    # Check correctness if the actual label is provided
    correctness = ""
    if actual_label:
        correctness = "Correct" if prediction_label.lower() == actual_label.lower() else "Wrong"

    # Redirect to the result page
    return redirect(url_for('result', message=message, prediction=prediction_label, correctness=correctness))

# 🛠️ Route for the result page
@app.route('/result')
def result():
    message = request.args.get('message')
    prediction = request.args.get('prediction')
    correctness = request.args.get('correctness')

    return render_template('result.html', message=message, prediction=prediction, correctness=correctness)

# ✅ Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
