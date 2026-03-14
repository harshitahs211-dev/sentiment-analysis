from flask import Flask, render_template, request, jsonify
import torch
import joblib
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import io
import base64
from matplotlib.figure import Figure
import os

app = Flask(__name__)

# Try to load custom model first, fallback to pre-trained model
try:
    if os.path.exists('./trained_model'):
        print("Loading custom trained model...")
        model = BertForSequenceClassification.from_pretrained('./trained_model')
        tokenizer = BertTokenizer.from_pretrained('./trained_model')
        label_encoder = joblib.load('./trained_model/label_encoder.joblib')
        model.eval()
        use_custom_model = True
        print("Custom model loaded successfully!")
    else:
        raise FileNotFoundError("Custom model not found")
except Exception as e:
    print(f"Custom model loading failed: {e}")
    print("Falling back to pre-trained sentiment model...")
    try:
        # Use a pre-trained sentiment analysis model
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        use_custom_model = False
        print("Pre-trained model loaded successfully!")
    except Exception as e2:
        print(f"Failed to load pre-trained model: {e2}")
        sentiment_pipeline = None
        use_custom_model = False

sentiment_data = []

def predict(text):
    if use_custom_model:
        # Use custom trained model
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
            predicted_class_id = logits.argmax().item()
            predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
            probability_percentage = probabilities[0][predicted_class_id].item() * 100
            negative_labels = {"anger", "sadness", "fear"}
            positive_labels = {"happy", "love", "surprise"}
            overall_sentiment = "negative" if predicted_label in negative_labels else "positive"
            intensity = (probability_percentage // 10) + 1
            return {"Overall Sentiment": overall_sentiment, "Intensity": intensity, "Predicted_label": predicted_label}
        except Exception as e:
            print(f"Custom model prediction failed: {e}")
            return {"error": "Prediction failed", "message": str(e)}
    else:
        # Use pre-trained sentiment pipeline
        try:
            if sentiment_pipeline is None:
                return {"error": "No model available", "message": "Neither custom nor pre-trained model is loaded"}
            
            result = sentiment_pipeline(text)[0]
            
            # Map the pre-trained model labels to our format
            label_mapping = {
                'LABEL_0': 'negative',  # or could be 'sadness'
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive',  # or could be 'happy'
                'NEGATIVE': 'negative',
                'POSITIVE': 'positive'
            }
            
            predicted_label = result['label']
            confidence = result['score']
            
            # Convert to our format
            overall_sentiment = label_mapping.get(predicted_label, predicted_label.lower())
            if 'positive' in overall_sentiment.lower() or 'joy' in overall_sentiment.lower():
                overall_sentiment = 'positive'
                predicted_label = 'happy'
            else:
                overall_sentiment = 'negative'
                predicted_label = 'sadness'
            
            intensity = min(int(confidence * 10) + 1, 10)
            
            return {
                "Overall Sentiment": overall_sentiment, 
                "Intensity": intensity, 
                "Predicted_label": predicted_label
            }
        except Exception as e:
            print(f"Pre-trained model prediction failed: {e}")
            return {"error": "Prediction failed", "message": str(e)}
@app.route('/average_feedback', methods=['GET'])
def average_feedback():
    if not sentiment_data:
        return jsonify({"feedback": "No mood data available yet.", "average_intensity": 0})

    # Calculate average intensity
    total_intensity = sum(data['Intensity'] for data in sentiment_data)
    average_intensity = total_intensity / len(sentiment_data)

    # Determine overall feedback based on average sentiment
    positive_count = sum(1 for data in sentiment_data if data['Overall Sentiment'] == "positive")
    negative_count = len(sentiment_data) - positive_count
    overall_sentiment = "positive" if positive_count >= negative_count else "negative"

    # Generate feedback based on overall mood
    if overall_sentiment == "positive":
        feedback_message = "On average, your moods are positive! Keep up the positivity!"
    else:
        feedback_message = "It seems there have been more negative moods lately. Remember to take time for self-care."

    return jsonify({"feedback": feedback_message, "average_intensity": average_intensity, "overall_sentiment": overall_sentiment})

def plot_sentiment_over_time(sentiment_data):
    time_steps = list(range(1, len(sentiment_data) + 1))
    intensities = [
        data['Intensity'] if data['Overall Sentiment'] == 'positive' else -data['Intensity']
        for data in sentiment_data
    ]
    fig = Figure()
    ax = fig.subplots()
    ax.plot(time_steps, intensities, marker='o', linestyle='-', color='blue')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("User Sentiment Over Time")
    ax.set_xlabel("Input Number")
    ax.set_ylabel("Sentiment Intensity")
    ax.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url

@app.route('/')
def introduction():
    return render_template('intro.html')  # Replace with the name of your introduction page template

# Route for the index page (where you'll go after clicking the "Start" button)
@app.route('/home')
def home():
    return render_template('index.html')  # This serves your index.html page

@app.route('/in-depth-analysis')
def in_depth_analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def analyze():
    try:
        text_input = request.form['text']
        result = predict(text_input)
        if 'error' not in result:
            sentiment_data.append(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Request failed", "message": f"An error occurred: {str(e)}"})

@app.route('/plot')
def plot():
    plot_url = plot_sentiment_over_time(sentiment_data)
    return jsonify({'plot_url': plot_url})

if __name__ == "__main__":
    app.run(debug=True)
