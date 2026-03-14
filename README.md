# Megathon-24 - Sentiment Analysis Web Application

A Flask-based web application for real-time sentiment analysis using BERT models and pre-trained transformers.

## 🚀 Technologies Used

### Backend
- **Flask** - Python web framework for API development
- **PyTorch** - Deep learning framework for model inference
- **Transformers (Hugging Face)** - Pre-trained BERT models and tokenizers
- **scikit-learn** - Machine learning utilities and label encoding
- **joblib** - Model serialization and loading

### Frontend
- **HTML/CSS/JavaScript** - Web interface
- **Matplotlib** - Data visualization and plotting
- **Base64 encoding** - Image rendering in web browser

### Machine Learning
- **BERT (Bidirectional Encoder Representations from Transformers)** - For text classification
- **RoBERTa** - Fallback pre-trained sentiment model
- **Label Encoding** - Emotion classification mapping

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for downloading pre-trained models)

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Harshita-K/megathon-24.git
cd megathon-24-main
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies included:**
- Flask==2.3.3
- torch==2.1.0
- joblib==1.3.2
- matplotlib==3.7.2
- transformers==4.34.0
- scikit-learn==1.3.0
- numpy==1.24.3
- Werkzeug==2.3.7
- Jinja2==3.1.2

### 3. Optional: Create Basic Model (if you want custom model)
```bash
python create_model.py
```

## 🎯 How to Run

### Method 1: Run with Auto-Fallback (Recommended)
```bash
python app.py
```

### Method 2: Run with Development Server
```bash
flask --app app.py run --debug
```

### Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
megathon-24-main/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── intro.html        # Introduction page
│   ├── index.html        # Main application page
│   └── analysis.html     # Analysis page
├── static/              # CSS, JS, images
└── trained_model/       # Custom model directory (optional)
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── label_encoder.joblib
```

## 🎮 Features

### Core Functionality
- **Real-time Sentiment Analysis** - Analyze text input for emotions
- **Emotion Classification** - Detects: anger, sadness, fear, happy, love, surprise
- **Intensity Scoring** - Provides confidence levels (1-10)
- **Visual Analytics** - Sentiment tracking over time
- **Average Feedback** - Overall mood analysis

### API Endpoints
- `GET /` - Introduction page
- `GET /home` - Main application interface
- `GET /in-depth-analysis` - Analysis dashboard
- `POST /predict` - Sentiment prediction API
- `GET /plot` - Generate sentiment visualization
- `GET /average_feedback` - Get overall mood statistics

## 🔄 Model Behavior

The application uses a **smart fallback system**:

1. **Primary**: Attempts to load custom trained BERT model from `./trained_model/`
2. **Fallback**: Uses pre-trained RoBERTa sentiment model from Hugging Face
3. **Error Handling**: Provides informative error messages if models fail

### Supported Emotions
- **Positive**: happy, love, surprise
- **Negative**: anger, sadness, fear

## 🐛 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
pip install -r requirements.txt
```

**2. Model Download Issues**
- Ensure stable internet connection
- The app will automatically download pre-trained models on first run

**3. Port Already in Use**
```bash
flask --app app.py run --port 5001
```

**4. Memory Issues**
- Close other applications
- The fallback model uses less memory than custom BERT

### Error Messages
- `"Model not loaded"` - Custom model missing, using fallback
- `"No model available"` - Both models failed, check internet connection
- `"Prediction failed"` - Input processing error, try different text

## 📊 Usage Examples

### Text Input Examples
- "I am feeling great today!" → Positive sentiment
- "This is disappointing" → Negative sentiment
- "I love this new feature" → Positive sentiment

### API Usage
```javascript
// POST to /predict
{
  "text": "Your text here"
}

// Response
{
  "Overall Sentiment": "positive",
  "Intensity": 8,
  "Predicted_label": "happy"
}
```
