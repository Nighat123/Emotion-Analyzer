# Emotion & Personality Analyzer

An AI-powered web application that detects human emotions from both **text** and **facial expressions**, performs **personality analysis**, and offers personalized **recommendations**. Built with Flask, OpenCV, and pre-trained ML models.

## Features

- **Text Emotion Detection**  
  Uses TF-IDF and Logistic Regression to classify user-submitted text into emotional categories.

- **Facial Emotion Recognition**  
  Captures a webcam image and analyzes emotion using DeepFace and OpenCV.

- **Personality Analysis**  
  Applies sentiment metrics (via TextBlob) to infer personality traits from user text.

- **Recommendations System**  
  Suggests activities or coping tips based on the detected emotion.

- **Web Interface**  
  Simple and responsive Flask-powered HTML form interface—no frontend frameworks.

## Folder Structure

```

project-root/
├── app.py
├── models/
│   ├── logistic\_model.pkl
│   ├── tfidf\_vectorizer.pkl
│   ├── recommendations.csv
│   ├── personalityanalyzer.py
│   └── FacialRecognition.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md

````

## How to Run

1. **Install dependencies**  
  ```bash:
   pip install -r requirements.txt
  ```

2. **Start the app**

   ```bash
   python app.py
   ```

3. **Open in browser**
   Navigate to `http://127.0.0.1:5000/`

## Webcam Controls (for Facial Emotion)

* `Q` – Take a picture
* `S` – Exit webcam

## Data Format

Your `recommendations.csv` should follow this structure:


emotion,recommendation
happiness,"Take a walk and enjoy the sunshine."
sadness,"Talk to a friend or write down your thoughts."





