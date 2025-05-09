import os
import pickle
import pandas as pd
from flask import Flask, render_template, request
from models.personalityanalyzer import analyze_personality
from models.FacialRecognition import capture_and_analyze_emotion, load_recommendations, get_recommendations


app = Flask(__name__)


MODEL_DIR = r'C:\Users\Hp\PycharmProjects\emotionanalyzer\models'
TFIDF_PICKLE = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
MODEL_PICKLE = os.path.join(MODEL_DIR, 'logistic_model.pkl')
RECOMMENDATION_FILE = os.path.join(MODEL_DIR, 'recommendations.csv')


with open(TFIDF_PICKLE, 'rb') as file:
   tfidf = pickle.load(file)


with open(MODEL_PICKLE, 'rb') as file:
   model = pickle.load(file)


recommendations_df = pd.read_csv(RECOMMENDATION_FILE)


def get_recommendations(emotion):
   emotion = emotion.lower()
   emotion_map = {
       'happy': 'happiness',
       'sad': 'sadness',
       'angry': 'anger',
       'fear': 'worry',
       'disgust': 'hate',
       'surprise': 'surprise',
       'neutral': 'neutral'
   }
   normalized = emotion_map.get(emotion, emotion)
   recs = recommendations_df[recommendations_df['emotion'] == normalized]['recommendation'].tolist()
   return recs if recs else ['neutral']


@app.route('/', methods=['GET', 'POST'])
def index():
   text_result = {}
   face_result = {}
   personality_result = {}


   if request.method == 'POST':
       if 'text_input' in request.form:
           text = request.form['text_input']
           if text.strip():
               vectorized = tfidf.transform([text])
               emotion = model.predict(vectorized)[0]
               recs = get_recommendations(emotion)
               text_result = {


                   "recommendations": recs
               }


       elif 'personality_input' in request.form:
           text = request.form['personality_input']
           if text.strip():
               personality_result = analyze_personality(text)


       elif 'face_analysis' in request.form:
           emotion = capture_and_analyze_emotion()
           if emotion:


               face_result["recommendations"] = get_recommendations(emotion)
           else:


               face_result["recommendations"] = get_recommendations("neutral")


   return render_template('index.html',
                          text_result=text_result,
                          face_result=face_result,
                          personality_result=personality_result)


if __name__ == '__main__':
   app.run(debug=True)
