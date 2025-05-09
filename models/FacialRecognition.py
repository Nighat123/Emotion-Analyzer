from deepface import DeepFace
import pandas as pd
import logging
import cv2
logging.basicConfig(level=logging.DEBUG)


def load_recommendations(csv_path="recommendations.csv"):

    try:
        df = pd.read_csv(csv_path)
        # Debugging line
        print("Columns in CSV:", df.columns)


        recommendations_dict = df.groupby('emotion')['recommendation'].apply(list).to_dict()
        return recommendations_dict
    except Exception as e:
        print(f"Error loading recommendations from CSV: {e}")
        return {}

recommendations_dict = load_recommendations()
def normalize_emotion(emotion):
    emotion_map = {
        'happy': 'happiness',
        'sad': 'sadness',
        'angry': 'anger',
        'fear': 'worry',
        'disgust': 'hate',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }

    return emotion_map.get(emotion.lower(), emotion.lower())
    print(f"Normalizing '{emotion}' to '{emotion_map.get(emotion.lower(), emotion.lower())}'")

def get_recommendations(emotion):

    emotion = emotion.lower()
    normalized_emotion = normalize_emotion(emotion)
    recommendations = recommendations_dict.get(normalized_emotion, ["No specific recommendation available."])

    return recommendations

def analyze_emotion_from_image(image_path):

    try:
        print("Analyzing emotion...")
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

        print("Raw analysis result:", analysis)


        if isinstance(analysis, list):
            analysis = analysis[0]

        primary_emotion = analysis['dominant_emotion']
        print(f"Detected Emotion: {primary_emotion}")
        return primary_emotion
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return None


def detect_and_crop_face(image):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected.")
        return None

    x, y, w, h = faces[0]
    cropped_face = image[y:y + h, x:x + w]
    return cropped_face


def capture_and_analyze_emotion():

    cap = cv2.VideoCapture(0)
    print("Press 's' to take a photo or 'q' to quit.")
    image_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Press 's' to capture or 'q' to quit.", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Photo captured.")
            cropped_face = detect_and_crop_face(frame)
            if cropped_face is not None:
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, cropped_face)
                break
            else:
                print("No face detected. Please try again.")
        elif key == ord('q'):
            print("Exiting without capturing.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    if image_path is not None:
        try:
            print("Analyzing emotion...")
            analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)


            print("Raw analysis result:", analysis)


            if isinstance(analysis, list):
                analysis = analysis[0]


            primary_emotion = analysis['dominant_emotion']
            print(f"Detected Emotion: {primary_emotion}")
            return primary_emotion
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return None
    else:
        print("No image captured.")
        return None



if __name__ == "__main__":
    emotion = capture_and_analyze_emotion()
    if emotion:
        recommendations = get_recommendations(emotion)
        print(f"Recommendations for {emotion}: {recommendations}")
    else:
        print("No emotion detected, using default recommendations.")
        print("Recommendations: No specific recommendation available.")