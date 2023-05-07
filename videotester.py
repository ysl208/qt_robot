import time
import cv2
import numpy as np
import pyttsx3
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from keras.preprocessing import image
from keras.models import  load_model
import tensorflow as tf

shape_ = (224, 244)
model_ = load_model("models/best_model.h5")

# Define emotions to skip and other parameters
emotions_dict_ = {
    'happy': "Glad you're happy! Do you want to tell me about your day?",
    'sad': "You look a bit sad. Do you want to tell me what happened?",
    'surprise': "Tell me why you look so surprised!",
    'angry': "You look a bit angry. Just take a breath....then tell me what is upsetting you.",
    'neutral': "What would you like to talk about today?",
    # unused:
    'fear': "Are you anxious about anything? Don't worry, you'll be fine.",
    'disgust': "Did you see anything that you find disgusting?",
}
emotions_to_skip_ = () #('disgust', 'fear')

def get_predicted_emotion(gray_img, test_img, x, y, w, h):
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
    roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
    roi_gray = cv2.resize(roi_gray, (224, 224))
    img_pixels = tf.keras.utils.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    predictions = model_.predict(img_pixels)

    # find max indexed array
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotions[max_index]
    if predicted_emotion in emotions_to_skip_:
        return None
    return predicted_emotion

def speak(engine, phrase):
    engine.say(phrase)
    engine.runAndWait()

def get_most_frequent_emotion(history, N):
    most_common_emotion = Counter(history[-N:]).most_common(1)[0]
    return most_common_emotion[0]

def detect_emotion():
    # Detect emotions in a video stream
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    highest_score = 0
    counter = 0

    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            predicted_emotion = get_predicted_emotion(gray_img, test_img, x, y, w, h)
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # speak(engine, emotions_dict_[predicted_emotion])

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion()
