import tensorflow as tf
import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import load_img
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import pyttsx3
import time

# load model
model = load_model("models/best_model.h5")
shape = 224

#model = load_model("models/EmotionDetectionModel.h5")
#shape = 48

N = 5
history = []
emotions_dict = {
    'happy': "Glad you're happy! Do you want to tell me about your day?",
    'sad': "You look a bit sad. Do you want to tell me what happened?",
    'surprise': "Tell me why you look so surprised!",
    'angry': "You look a bit angry. Just take a breath....then tell me what is upsetting you.",
    'neutral': "What would you like to talk about today?",
# unused:
   'fear': "Are you anxious about anything? Don't worry, you'll be fine.",
   'disgust': "Did you see anything that you find disgusting?",
}

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
engine.setProperty("rate", 150)
#engine.setProperty("pitch", 120)
#voices = engine.getProperty('voices')
#engine.setProperty('voice', )

last_emotion = ''
fear_counter = 0
fear_rate = 2

emotions_to_skip = () #('disgust', 'fear')
counter = 0

engine.say("Hi, how are you today?")
engine.runAndWait()

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (shape, shape))
        img_pixels = tf.keras.utils.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        if predicted_emotion in emotions_to_skip:
           continue
        # if predicted_emotion == 'fear':
        #    fear_counter += 1
        #    if fear_counter % fear_rate == 0:
        #       continue
        if predicted_emotion == last_emotion:
            break
        last_emotion = predicted_emotion
        print(predicted_emotion)
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        engine.say(emotions_dict[predicted_emotion])
        engine.runAndWait()

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
