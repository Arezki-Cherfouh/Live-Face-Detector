import cv2
import numpy as np
import os
import urllib.request
# def download_file(url, filename):
#     if not os.path.exists(filename):
#         print(f"Downloading {filename}...")
#         try:
#             urllib.request.urlretrieve(url, filename)
#             print(f"{filename} downloaded.")
#         except Exception as e:
#             print(f"Failed to download {filename}: {e}")
#             raise SystemExit
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # models = {
    #     "deploy_age.prototxt": "https://raw.githubusercontent.com/ChandrikaDeb/age-gender-estimation-opencv/master/deploy_age.prototxt",
    #     "age_net.caffemodel": "https://github.com/ChandrikaDeb/age-gender-estimation-opencv/raw/master/age_net.caffemodel",
    #     "deploy_gender.prototxt": "https://raw.githubusercontent.com/ChandrikaDeb/age-gender-estimation-opencv/master/deploy_gender.prototxt",
    #     "gender_net.caffemodel": "https://github.com/ChandrikaDeb/age-gender-estimation-opencv/raw/master/gender_net.caffemodel",
    # }
    # for filename, url in models.items():
    #     path = os.path.join(base_dir, filename)
    #     download_file(url, path)
    age_net = cv2.dnn.readNetFromCaffe(os.path.join(base_dir, "deploy_age.prototxt"),os.path.join(base_dir, "age_net.caffemodel"))
    gender_net = cv2.dnn.readNetFromCaffe(os.path.join(base_dir, "deploy_gender.prototxt"),os.path.join(base_dir, "gender_net.caffemodel"))
    emotion_net = cv2.dnn.readNetFromONNX(os.path.join(base_dir, "emotion-ferplus-8.onnx"))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)','(25-32)', '(38-43)', '(48-53)', '(60-100)']
    GENDER_LIST = ['Male', 'Female']
    EMOTIONS = ['Neutral', 'Happiness', 'Surprise','Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
            emotion_blob = cv2.dnn.blobFromImage(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 1/255.0, (64, 64))
            emotion_net.setInput(emotion_blob)
            emotion_preds = emotion_net.forward()
            emotion = EMOTIONS[np.argmax(emotion_preds)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            label = f"{gender}, {age}, {emotion}"
            cv2.putText(frame, label, (x-30, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Face Age-Gender-Mood Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
main()