import cv2
import numpy as np
def main():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check permissions / device index.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(60, 60)
        )
        for (x, y, w, h) in faces:
            p1 = (x + w // 2, y)        
            p2 = (x, y + h)            
            p3 = (x + w, y + h)     
            pts = np.array([
                (x, y),         
                (x + w, y),         
                (x + w, y + h),     
                (x, y + h)     
            ], dtype=np.int32)

            cv2.polylines(
                frame,
                [pts],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow("Face Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
main()
