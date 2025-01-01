import cv2
import requests
import hashlib
import time
import os

api_upload_url = os.getenv("API_UPLOAD_URL")

def face_detection_and_send():
    cascade_path = "./filters/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    last_sent_time = 0
    send_interval = 5 
    sent_hashes = set()

    def send_frame(frame):
        try:
            _, encoded_image = cv2.imencode('.jpg', frame)
            image_data = encoded_image.tobytes()

            image_hash = hashlib.md5(image_data).hexdigest()
            if image_hash in sent_hashes:
                print("The frame already sent. Skip.")
                return

            response = requests.post(api_upload_url, files={'file': image_data})
            if response.status_code == 200:
                print("Successfully sent.")
                sent_hashes.add(image_hash)
                print(f"Response: {response.json()}")
            else:
                print(f"Error sending a frame: {response.status_code}")
        except Exception as e:
            print(f"Error sending a frame: {e}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            print(f"Обнаружено {len(faces)} лиц(а).")

            current_time = time.time()
            if current_time - last_sent_time >= send_interval:
                last_sent_time = current_time
                send_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    face_detection_and_send()

main()
