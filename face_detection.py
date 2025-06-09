import cv2

# Load Haar cascades for face, eye, and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start webcam
video_cap = cv2.VideoCapture(0)

# Privacy mode toggle
blur_faces = False

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Display number of faces
    cv2.putText(frame, f'Faces Detected: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    for (x, y, w, h) in faces:
        if blur_faces:
            # Blur the face
            face_region = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_region, (99, 99), 30)
        else:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Region of Interest (ROI) for eyes and smile
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)

            # Detect smile
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)

    # Display the video frame
    cv2.imshow("Face Detection - Enhanced", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('b'):
        blur_faces = not blur_faces  # Toggle blur mode

# Release resources
video_cap.release()
cv2.destroyAllWindows()
