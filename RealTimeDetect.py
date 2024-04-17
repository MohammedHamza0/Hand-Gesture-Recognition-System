import cv2
import mediapipe as mp

# Initialize the video capture device
video_capture = cv2.VideoCapture(0)

# Load the Haar cascades for face, eye, and smile detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Initialize MediaPipe Hands for hand gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Faces for facial landmark detection
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()
face_drawing = mp.solutions.drawing_utils
drawing_spec = face_drawing.DrawingSpec(thickness=1, circle_radius=1)

while True:
    ret, frame = video_capture.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results_hands = hands.process(rgb_frame)
    
    # Process the frame with MediaPipe Faces
    results_faces = face_mesh.process(rgb_frame)
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 30), 1)
        
        # Detect eyes within the face region
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Iterate over detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw bounding box around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (200, 255, 150), 1)
            # Mark the center of each eye with an 'X'
            eye_x = int((ex+(ew/2))) + 2
            eye_y = int((ey+(eh/2))) + 10
            cv2.putText(roi_color, 'X', (eye_x, eye_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 50, 255), 3, cv2.LINE_4)
        
        # Detect smiles within the face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=19)
        
        # Iterate over detected smiles
        for (sx, sy, sw, sh) in smiles:
            # Draw bounding box around the smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (200, 0, 150), 1)
            cv2.putText(roi_color, 'Smile', (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 90, 50), 3, cv2.LINE_4)
            
    # Draw hand landmarks if hands are detected
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    # Draw facial landmarks if faces are detected
    if results_faces.multi_face_landmarks:
        for face_landmarks in results_faces.multi_face_landmarks:
            face_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS, drawing_spec)
        
    # Display the resulting frame
    cv2.imshow("Video Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video_capture.release()
cv2.destroyAllWindows()
