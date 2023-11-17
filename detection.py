import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the pre-trained model architecture
model_architecture_path = 'facialemotionmodel.json'
with open(model_architecture_path, 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

# Load the pre-trained model weights
model_weights_path = 'facialemotionmodel.h5'
model.load_weights(model_weights_path)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face image to match the input size of the model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Normalize the pixel values
        face_roi = face_roi / 255.0

        # Make a prediction using the loaded model
        emotion_prediction = model.predict(face_roi)

        # Get the emotion label
        emotion_label = np.argmax(emotion_prediction)
        emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        predicted_emotion = emotion_map[emotion_label]

        # Draw the bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
