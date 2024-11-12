import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('traffic_sign_model.h5')

labels = {
    0: "Speed Limit 20 km/h", 
    1: "Speed Limit 30 km/h", 
    2: "Speed Limit 50 km/h"
}

hindi_translations = {
    "Speed Limit 20 km/h": "गति सीमा 20 किमी/घंटा",
    "Speed Limit 30 km/h": "गति सीमा 30 किमी/घंटा",
    "Speed Limit 50 km/h": "गति सीमा 50 किमी/घंटा"
}

def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_signal(frame):
    processed_frame = preprocess_image(frame)
    predictions = model.predict(processed_frame)
    class_id = np.argmax(predictions)
    label = labels.get(class_id, "Unknown")
    hindi_label = hindi_translations.get(label, "अनजान संकेत")
    return label, hindi_label

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    label, hindi_label = predict_signal(frame)
    
    cv2.putText(frame, f"Signal: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Hindi: {hindi_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Traffic Signal Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
