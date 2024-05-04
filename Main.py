import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer.yml")  # Load the saved recognizer

# Re-define label dictionary with the new entries (consistent with training code)
label_dict = {
    "pipe": 0,
    "paola": 1,
    
    
}

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture a frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]  # Correct indexing for ROI

            # Predict the label and get the confidence score
            label, confidence = recognizer.predict(roi_gray)

            # Determine the name of the recognized person
            person_name = "Unknown"
            if label in label_dict.values():  # Check if label is in the dictionary
                person_name = list(label_dict.keys())[list(label_dict.values()).index(label)]

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
