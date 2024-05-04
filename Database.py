import cv2
import os
import numpy as np

# Re-define label dictionary with the new entries
label_dict = {
    "pipe": 0,
    "paola": 1,
    
    

}

# Prepare lists for faces and labels
faces = []
labels = []


# Directory with known faces
known_faces_dir = "known_faces"

# Load and assign labels for each image
for root, _, files in os.walk(known_faces_dir):
    for file in files:
        if file.endswith(("jpg", "jpeg", "png")):
            label = os.path.splitext(file)[0]  # Extract the label from the filename
            img_path = os.path.join(root, file)  # Full path to the image

            img = cv2.imread(img_path)  # Read the image
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            # Assign correct label to each image
            faces.append(gray_img)  # Add to list of faces
            labels.append(label_dict[label])

# Create and train the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save the trained recognizer to a file
recognizer.save("recognizer.yml")
