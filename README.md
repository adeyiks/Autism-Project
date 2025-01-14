# AUTISM PROJECT
Predicting Onset of Autism in Kids using Face Segments

## DATA.
For this project, i got photos of kids from an Autism Dataset (Autism-Image- Dataset) that is available on Kaggle. Most of the children in the sample were between the ages of two and eight, while their ages ranged from two to fourteen. The pictures were all in the JPEG format and were typical 2D RGB colour pictures. There were 2,940 photographs of children in this dataset overall, divided into two categories with 1,470 in each category.
 ![full face](./folder/images/full picture.jpg)

 ## DATA PREPARATION.
 ### MEDIAPIPE.
 - Using the Mediapipe, i created a face-mesh landmarks shown in the image below.

```python

!pip install mediapipe opencv-python
from google.colab.patches import cv2_imshow
import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
```
```

# Load the image
image_path = "/content/This-paper-introduces-MaskFaceGAN-a-novel-approach-to-face-attribute-editing-capable-of.png"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform facial landmark detection
results = face_mesh.process(rgb_image)
```
```
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw landmarks on the image
        for idx, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            # Draw a small circle at each landmark position
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            # Optionally, put the landmark index number
            cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
```
```
    # Save the image with landmarks
    cv2.imwrite("output_image_with_landmarks.jpg", image)
```
![Face Mesh](./folder/images/face mesh.jpg).

### FACE SEGMENT EXTRACTION.
   
