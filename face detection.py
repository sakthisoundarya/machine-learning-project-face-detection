import cv2
from mtcnn.mtcnn import MTCNN

# Haar Cascade Face Detection
def detect_faces_haar(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Detected Faces (Haar Cascade)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# MTCNN Face Detection
def detect_faces_mtcnn(image_path):
    detector = MTCNN()
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)
    
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Detected Faces (MTCNN)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage for Haar Cascade
image_path_haar = 'path/to/your/image.jpg'
detect_faces_haar(image_path_haar)

# Example usage for MTCNN
image_path_mtcnn = 'path/to/your/image.jpg'
detect_faces_mtcnn(image_path_mtcnn)
