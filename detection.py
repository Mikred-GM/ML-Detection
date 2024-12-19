import cv2
import numpy as np
from pathlib import Path

# Paths to models
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_INTERVALS = [
    '(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
    '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)'
]

# Load models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):
    """
    Detect faces in the given frame using OpenCV's DNN module.
    :param frame: Input image
    :param confidence_threshold: Minimum confidence threshold for detections
    :return: List of bounding boxes for detected faces
    """
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([
                frame.shape[1], frame.shape[0],
                frame.shape[1], frame.shape[0]
            ])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x = max(start_x - 10, 0)
            start_y = max(start_y - 10, 0)
            end_x = min(end_x + 10, frame.shape[1])
            end_y = min(end_y + 10, frame.shape[0])
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def display_img(title, img):
    """
    Displays an image on screen and waits for a key press before closing the window.
    :param title: Title of the window
    :param img: Image to display
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an image without distortion.
    :param image: Input image
    :param width: Desired width
    :param height: Desired height
    :param inter: Interpolation method
    :return: Resized image
    """
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / h
        dim = (int(w * r), height)
    else:
        r = width / w
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def get_predictions(face_img, net_type='gender'):
    """
    Get predictions from either the gender or age network.
    :param face_img: Face image
    :param net_type: Type of prediction ('gender' or 'age')
    :return: Predicted values
    """
    blob = cv2.dnn.blobFromImage(
        image=face_img,
        scalefactor=1.0,
        size=(227, 227),
        mean=MODEL_MEAN_VALUES,
        swapRB=False,
        crop=False
    )
    
    if net_type == 'gender':
        net = gender_net
    elif net_type == 'age':
        net = age_net
        
    net.setInput(blob)
    return net.forward()

def predict_age_and_gender(input_path: str):
    """
    Predict the age and gender of faces in the provided image.
    :param input_path: Path to the input image
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f'File {input_path} does not exist.')
    
    try:
        with open(input_path, 'rb') as file:
            img = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), -1)
    except Exception as e:
        print(f'An error occurred while reading the image: {str(e)}')
        return
    
    frame = img.copy()
    if frame.shape[1] > 1280:
        frame = image_resize(frame, width=1280)
    
    faces = get_faces(frame)
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y:end_y, start_x:end_x]
        age_preds = get_predictions(face_img, 'age')
        gender_preds = get_predictions(face_img, 'gender')
        
        gender_idx = np.argmax(gender_preds[0])
        gender = GENDER_LIST[gender_idx]
        gender_confidence_score = gender_preds[0][gender_idx]
        
        age_idx = np.argmax(age_preds[0])
        age = AGE_INTERVALS[age_idx]
        age_confidence_score = age_preds[0][age_idx]
        
        label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
        y_pos = start_y - 15
        while y_pos < 15:
            y_pos += 15
            
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        cv2.putText(frame, label, (start_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)
    
    display_img("Gender Estimator", frame)
    cv2.imwrite("output.jpg", frame)

if __name__ == "__main__":
    input_path = "C:/Users/миша/Desktop/project/2.jpg"
    predict_age_and_gender(input_path)
