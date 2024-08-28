# Librerias ------------------------------------
import cv2
from time import sleep

def detect_face_and_eyes(image):
    print("Detectando caras y ojos...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detectar caras en la imagen
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detectar ojos en la imagen
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # detectar hombros
    shoulder_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_shoulder.xml')
    
    sholders = shoulder_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in sholders:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0,
        255), 2)
        
        cv2.circle(image, (x + w//2, y + h//2), 2, (0, 0, 255), -1)
        cv2.circle(image, (x, y + h//2), 2, (0, 0, 255), -1)
        cv2.circle(image, (x + w, y + h//2), 2, (0, 0, 255), -1)
        cv2.circle(image, (x + w//2, y), 2, (0, 0, 255), -1)
        cv2.circle(image, (x + w//2, y + h), 2, (0, 0, 255), -1)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Para cada cara detectada
    for (x, y, w, h) in faces:
        # Dibujar un rectangulo alrededor de la cara
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Recortar la region de interes (ROI) que contiene la cara
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Detectar ojos en la cara
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Para cada ojo detectado en la cara, poner puntos en los parpados
        
        for (ex, ey, ew, eh) in eyes:
            # Dibujar un rectangulo alrededor del ojo
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Dibujar puntos en los parpados
            cv2.circle(roi_color, (ex + ew//2, ey + eh//2), 2, (0, 0, 255), -1)
            cv2.circle(roi_color, (ex, ey + eh//2), 2, (0, 0, 255), -1)
            cv2.circle(roi_color, (ex + ew, ey + eh//2), 2, (0, 0, 255), -1)
            cv2.circle(roi_color, (ex + ew//2, ey), 2, (0, 0, 255), -1)
            cv2.circle(roi_color, (ex + ew//2, ey + eh), 2, (0, 0, 255), -1)
        
            
                
                
    return image

def show_image(image, window_name='image'):
    print("Mostrando imagen...")
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_image(file_path):
    print(f"Cargando imagen desde '{file_path}'...")
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{file_path}' not found.")
    return image

# Cargar una imagen desde un archivo
try:
    i = 2
    image = load_image(f'Modelos/face_eyes_db/face{i}.jpeg')
    # Detectar caras y ojos en la imagen
    result_image = detect_face_and_eyes(image)
    # Mostrar la imagen con las caras y ojos detectados
    show_image(result_image, 'Caras y Ojos Detectados')
    sleep(1)
    # close the window image
        
        
except FileNotFoundError as e:
    print(e)