import cv2
from deepface import DeepFace
import threading as threading

# Cargar el clasificador pre-entrenado de OpenCV para la detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)  # En Linux, no especificamos un backend específico, se usará el predeterminado

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

counter = 0
face_match = False
name = 'Andres_Yanez'

# Asegúrate de que la imagen de referencia esté en la misma carpeta
reference_image = cv2.imread(f'{name}.jpeg', cv2.IMREAD_GRAYSCALE)

if reference_image is None:
    print("Error: No se pudo cargar la imagen de referencia.")
    exit()

def check_face(frame, reference_image):
    global face_match
    try:
        # Convertir el frame a escala de grises para compararlo con la imagen de referencia
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if DeepFace.verify(reference_image, frame_gray, model_name='Facenet', distance_metric='euclidean')['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    
    # Verificar si el frame fue capturado correctamente
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame, reference_image)).start()
            except ValueError:
                pass
        counter += 1
    
        # Mostrar resultados de la verificación facial
        if face_match:
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Mostrar el frame en una única ventana
        cv2.imshow('Detección de Rostros', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    
# Liberar la cámara y cerrar todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
