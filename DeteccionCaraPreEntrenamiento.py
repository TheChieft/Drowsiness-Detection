import cv2
from deepface import DeepFace
import time
import threading
import multiprocessing as mp

# Cargar el clasificador pre-entrenado de OpenCV para la detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video desde la cámara web
vid = cv2.VideoCapture(0)

# Nombre y ruta de la imagen de referencia
name = 'Angela_Pinzon'
__path__ = f'./{name}.jpeg'
reference_image = cv2.imread(__path__)
# Reducir el tamaño de la imagen de referencia para mejorar el rendimiento
reference_image = cv2.resize(reference_image, (160, 160))

print("Cargando imagen de referencia...")
if reference_image is None:
    print("Error: No se pudo cargar la imagen de referencia.")
    exit()

# Definir el tiempo entre frames para lograr 24 FPS
fps = 24
delay = 1.0 / fps

while True:
    start_time = time.time()  # Tiempo de inicio de la iteración

    ret, frame = vid.read()

    if not ret:
        print("Error al capturar el video")
        break

    # Convertir a escala de grises para la detección de caras
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Recortar la cara detectada
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Verificar la cara detectada con la imagen de referencia usando DeepFace
            result = DeepFace.verify(face_roi, reference_image, model_name='Facenet', distance_metric='euclidean')
            if result['verified']:
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Mostrar la distancia de la imagen de referencia al lado
            cv2.putText(frame, f"{result['distance']:.2f}", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Error durante la verificación: {e}")
            cv2.putText(frame, 'Error', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar el frame con las caras y la verificación
    cv2.imshow('Reconocimiento Facial', frame)

    # Calcular el tiempo que queda para completar el frame y hacer la pausa necesaria
    elapsed_time = time.time() - start_time
    time.sleep(max(0, delay - elapsed_time))  # Pausa para mantener 24 FPS

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura de video y cerrar todas las ventanas
vid.release()
cv2.destroyAllWindows()
