# simple_controller_act_3_1_Equipo9.py
# Controlador autónomo con SVM para Webots R2023b
#
# TECNOLOGICO DE MONTERREY - MNA
# Navegación autónoma - Actividad 3.1 - Detección de Peatones con SVM


from controller import Display, Keyboard, Robot, Camera, Lidar
from vehicle import Car, Driver
import numpy as np
import cv2
import joblib
from skimage.feature import hog
import time

# ─── Parámetros de conducción ─────────────────────────────────────────────────
MAX_ANGLE     = 0.5   
CRUISE_SPEED  = 50    
DEFAULT_ANGLE = 0.0   
MIN_SLOPE     = 0.3
KP = 0.005
KI = 0.001
KD = 0.000001

# ─── Funciones LiDAR ─────────────────────────────────────────────────────────
def process_sick_data(lidar_data, sick_width, sick_fov):
    """
    Filtra el LiDAR para revisar solo el centro (aprox 30 grados)
    y detectar obstáculos a menos de 20 metros.
    """
    HALF_AREA = 15 # 15 a la izquierda y 15 a la derecha (30 grados)
    collision_count = 0
    
    # Calcular índices centrales
    center_idx = int(sick_width / 2)
    start_idx = center_idx - HALF_AREA
    end_idx = center_idx + HALF_AREA
    
    for x in range(start_idx, end_idx):
        # LiDAR devuelve infinito si no hay rebote, limitamos a 20m
        if lidar_data[x] < 20.0:
            collision_count += 1
            
    # Si hay colisiones en el cono central, detectamos obstáculo
    if collision_count > 0:
        return True
    return False

# ─── Funciones SVM y Sliding Window ──────────────────────────────────────────
def extract_hog_features(img_window):
    """Extrae características HOG de una ventana de 64x128."""
    features = hog(img_window, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    return features

def detect_pedestrian_with_svm(image, svm_model):
    """
    Implementa Sliding Window para buscar peatones en la imagen.
    La cámara es de 256x128. El SVM espera ventanas de 64x128.
    """
    # Chequeo de seguridad para evitar crashes de memoria con OpenCV
    if image is None or image.size == 0:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    window_width = 64
    window_height = 128
    step_size = 16 # Ajuste de rendimiento
    
    # Deslizar la ventana horizontalmente a través de la imagen
    for x in range(0, gray.shape[1] - window_width + 1, step_size):
        window = gray[0:window_height, x:x + window_width]
        
        # Extraer características y predecir
        features = extract_hog_features(window)
        prediction = svm_model.predict([features])
        
        if prediction[0] == 1:
            return True # Peatón detectado en esta ventana
            
    return False # No se encontraron peatones

# ─── Funciones de procesamiento de imagen originales ─────────────────────────
def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))

def greyscale_cv2(image):
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

def canny_edge_cv2(image, low_threshold=30, high_threshold=80):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.Canny(blurred, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)

def hough_lines(image, rho=1, theta=np.pi / 180, threshold=15, min_line_length=20, max_line_gap=200):
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

# ─── Controlador PID Original ────────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, measurement, dt):
        error = measurement - self.setpoint          
        P_out = self.Kp * error
        self.integral += error * dt                  
        I_out = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        D_out = self.Kd * derivative
        self.prev_error = error
        return P_out + I_out + D_out

    def reset_integral(self):
        self.integral = 0.0

def find_closest_line_midpoint(lines, setpoint):
    if lines is None: return None
    best_x = None
    best_dist = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 or abs(dy / dx) < MIN_SLOPE: continue
        mid_x = (x1 + x2) / 2.0
        dist = abs(mid_x - setpoint)
        if dist < best_dist:
            best_dist = dist
            best_x = mid_x
    return best_x


def display_image(display, image):
    """Muestra imagen procesada en el Display a bordo de Webots."""
    image_rgb = np.dstack((image, image, image))
    ref = display.imageNew(
        image_rgb.tobytes(), Display.RGB,
        width=image_rgb.shape[1], height=image_rgb.shape[0]
    )
    display.imagePaste(ref, 0, 0, False)


# ─── Bucle principal ──────────────────────────────────────────────────────────
def main():
    # Pequeña espera para asegurar que el motor de físicas de Webots esté listo
    time.sleep(1) 
    
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    # 1. Cargar el modelo SVM entrenado
    print("Cargando modelo SVM...")
    try:
        svm_model = joblib.load('svm_pedestrian_model.pkl')
        print("Modelo SVM cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando SVM: {e}. Asegúrate de que el archivo .pkl esté en la carpeta.")
        return

    # 2. Inicializar Sensores (Cámara y LiDAR)
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = robot.getDevice("display_image")
    
    # Habilitar el Sick LiDAR
    lidar = robot.getDevice("Sick LMS 291")
    lidar.enable(timestep)
    lidar.enablePointCloud()

    cam_w = camera.getWidth()
    cam_h = camera.getHeight()
    setpoint = cam_w / 2.0
    pid = PIDController(Kp=KP, Ki=KI, Kd=KD, setpoint=setpoint)

    prev_sim_time = robot.getTime()
    
    # Estado inicial
    driver.setCruisingSpeed(CRUISE_SPEED)
    driver.setBrakeIntensity(0.0)
    driver.setHazardFlashers(False)

    # --- VARIABLE DE ESTADO para evitar el spam y el crash ---
    last_state = "CLEAR"

    while robot.step() != -1:
        current_sim_time = robot.getTime()
        dt = current_sim_time - prev_sim_time
        if dt <= 0: dt = timestep / 1000.0   
        prev_sim_time = current_sim_time

        image = get_image(camera)
        cv2.imshow("Live Car Camera", image)
        cv2.waitKey(1) # Required for OpenCV to refresh the window
        
        # ── LÓGICA: Detección de Obstáculos ─────────────────────────────
        lidar_data = lidar.getRangeImage()
        sick_width = lidar.getHorizontalResolution()
        sick_fov = lidar.getFov()
        
        obstacle_detected = process_sick_data(lidar_data, sick_width, sick_fov)
        
        if obstacle_detected:
            # Freno de emergencia general y aplicación de frenos físicos
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
            
            # Pasar imagen de cámara por el SVM
            is_pedestrian = detect_pedestrian_with_svm(image, svm_model)
            current_state = "PEDESTRIAN" if is_pedestrian else "BARREL"
            
            # SOLO actuar si el estado cambió (esto detiene el spam de comandos y el crash)
            if current_state != last_state:
                if is_pedestrian:
                    driver.setHazardFlashers(False)
                    print("PEATON DETECTADO - Freno activo (Luces OFF)")
                else:
                    driver.setHazardFlashers(True)
                    print("BARRIL DETECTADO - Freno activo (Luces ON)")
                
                # Actualizar el estado
                last_state = current_state
                
        else:
            # Solo resetear si veníamos de un estado de detección
            if last_state != "CLEAR":
                driver.setBrakeIntensity(0.0)
                driver.setCruisingSpeed(CRUISE_SPEED)
                driver.setHazardFlashers(False)
                pid.reset_integral()
                print("CAMINO DESPEJADO - Reanudando velocidad de crucero") 
                last_state = "CLEAR"

            # ── Lógica de Seguimiento de Línea (Solo si no hay obstáculo) ─────
            grey = greyscale_cv2(image)
            edges = canny_edge_cv2(grey, low_threshold=30, high_threshold=80)
            
            h, w = edges.shape
            roi_vertices = np.array([[
                (0, h),            
                (int(w * 0.40), int(h * 0.5)), 
                (int(w * 0.60), int(h * 0.5)), 
                (w, h),            
            ]], dtype=np.int32)
            cropped = region_of_interest(edges, roi_vertices)

            display_image(display_img, cropped)

            lines = hough_lines(cropped, rho=1, theta=np.pi / 180, threshold=15, min_line_length=20, max_line_gap=200)
            mid_x = find_closest_line_midpoint(lines, setpoint)

            if mid_x is not None:
                raw_angle = pid.compute(mid_x, dt)
                angle = max(-MAX_ANGLE, min(MAX_ANGLE, raw_angle))  
            else:
                raw_angle = 0.0
                angle = DEFAULT_ANGLE
                pid.reset_integral()   

            driver.setSteeringAngle(angle)

if __name__ == "__main__":
    main()