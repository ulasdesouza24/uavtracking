import time
import cv2
from ultralytics import YOLO
import numpy as np
#from pymavlink import mavutil

# Pixhawk ile bağlantı kurun (örneğin: USB bağlantısı üzerinden)
#master = mavutil.mavlink_connection('COM3', baud=115200)  # COM3 yerine kendi Pixhawk portunuzu yazın
#master.wait_heartbeat()

# Kullanılacak modelin adı belirtilmeli
model = YOLO("C:\\Users\\ulasdesouza\\Desktop\\bismillah\\best6.pt")
cap = cv2.VideoCapture("C:\\Users\\ulasdesouza\\Desktop\\bismillah\\sinasenucus2.mp4")
#cap = cv2.VideoCapture(0)
classNames = ["siha"]

prev_frame_time = 0
rotation_angle = 0
rotation_step = 7  # Maksimum dönüş açısı
rotation_step_min = 4  # Minimum dönüş açısı 
rotation_smoothness = 0.5 # Dönüşün yumuşaklığı
is_rotated = False
last_detection_time = time.time()
detection_locked_time = 0

start_time = time.time()

# Model uçak resminin dosya yolu
model_ucak_resmi = "C:\\Users\\ulasdesouza\\Desktop\\bismillah\\model_ucak.png"
# Gösterge paneli görseller
hsi_case_img = cv2.imread("C:\\Users\\ulasdesouza\\Desktop\\bismillah\\hsi_case_v2.png")
hsi_face_img = cv2.imread("C:\\Users\\ulasdesouza\\Desktop\\bismillah\\hsi_face_v3.png")

# Gösterge paneli görsellerini %50 oranında küçültme
scale_factor = 0.5
hsi_case_resized = cv2.resize(hsi_case_img, (int(hsi_case_img.shape[1] * scale_factor), int(hsi_case_img.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)
hsi_face_resized = cv2.resize(hsi_face_img, (int(hsi_face_img.shape[1] * scale_factor), int(hsi_face_img.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

# Küçültülen görselleri kaydetme (opsiyonel)
cv2.imwrite("hsi_case_resized.png", hsi_case_resized)
cv2.imwrite("hsi_face_resized.png", hsi_face_resized)

# Alfa kanalı ekleme fonksiyonu
def ensure_alpha_channel(img):
    if img.shape[2] == 3:  # RGB formatı
        alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
        img = np.concatenate((img, alpha_channel), axis=2)
    return img

hsi_case_resized = ensure_alpha_channel(hsi_case_resized)
hsi_face_resized = ensure_alpha_channel(hsi_face_resized)


# Yarışma kurallarına göre ölçeklendirilmiş çerçeve
def dikdortgen(frame, width, height):
    corner1 = (int(width / 4), int(height / 10))
    corner3 = (3 * int(width / 4), 9 * int(height / 10))
    cv2.rectangle(frame, corner1, corner3, (0, 255, 0), 2)
    return frame, corner1, corner3

# Nesne takibi yapılması için gerekli olan fonksiyon
def tanimlama(results, frame, corner1, corner3):
    global rotation_angle, is_rotated, last_detection_time, detection_locked_time, target_box, x1, y1, x2, y2
    highest_confidence = 0
    target_box = None
    target_locked = False
    
    try:
        if results:
            last_detection_time = time.time()  # Son tespit zamanı güncellenir
            r = results[0]
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if currentClass == "siha":
                    confidence = float(box.conf)
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        target_box = box
                    if target_box:
                        x1, y1, x2, y2 = map(int, target_box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        if corner1[0] <= center_x <= corner3[0] and corner1[1] <= center_y <= corner3[1]:
                            target_locked = True
                            detection_locked_time = time.time()
                            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                            cv2.line(frame, (frame.shape[1] // 2, frame.shape[0] // 2), (center_x, center_y), (0, 255, 0), 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{currentClass}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if target_locked:
                cv2.putText(frame, "Hedefe Kilitlendi", (370, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Hedef Yok", (370, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                is_rotated = False  # Hedef kaybolduğunda döndürüldü işaretini kaldır
            return frame
        else:
            cv2.putText(frame, "Hedef Yok", (370, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            is_rotated = False  # Hedef kaybolduğunda döndürüldü işaretini kaldır
        return frame
    except Exception as e:
        print(e)
        return frame

# Ekranda FPS değerini göstermek için gerekli olan fonksiyon
def fps_gosterme(frame, prev_frame_time):
    new_frame_time = time.time()
    if prev_frame_time != 0:
        time_diff = new_frame_time - prev_frame_time
        fps = 1.0 / time_diff
    else:
        fps = 0.0
    prev_frame_time = new_frame_time

    # FPS değerini ekrana yazdırma
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return prev_frame_time

# Geçen süreyi ekrana yazdırmak için gerekli fonksiyon
def sure_gosterme(frame, start_time):
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # Geçen süreyi ekrana yazdırma
    height, width, _ = frame.shape
    cv2.putText(frame, f"Gecen Zaman: {elapsed_time_str}", (width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Model uçak simülasyonunu göstermek için gerekli fonksiyon
def ucak_simulasyonu(frame):
    global rotation_angle
    height, width, _ = frame.shape
    
    # Model uçak resmini yükle
    airplane_img = cv2.imread(model_ucak_resmi, cv2.IMREAD_UNCHANGED)
    airplane_height, airplane_width, _ = airplane_img.shape

    # Uçak resmi için uygun bir boyutlandırma yap
    scale_factor = min(width / airplane_width, height / airplane_height)
    new_size = (int(airplane_width * scale_factor), int(airplane_height * scale_factor))
    resized_airplane_img = cv2.resize(airplane_img, new_size)

    resized_airplane_height, resized_airplane_width, _ = resized_airplane_img.shape
    
    # Uçağın pozisyonunu belirle
    current_time = time.time()
    time_since_last_detection = current_time - last_detection_time
    time_since_last_locked_detection = current_time - detection_locked_time
    
    frame_center_x = width // 2
    frame_center_y = height // 2

    if time_since_last_locked_detection > 3:
        # 3 saniyeden fazla kilitlenme varsa, uçağı yavaş yavaş ortala
        if abs(rotation_angle) > rotation_step_min:
            rotation_angle -= np.sign(rotation_angle) * rotation_step_min  # Rotation angle yavaş yavaş sıfırlanır
        else:
            rotation_angle = 0
        
        # Uçağın ekranın merkezine yerleştirilmesi
        airplane_x = (width - resized_airplane_width) // 2
        airplane_y = (height - resized_airplane_height) // 2
    else:
        if time_since_last_detection > 1:
            # 1 saniye boyunca nesne tespit edilmediyse, uçağı yavaş yavaş ortala
            if abs(rotation_angle) > rotation_step_min:
                rotation_angle -= np.sign(rotation_angle) * rotation_step  # Rotation angle yavaş yavaş sıfırlanır
            else:
                rotation_angle = 0

            # Uçak ekranın alt kısmında ve merkezde
            airplane_x = (width - resized_airplane_width) // 2
            airplane_y = height - resized_airplane_height - 10
        else:
            # Hedefe kilitlenildiğinde ve 3 saniye geçmediyse, uçağı ekranın alt kısmında ve merkezde tut
            airplane_x = (width - resized_airplane_width) // 2
            airplane_y = height - resized_airplane_height - 10
            
            # Nesnenin merkezinin yeşil dikdörtgenin merkezine olan uzaklığı
            if target_box:
                object_center_x = (x1 + x2) // 2
                object_center_y = (y1 + y2) // 2
                distance_x = object_center_x - frame_center_x
                
                # Nesnenin merkezinin yeşil dikdörtgenin merkezine olan uzaklığına göre dönüş açısını ayarla
                max_distance_x = width // 2  # Maksimum mesafe değeri
                desired_rotation_angle = (distance_x / max_distance_x) * rotation_step * 5

                # Dönüş açısını yumuşat
                rotation_angle += (desired_rotation_angle - rotation_angle) * rotation_smoothness

    # Resmin merkezini hesapla
    center = (resized_airplane_width // 2, resized_airplane_height // 2)
    
    # Dönüş matrisi oluştur
    matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_airplane = cv2.warpAffine(resized_airplane_img, matrix, (resized_airplane_width, resized_airplane_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    # Simülasyon arka planını oluştur (beyaz arka plan)
    simulasyon_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Uçak resmi ile arka planı birleştir
    alpha_airplane = rotated_airplane[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_airplane
    for c in range(0, 3):
        simulasyon_frame[airplane_y:airplane_y + resized_airplane_height, airplane_x:airplane_x + resized_airplane_width, c] = (
            alpha_airplane * rotated_airplane[:, :, c] +
            alpha_background * simulasyon_frame[airplane_y:airplane_y + resized_airplane_height, airplane_x:airplane_x + resized_airplane_width, c]
        )

    # Simülasyon ekranını döndür
    return simulasyon_frame

def gosterge_paneli(rotation_angle):
    global hsi_case_resized, hsi_face_resized
    
    # Gösterge paneli boyutları
    panel_height, panel_width, _ = hsi_case_resized.shape

    # Gösterge panelinin iç kısmını döndür
    rotation_matrix = cv2.getRotationMatrix2D((panel_width // 2, panel_height // 2), -rotation_angle, 1.0)
    rotated_face_img = cv2.warpAffine(hsi_face_resized, rotation_matrix, (panel_width, panel_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Gösterge paneli çerçevesi ve dönen iç kısmını birleştir
    combined_img = hsi_case_resized.copy()
    
    # Çerçeve ve dönen iç kısmını birleştir
    # Çerçeveyi arka plan olarak kullanarak dönen iç kısmı yerleştir
    alpha_face = rotated_face_img[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_face

    for c in range(0, 3):
        combined_img[:, :, c] = (alpha_face * rotated_face_img[:, :, c] + alpha_background * combined_img[:, :, c])

    return combined_img

# Gösterge paneli resimlerini RGBA formatına dönüştür
hsi_case_resized = ensure_alpha_channel(hsi_case_resized)
hsi_face_resized = ensure_alpha_channel(hsi_face_resized)


#def set_servo(servo_pin, pwm_value):
    # MAV_CMD_DO_SET_SERVO komutu ile servoyu kontrol et
    #master.mav.command_long_send(
        #master.target_system,
        #master.target_component,
        #mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        #0,  # Confirmation
        #servo_pin,  # Servo number
        #pwm_value,  # PWM value
        #0, 0, 0, 0, 0  # Unused parameters
    #)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frameCopy = frame.copy()

    prev_frame_time = fps_gosterme(frame, prev_frame_time)

    height, width, _ = frameCopy.shape
    d_frame, corner1, corner3 = dikdortgen(frame, width, height)

    results = model.predict(frame)
    frame = tanimlama(results, d_frame, corner1, corner3)
    
    sure_gosterme(frame, start_time)

    # Model uçak simülasyonunu oluşturun
    simulasyon_frame = ucak_simulasyonu(d_frame)
    
    # Uçak dönüş göstergesini oluşturun
    gauge_frame = gosterge_paneli(rotation_angle)

    # Sonuçları ekranda göster
    cv2.imshow("Video", frame)
    cv2.imshow("Simulasyon", simulasyon_frame)
    cv2.imshow("Gosterge", gauge_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()