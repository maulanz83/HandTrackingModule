import cv2
import mediapipe as mp
import time
# Membuka kamera
cap = cv2.VideoCapture(0)
# Inisialisasi Mediapipe untuk deteksi tangan dan gambar
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
# Variabel untuk menghitung FPS
pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    if not success:
        print("Gagal mengambil gambar dari kamera")
        break
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
         for handsLms in result.multi_hand_landmarks:
              for id, lm in enumerate(handsLms.landmark):
                   #print(id,lm)
                   h,w,c = img.shape
                   cx, cy = int(lm.x*w), int(lm.y*h)
                   print(id,cx,cy)
                   if id == 0:
                    cv2.circle(img, 
                               (cx, cy), # Koordinat pusat lingkaran
                               15,       # Ukuran radius lingkaran
                               (255, 0, 255), # Warna lingkaran (magenta)
                               cv2.FILLED) # Mengisi lingkaran
            # Menggambar semua landmark dan koneksi pada tangan
              mpDraw.draw_landmarks(img,handsLms,mpHands.HAND_CONNECTIONS)

    # Menampilkan dan Menghitung FPS   

    cTime = time.time()    
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    # Tampilkan gambar
    cv2.imshow("Image", img)

        # Menunggu input dari keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Rilis sumber daya kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
    

