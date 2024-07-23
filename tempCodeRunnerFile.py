import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False , maxHands =2, detectionCon =0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # Inisialisasi Mediapipe untuk deteksi tangan dan gambar
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self,img,draw =True):
        # Variabel untuk menghitung FPS
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB)
        #print(result.multi_hand_landmarks)

        if result.multi_hand_landmarks:
            for handsLms in result.multi_hand_landmarks: 
                # Menggambar semua landmark dan koneksi pada tangan
                if draw:
                    self.mpDraw.draw_landmarks(img,handsLms,
                                             self.mpHands.HAND_CONNECTIONS)
        return img 
                #for id, lm in enumerate(handsLms.landmark):
                    #print(id,lm)
                    #h,w,c = img.shape
                    #cx, cy = int(lm.x*w), int(lm.y*h)
                    #print(id,cx,cy)
                    #cv2.circle(img, 
                                #(cx, cy), # Koordinat pusat lingkaran
                                #15,       # Ukuran radius lingkaran
                                #(255, 0, 255), # Warna lingkaran (magenta)
                                #cv2.FILLED) # Mengisi lingkaran

def main():
     # Variabel untuk menghitung FPS
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        # Menampilkan dan Menghitung FPS   
        img = detector.findhands(img)
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
    
if __name__ == "__main__":
    main()

