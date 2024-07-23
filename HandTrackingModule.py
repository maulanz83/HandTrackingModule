#Import library
import cv2
import mediapipe as mp
import time
#handDetector Class
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Inisialisasi Mediapipe untuk deteksi tangan dan gambar
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    #findHands Method
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS)
        return img 

    #findPosition Method
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                    #print(id,lm)
                h,w,c = img.shape 
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, 
                    (cx, cy), # Koordinat pusat lingkaran
                    7,       # Ukuran radius lingkaran
                    (255, 0, 0), # Warna lingkaran (magenta)
                    cv2.FILLED) # Mengisi lingkaran
        return lmList
#Main Function
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
       
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()    
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
#running script
if __name__ == "__main__":
    main()