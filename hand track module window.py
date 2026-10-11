import cv2
import mediapipe as mp
import math

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        self.lmList = []
        self.handType = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms, handInfo in zip(
                self.results.multi_hand_landmarks,
                self.results.multi_handedness
            ):
                self.handType = handInfo.classification[0].label
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList, yList = [], []

            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                xList.append(cx)
                yList.append(cy)

            bbox = min(xList), min(yList), max(xList), max(yList)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]

        # Thumb
        if self.handType == "Right":
            fingers.append(1 if self.lmList[4][1] > self.lmList[3][1] else 0)
        else:
            fingers.append(1 if self.lmList[4][1] < self.lmList[3][1] else 0)

        # Other fingers
        for id in range(1, 5):
            fingers.append(
                1 if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2] else 0
            )

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, None

    # ✅ REQUIRED FOR WINDOW CONTROL
    def palmOrientation(self):
        """
        Front → Palm facing camera
        Back  → Back of hand facing camera
        """
        wrist_y = self.lmList[0][2]
        middle_mcp_y = self.lmList[9][2]

        if middle_mcp_y < wrist_y:
            return "Front"
        else:
            return "Back"
