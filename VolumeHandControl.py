import cv2
import time
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0.7, maxHands=2)

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Brightness smoothing
prevBright = sbc.get_brightness()[0]
lastBrightTime = 0

pTime = 0
volPer = 0
brightPer = prevBright

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    handType = detector.handType

    if lmList:
        fingers = detector.fingersUp()
        length, img, _ = detector.findDistance(4, 8, img)

        # ---------------- LEFT HAND → VOLUME ----------------
        if handType == "Left" and len(fingers) == 5 and not fingers[4]:
            volPer = np.interp(length, [40, 180], [0, 100])
            volPer = int(10 * round(volPer / 10))
            volume.SetMasterVolumeLevelScalar(volPer / 100, None)

        # ---------------- RIGHT HAND → BRIGHTNESS ----------------
        if handType == "Right" and len(fingers) == 5 and not fingers[3]:
            target = np.interp(length, [40, 180], [0, 100])
            target = np.clip(target, 0, 100)

            brightPer = int(prevBright + 0.2 * (target - prevBright))
            now = time.time()

            if abs(brightPer - prevBright) >= 2 and (now - lastBrightTime) > 0.12:
                sbc.set_brightness(brightPer)
                prevBright = brightPer
                lastBrightTime = now

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Vol: {int(volPer)}%', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Bright: {int(brightPer)}%', (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
