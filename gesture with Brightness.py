import cv2
import time
import numpy as np
import HandTrackingModule as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import pygetwindow as gw


# ==============================
# CAMERA SETTINGS
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Webcam not detected.")
    exit()

detector = htm.handDetector(
    detectionCon=0.7,
    maxHands=2
)


# ==============================
# VOLUME CONTROL
# ==============================
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_,
        CLSCTX_ALL,
        None
    )
    volume = cast(
        interface,
        POINTER(IAudioEndpointVolume)
    )
    volumeAvailable = True
except Exception as e:
    print("Volume initialization failed:", e)
    volumeAvailable = False


# ==============================
# BRIGHTNESS CONTROL
# ==============================
try:
    prevBright = sbc.get_brightness()[0]
except Exception as e:
    print("Brightness initialization failed:", e)
    prevBright = 50

brightPer = prevBright
lastBrightTime = 0


# ==============================
# WINDOW CONTROL
# ==============================
gestureStart = 0
actionDone = False

# Gesture:
# Open right hand  -> Maximize
# Fist right hand  -> Minimize
GESTURE_HOLD_TIME = 3


# ==============================
# FPS
# ==============================
pTime = 0
volPer = 0


# ==============================
# MAIN LOOP
# ==============================
while True:

    success, img = cap.read()

    if not success:
        print("Failed to read webcam.")
        break

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    # Default values
    fingers = []
    handType = None

    # Check detected hand
    if lmList:

        fingers = detector.fingersUp()

        # Use hand type only if module provides it
        handType = getattr(detector, "handType", None)

        # Distance between thumb and index finger
        length, img, info = detector.findDistance(
            4, 8, img
        )

        # ==============================
        # LEFT HAND: VOLUME CONTROL
        # ==============================
        if (
            handType == "Left"
            and len(fingers) == 5
            and fingers[4] == 0
            and volumeAvailable
        ):

            volPer = int(
                np.interp(length, [40, 180], [0, 100])
            )

            volPer = int(np.clip(volPer, 0, 100))

            try:
                volume.SetMasterVolumeLevelScalar(
                    volPer / 100,
                    None
                )
            except Exception as e:
                print("Volume error:", e)

        # ==============================
        # RIGHT HAND: BRIGHTNESS CONTROL
        # ==============================
        if (
            handType == "Right"
            and len(fingers) == 5
            and fingers[3] == 0
        ):

            target = int(
                np.interp(length, [40, 180], [0, 100])
            )

            target = int(np.clip(target, 0, 100))

            # Smooth brightness transition
            brightPer = int(
                prevBright + 0.2 * (target - prevBright)
            )

            now = time.time()

            if (
                abs(brightPer - prevBright) >= 2
                and now - lastBrightTime > 0.15
            ):
                try:
                    sbc.set_brightness(brightPer)
                    prevBright = brightPer
                    lastBrightTime = now
                except Exception as e:
                    print("Brightness error:", e)

        # ==============================
        # WINDOW CONTROL
        # ==============================
        # Open right hand = Maximize
        # Fist right hand = Minimize
        if handType == "Right" and len(fingers) == 5:

            isOpenPalm = all(f == 1 for f in fingers)
            isFist = all(f == 0 for f in fingers)

            if isOpenPalm or isFist:

                if gestureStart == 0:
                    gestureStart = time.time()

                elapsed = time.time() - gestureStart

                if elapsed >= GESTURE_HOLD_TIME and not actionDone:

                    try:
                        activeWin = gw.getActiveWindow()

                        if activeWin:
                            if isOpenPalm:
                                activeWin.maximize()
                                print("Window Maximized")

                            elif isFist:
                                activeWin.minimize()
                                print("Window Minimized")

                    except Exception as e:
                        print("Window control error:", e)

                    actionDone = True

            else:
                gestureStart = 0
                actionDone = False

        else:
            gestureStart = 0
            actionDone = False

    # ==============================
    # DISPLAY FPS
    # ==============================
    cTime = time.time()

    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Volume: {int(volPer)}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        img,
        f"Brightness: {int(brightPer)}%",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img,
        "Press Q to Exit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Gesture Volume Brightness Control", img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()

print("Gesture Control Stopped.")