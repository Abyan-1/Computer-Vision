import cv2
import numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

# Mode awal: squat atau pushup (toggle dengan 'm')
MODE = "squat"

# Ambang sudut lutut (derajat) untuk squat
KNEE_DOWN, KNEE_UP = 80, 160

# Ambang rasio posisi tubuh untuk push-up
DOWN_R, UP_R = 0.85, 1.00

# Jumlah frame konsisten sebelum pergantian status (debounce)
SAMPLE_OK = 4

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi pose detector
detector = PoseDetector(
    staticMode=False,
    modelComplexity=1,
    enableSegmentation=False,
    detectionCon=0.5,
    trackCon=0.5
)

count, state = 0, "up"
debounce = deque(maxlen=6)


def ratio_pushup(lm):
    """
    Hitung rasio (shoulder–wrist) / (shoulder–hip)
    Rasio kecil menandakan posisi tubuh turun (push-up down)
    """
    # Gunakan sisi kiri: 11=shoulderL, 15=wristL, 23=hipL
    sh = np.array(lm[11][1:3])
    wr = np.array(lm[15][1:3])
    hp = np.array(lm[23][1:3])
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)


while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi pose
    img = detector.findPose(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)
    flag = None

    if lmList:
        if MODE == "squat":
            # Hitung rata-rata sudut lutut kiri & kanan
            angL, img = detector.findAngle(
                lmList[23][0:2],
                lmList[25][0:2],
                lmList[27][0:2],
                img=img,
                color=(0, 0, 255),
                scale=10
            )

            angR, img = detector.findAngle(
                lmList[24][0:2],
                lmList[26][0:2],
                lmList[28][0:2],
                img=img,
                color=(0, 255, 0),
                scale=10
            )

            ang = (angL + angR) / 2.0

            # Tentukan posisi squat
            if ang < KNEE_DOWN:
                flag = "down"
            elif ang > KNEE_UP:
                flag = "up"

            cv2.putText(
                img,
                f"Knee: {ang:5.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:  # push-up mode
            r = ratio_pushup(lmList)

            # Tentukan posisi push-up
            if r < DOWN_R:
                flag = "down"
            elif r > UP_R:
                flag = "up"

            cv2.putText(
                img,
                f"Ratio: {r:4.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # Tambahkan flag ke buffer debounce
        debounce.append(flag)

        # Deteksi transisi posisi (up -> down -> up)
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1

    # Tampilkan info di layar
    cv2.putText(
        img,
        f"Mode: {MODE.upper()}  Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )
    cv2.putText(
        img,
        f"State: {state}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Pose Counter", img)

    # Input keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
