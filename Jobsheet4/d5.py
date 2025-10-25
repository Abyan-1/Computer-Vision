import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


def dist(a, b):
    """Hitung jarak Euclidean antara dua titik."""
    return np.linalg.norm(np.array(a) - np.array(b))


def classify_gesture(hand):
    """
    Mengklasifikasikan gesture tangan berdasarkan posisi titik kunci.
    hand["lmList"] berisi 21 titik (x, y, z) dalam piksel (flipType=True).
    """
    lm = hand["lmList"]

    # Landmark penting
    wrist = np.array(lm[0][:2])
    thumb_tip = np.array(lm[4][:2])
    index_tip = np.array(lm[8][:2])
    middle_tip = np.array(lm[12][:2])
    ring_tip = np.array(lm[16][:2])
    pinky_tip = np.array(lm[20][:2])

    # Heuristik jarak relatif
    r_mean = np.mean([
        dist(index_tip, wrist),
        dist(middle_tip, wrist),
        dist(ring_tip, wrist),
        dist(pinky_tip, wrist),
        dist(thumb_tip, wrist)
    ])

    # Aturan pengenalan gesture
    # 1. OK gesture: ibu jari dan telunjuk saling menyentuh
    if dist(thumb_tip, index_tip) < 35:
        return "OK"

    # 2. Thumbs up: ibu jari tinggi dan jauh dari pergelangan tangan
    if (thumb_tip[1] < wrist[1] - 40) and (
        dist(thumb_tip, wrist) > 0.8 * dist(index_tip, wrist)
    ):
        return "THUMBS_UP"

    # 3. ROCK: semua jari relatif dekat (genggaman)
    if r_mean < 120:
        return "ROCK"

    # 4. PAPER: semua jari terbuka lebar
    if r_mean > 200:
        return "PAPER"

    # 5. SCISSORS: hanya telunjuk dan jari tengah terbuka
    if (
        dist(index_tip, wrist) > 180
        and dist(middle_tip, wrist) > 180
        and dist(ring_tip, wrist) < 160
        and dist(pinky_tip, wrist) < 160
    ):
        return "SCISSORS"

    # Tidak dikenali
    return "UNKNOWN"


# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi HandDetector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Klasifikasi gesture
        label = classify_gesture(hands[0])

        # Tampilkan hasil pada frame
        cv2.putText(
            img,
            f"Gesture: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

    # Tampilkan hasil
    cv2.imshow("Hand Gestures (cvzone)", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
