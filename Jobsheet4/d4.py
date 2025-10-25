import cv2
from cvzone.HandTrackingModule import HandDetector

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
    # Baca frame dari kamera
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan (flipType=True untuk tampilan cermin)
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]  # Dictionary berisi "lmList", "bbox", dll.

        # Deteksi status jari (0 = turun, 1 = naik)
        fingers = detector.fingersUp(hand)

        # Hitung jumlah jari yang terangkat
        count = sum(fingers)

        # Tampilkan hasil di layar
        cv2.putText(
            img,
            f"Fingers: {count} {fingers}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Tampilkan hasil
    cv2.imshow("Hands + Fingers", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
