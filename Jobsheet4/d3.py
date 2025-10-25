import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Indeks mata kiri (contoh): vertikal (159,145), horizontal (33,133)
L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133


def dist(p1, p2):
    """Hitung jarak Euclidean antara dua titik."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi objek FaceMeshDetector
# staticMode: Jika True, deteksi hanya terjadi sekali; jika False, setiap frame
# maxFaces: Jumlah maksimum wajah yang dideteksi
# minDetectionCon: Ambang kepercayaan deteksi minimum
# minTrackCon: Ambang kepercayaan pelacakan minimum
detector = FaceMeshDetector(
    staticMode=False,
    maxFaces=2,
    minDetectionCon=0.5,
    minTrackCon=0.5
)

# Variabel untuk menghitung kedipan sederhana
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3  # jumlah frame berturut-turut untuk dianggap kedipan
EYE_AR_THRESHOLD = 0.20      # ambang Eye Aspect Ratio (EAR) untuk menilai mata tertutup
is_closed = False

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi mesh wajah
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]  # List dari 468 titik (x, y)

        # Hitung jarak vertikal dan horizontal mata kiri
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])

        # Hitung rasio aspek mata (EAR)
        ear = v / (h + 1e-8)

        # Tampilkan nilai EAR di layar
        cv2.putText(
            img, f"EAR(L): {ear:.3f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )

        # Logika mendeteksi kedipan
        # Jika EAR < EYE_AR_THRESHOLD selama CLOSED_FRAMES_THRESHOLD frame -> dihitung sebagai kedipan
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        # Tampilkan jumlah kedipan pada frame
        cv2.putText(
            img, f"Blink: {blink_count}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    # Tampilkan hasil
    cv2.imshow("FaceMesh + EAR", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
