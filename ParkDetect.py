import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
import numpy as np
#####
## Das Ziel dieses Skript wird in zwei aufgaben unterteilt die erreicht werden sollen
## 1. Erkenne die Parkplätze auf dem Bild
## 2. Erkenne ob ein Parkplatz besetzt ist oder nicht

## Was ist zu tun:
## 1. Wir müssen feststellen durch hochladen eines Bildes wo die Parkplätze sind.
#### 1.1. Wir werden durch die Linien auf den Boden die Parkplätze erkennen
#### 1.2. Die Parkplätze müssen dann Markiert werden das es ein Parkplatz ist

## 2. Wir müssen erkennen ob das Auto darauf ist oder nicht
#### 2.1. Die Autos müssen erkannt werden ob sie auf dem Parkplatz sind oder nicht
#### 2.2. Die Autos müssen dann markiert werden das sie auf dem Parkplatz sind
#### 2.3. Die Autos müssen dann markiert werden das sie nicht auf dem Parkplatz sind

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("/Volumes/RICCA_SSD/Pictures/model.p", "rb"))


def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


# Bild einlesen
mask = "/Volumes/RICCA_SSD/Pictures/ParkDetect/mask for Parking Spot.png"
video_path = "/Volumes/RICCA_SSD/Pictures/ParkDetect/Cropped_parking_spot.mp4"

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

frame_nmr = 0
ret = True
step = 30
while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        print([diffs[j] for j in np.argsort(diffs)][::-1])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (1,1), (1, 1), (0, 0, 0), -1)
    cv2.putText(frame, 
            'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), 
            (10, 30),  # Position des Texts (x, y)
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5,  # Kleinere Schriftgröße
            (255, 255, 255),  # Textfarbe (Weiß)
            1)  # Dünnere Schrift
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()