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

#MODEL = pickle.load(open("model.p", "rb"))


# def empty_or_not(spot_bgr):

#     flat_data = []

#     img_resized = resize(spot_bgr, (15, 15, 3))
#     flat_data.append(img_resized.flatten())
#     flat_data = np.array(flat_data)

#     y_output = MODEL.predict(flat_data)

#     if y_output == 0:
#         return EMPTY
#     else:
#         return NOT_EMPTY


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



# Bild einlesen
mask = "/Volumes/RICCA_SSD/Pictures/mask for Parking Spot.png"
video_path = "/Volumes/RICCA_SSD/Pictures/Cropped_parking_spot.mp4"

mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)


spots = get_parking_spots_bboxes(connected_components)
print(spots[0])


ret = True
while ret:
    ret, frame = cap.read()

    for spot in spots:
        x1, y1, w, h = spot
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0))
                      
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()