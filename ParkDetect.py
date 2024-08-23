import cv2
import numpy as np
import matplotlib.pyplot as plt
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



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bild einlesen
# path_without_cars = "Scripts/data/Pictures/Parked cars/Cars-parked-in-parking-lot.jpg"
# img = cv2.imread(filename=path_without_cars)

# # Graustufenbild erstellen
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Bild glätten
# kernel_size = 5
# blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# # Kanten mit Canny-Detektor finden
# low_threshold = 50
# high_threshold = 150
# edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# # Hough-Transformation Parameter
# rho = 1
# theta = np.pi / 180
# threshold = 15
# min_line_length = 50
# max_line_gap = 20
# line_image = np.copy(img) * 0

# # Hough-Transformation anwenden
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                         min_line_length, max_line_gap)



# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# # Kombinieren der Linien mit dem Originalbild
# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

# Ergebnis anzeigen
# plt.subplot(1, 1, 1)
# plt.imshow(lines_edges)
# plt.show()



# # Auto-Klassifikator laden
# car_Classifier = cv2.CascadeClassifier("Scripts/models/haarcascade_car.xml")
# detected_cars = car_Classifier.detectMultiScale(img, minSize=(20, 20))
# print(detected_cars)
# # Autos auf dem kombinierten Bild markieren
# img_rgb = cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB)

# amount = len(detected_cars)
# print(f"Anzahl der Autos: {amount}")


# if amount != 0:
#     for (x, y, width, height) in detected_cars:
#         offset = 10             # die abstand der Linien in dem Quadrat
#         thickness = 5
#         cv2.line(img_rgb, (x, y), (x + width // 2 - offset, y), (0, 255, 0), thickness)
#         cv2.line(img_rgb, (x + width // 2 + offset, y), (x + width, y), (0, 255, 0), thickness)
#         cv2.line(img_rgb, (x, y + height), (x + width // 2 - offset, y + height), (0, 255, 0), thickness)
#         cv2.line(img_rgb, (x + width // 2 + offset, y + height), (x + width, y + height), (0, 255, 0), thickness)
#         cv2.line(img_rgb, (x, y), (x, y + height // 2 - offset), (0, 255, 0), thickness)
#         cv2.line(img_rgb, (x, y + height // 2 + offset), (x, y + height), (0, 255, 0), thickness)
#         cv2.line(img_rgb, (x + width, y), (x + width, y + height // 2 - offset), (0, 255, 0), thickness)
#         cv2.line(img_rgb, (x + width, y + height // 2 + offset), (x + width, y + height), (0, 255, 0), thickness)

# # Ergebnis anzeigen
# plt.subplot(1, 1, 1)
# plt.imshow(img_rgb)
# plt.show()


# Video öffnen
cap = cv2.VideoCapture('Scripts/data/Pictures/Parked cars/3858833-uhd_3840_2160_24fps.mp4')

# Überprüfen, ob das Video geöffnet wurde
if not cap.isOpened():
    print("Fehler beim Öffnen der Videodatei.")
    exit()

# Video öffnen

# Überprüfen, ob das Video geöffnet wurde
if not cap.isOpened():
    print("Fehler beim Öffnen der Videodatei.")
    exit()

while cap.isOpened():
    # Lesen eines Frames aus dem Video
    ret, frame = cap.read()
    
    # Überprüfen, ob der Frame erfolgreich gelesen wurde
    if not ret:
        print("Fehler beim Lesen des Frames oder Ende des Videos erreicht.")
        break

    # Auto-Klassifikator laden
    car_Classifier = cv2.CascadeClassifier("Scripts/models/haarcascade_car.xml")
    detected_cars = car_Classifier.detectMultiScale(frame, minSize=(20, 20))

    # Autos auf dem Originalbild markieren
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    amount = len(detected_cars)
    print(f"Anzahl der Autos: {amount}")

    if amount != 0:
        for (x, y, width, height) in detected_cars:
            offset = 10  # Abstand der Linien in dem Quadrat
            thickness = 5

            # Linien oben und unten
            cv2.line(img_rgb, (x, y), (x + width // 2 - offset, y), (0, 255, 0), thickness)
            cv2.line(img_rgb, (x + width // 2 + offset, y), (x + width, y), (0, 255, 0), thickness)
            cv2.line(img_rgb, (x, y + height), (x + width // 2 - offset, y + height), (0, 255, 0), thickness)
            cv2.line(img_rgb, (x + width // 2 + offset, y + height), (x + width, y + height), (0, 255, 0), thickness)

            # Linien links und rechts
            cv2.line(img_rgb, (x, y), (x, y + height // 2 - offset), (0, 255, 0), thickness)
            cv2.line(img_rgb, (x, y + height // 2 + offset), (x, y + height), (0, 255, 0), thickness)
            cv2.line(img_rgb, (x + width, y), (x + width, y + height // 2 - offset), (0, 255, 0), thickness)
            cv2.line(img_rgb, (x + width, y + height // 2 + offset), (x + width, y + height), (0, 255, 0), thickness)

    # Zeigen des Ergebnisses
    cv2.imshow('Car Detection', img_rgb)

    # Beenden bei Tastendruck 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Video freigeben und Fenster schließen
cap.release()
cv2.destroyAllWindows()