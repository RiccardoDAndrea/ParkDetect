import os 
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

input_dir = "/Volumes/RICCA_SSD/Pictures/ParkDetect/clf-data"
categories = ["empty", "not_empty"]

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        if file.startswith('._') or not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)


data = np.asarray(data)
labels = np.asarray(labels)
#print(data.shape, labels.shape)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

classifier = SVC()

parameters = [{"gamma": [0.01, 0.1, 1], "C": [1, 10, 100]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))


pickle.dump(best_estimator, open('./model.p', 'wb'))