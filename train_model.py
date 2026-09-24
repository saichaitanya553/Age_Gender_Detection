import os
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset_path = "dataset/crop_part1"

images = []
ages = []
genders = []

for img_name in os.listdir(dataset_path):

    try:
        age = int(img_name.split("_")[0])
        gender = int(img_name.split("_")[1])

        img_path = os.path.join(dataset_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (128,128))

        images.append(img)
        ages.append(age)
        genders.append(gender)

    except:
        continue


X = np.array(images) / 255.0
age_y = np.array(ages)
gender_y = np.array(genders)

X_train, X_test, age_train, age_test = train_test_split(
    X, age_y, test_size=0.2
)

X_train_g, X_test_g, gender_train, gender_test = train_test_split(
    X, gender_y, test_size=0.2
)

# ---------------- AGE MODEL ----------------

age_model = Sequential()

age_model.add(Input(shape=(128,128,3)))

age_model.add(Conv2D(32,(3,3),activation='relu'))
age_model.add(MaxPooling2D(2,2))

age_model.add(Conv2D(64,(3,3),activation='relu'))
age_model.add(MaxPooling2D(2,2))

age_model.add(Conv2D(128,(3,3),activation='relu'))
age_model.add(MaxPooling2D(2,2))



age_model.add(Flatten())

age_model.add(Dense(128,activation='relu'))
age_model.add(Dense(1))

age_model.compile(loss="mse", optimizer="adam")

age_model.fit(X_train, age_train, epochs=40, batch_size=32)

age_model.save("age_model.h5")


# ---------------- GENDER MODEL ----------------

gender_model = Sequential()

gender_model.add(Input(shape=(128,128,3)))

gender_model.add(Conv2D(32,(3,3),activation='relu'))
gender_model.add(MaxPooling2D(2,2))

gender_model.add(Conv2D(64,(3,3),activation='relu'))
gender_model.add(MaxPooling2D(2,2))

gender_model.add(Flatten())

gender_model.add(Dense(128,activation='relu'))
gender_model.add(Dense(1, activation="sigmoid"))

gender_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

gender_model.fit(X_train_g, gender_train, epochs=30, batch_size=32)

gender_model.save("gender_model.h5")

print("Models trained and saved")