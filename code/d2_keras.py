"""
 Created by IntelliJ IDEA.
 Project: Keras-CNN-RGB-Images
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-02
 Time: 오후 10:52
"""

import cv2
import glob
import pandas as pd
import os
import numpy as np

import tensorflow.keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split


"""
Read Dataset
"""
abst_filename_list = [img for img in glob.glob("../dataset/*.jpg")]

filename_list = [name.split("./dataset\\")[1] for name in abst_filename_list]
filename_list = [name.split(".jpg")[0] for name in filename_list]

model_id_list = list()
product_id_list = list()
image_id_list = list()

for info in [filename.split("_") for filename in filename_list]:
    model_id_list.append(info[0])
    product_id_list.append(info[1])
    image_id_list.append(info[2])

dataset_list = list(zip(abst_filename_list, filename_list, model_id_list, product_id_list, image_id_list))

df = pd.DataFrame(dataset_list, columns=["absolute_path", "file_name", "model_id", "product_id_list", "image_id_list"])

del abst_filename_list, model_id_list, product_id_list, image_id_list, filename_list, dataset_list

###

converted_id_dict = dict()
converted_id_list = list()

labels = [int(id) for id in df["model_id"]]
labels_set = list(set(labels))

for i, id_ in enumerate(labels_set):
    converted_id_dict[id_] = i

for label in labels:
    converted_id_list.append(converted_id_dict.get(label))

df["label"] = converted_id_list


"""
Read Features
"""
dataset = list()
label_dataset = list()

img_size = 50

for path in df["absolute_path"]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    dataset.append(np.array(img))

reshaped_dataset = np.array([np.reshape(data, (img_size, img_size, 3)) for data in dataset])
label_dataset = np.array([[label] for label in df["label"]])

print("Dataset Shape : {}".format(reshaped_dataset.shape))
print("Label Shape : {}".format(label_dataset.shape))

num_classes = 153

label_dataset_one_hot = k.utils.to_categorical(label_dataset, num_classes)

reshaped_dataset = reshaped_dataset.astype("float32")
reshaped_dataset /= 255.

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(
    reshaped_dataset,
    label_dataset_one_hot,
    test_size=0.3,
    random_state=1
)


"""
Making Model
"""
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

# Hyper Parameters
epochs = 100
batch_size = 128
lr = 0.0001
decay = 1e-6

optimizer = k.optimizers.RMSprop(lr=lr, decay=decay)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs, validation_data=(X_test, y_test),
    shuffle=True,
    verbose=2
)

print("Accuracy : %.2f" % history.history["acc"][-1])
print("Valid Accuracy : %.2f" % history.history["val_acc"][-1])

# Saving Model and Weights
model_json = model.to_json()
with open("./d2_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("./d2_weights.h5")


"""
Predict Label
"""
y_pred = model.predict(reshaped_dataset)
y_pred_label = np.argmax(y_pred, axis=1)


"""
Saving to .txt and .csv
"""
# Writing to .txt
f = open("./labels_pred.txt", "w")
for label in y_pred_label:
    label_input = str(label)
    f.write(label_input + "\n")
f.close()

# Writing to .csv (with clustering)
csv_dict = {"file_name": df["file_name"], "labels_pred": y_pred_label}
df_csv = pd.DataFrame(data=csv_dict)
df_csv.to_csv("./pred_db.csv")
