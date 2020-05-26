# Almutwakel Hassan

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# import dataset
data = keras.datasets.fashion_mnist

# split data into training and testing data

(train_images , train_labels),  (test_images, test_labels) = data.load_data()

classification_names = ["T-Shirt",'Pants','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']
# shrink data into smaller scale numbers
train_images = train_images/255.0
test_images = test_images/255.0

""" # training data for 1 item
print(train_images[5])
# showing a specific image from the dataset
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()"""

# build the architecture of the network
# 728 inputs (1 from each pixel), 128 hidden layer nodes, 10 outputs

model = keras.Sequential([
    # flatten data into 1 long array
    keras.layers.Flatten(input_shape=(28,28)),
    # create hidden layer in between with 128 nodes and a RELU activation function
    keras.layers.Dense(128, activation = "relu"),
    # create the output layer with softmax activation function, which shows probability
    keras.layers.Dense(10, activation="softmax")
    ])

# create loss function to optimize data
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train the model
model.fit(train_images, train_labels, epochs=5)

"""# test the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested accuracy:", test_acc)"""

# use it to evaluate

# makes an array of predictions
prediction = model(test_images)

# shows the prediction probabilities for the first entry
# print(prediction[0])

# shows the name of predicted outcome using argmax, which gives the index of the highest number in an array
# print(classification_names[np.argmax(prediction[0])])

plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(classification_names[test_labels[i]])
    plt.title(classification_names[np.argmax(prediction[i])])
    plt.show()