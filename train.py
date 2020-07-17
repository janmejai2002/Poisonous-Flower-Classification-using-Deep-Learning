# set matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use('Agg')

#import necessary packages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model.mobilenetv2 import MobilenetV2_adv
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 200
INIT_LR = 0.003
BS = 32
IMAGE_DIMS = (224,224,3)

data = []
labels = []

#grab image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for  imagePath in imagePaths:
    # load image, preprocess it and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]

    labels.append(label)


data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

#binarize the labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.2, random_state = 42)

aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2,
    shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")

# initialize the model
print("[INFO] compiling the model ...")

model = MobilenetV2_adv.build(classes = len(lb.classes_))
opt = Adadelta(learning_rate = INIT_LR)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics= ['accuracy'])

# training model
print("[INFO] training the model...")

history =  model.fit(x = aug.flow(x_train, y_train, batch_size = BS),
                    validation_data = (x_test, y_test),
                    steps_per_epoch = len(x_train) // BS,
                    epochs = EPOCHS, verbose = 1)

print("[INFO] serializing network")
model.save(args["model"], save_format = "h5")

#save the label binarizer to disk

print("[INFO] serializing label binarizer")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()


plt.style.use("ggplot")
plt.figure()
N = EPOCHS

plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])