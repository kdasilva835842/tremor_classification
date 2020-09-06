import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from utilities.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from utilities.preprocessing import SimplePreprocessor
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.conv import ShallowNet
from utilities.nn.conv import LeNet
from utilities.nn.conv import MiniVGGNet
from utilities.nn.conv import AlexNet
from utilities.nn.conv import ResNet
from utilities.visualise import Visualise
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.utils import class_weight
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from imutils import paths
import argparse
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
args = vars(ap.parse_args())

imageSize = 80
epoch = 4000
decayEpoch = 4000
learningRate = 1e-5
folds = 5

# fix random seed for reproducibility
seed = 11
np.random.seed(seed)

# Grab the list of images
print('[INFO]: Loading images....')
image_paths = list(paths.list_images(args['dataset']))

print("[INFO] process ID: {}".format(os.getpid()))

# Initialise preprocessors
sp = SimplePreprocessor(imageSize,imageSize)
iap = ImageToArrayPreprocessor()

# Load dataset from file and scale pixel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels) = sdl.load(image_paths,verbose=500)
labelsCopy = labels
classes = np.unique(labels)
data = data.astype("float")/255.0

lb = LabelEncoder()
class_names=['Mild','Zeros']

labels = lb.fit_transform(labels)

# define cross validation
kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
cvscores = []

acc_arr = []
prec_arr = []
rec_arr = []
f1_arr = []

for iteration_number, (train, test) in enumerate(kfold.split(data, labels)):

    print("[INFO]: Fold number: ", iteration_number)

    # Setting up class weights for unbalanced classes
    classWeight = class_weight.compute_class_weight('balanced',classes ,labelsCopy[train])
    print(classWeight)

    # Setting up paths
    figPath = os.path.sep.join([args["output"],"{}-{}.png".format(os.getpid(),iteration_number)])
    jsonPath = os.path.sep.join([args["output"],"{}-{}.json".format(os.getpid(),iteration_number)])
    modelPath = os.path.sep.join([args["output"], "model-{}-{}.h5".format(os.getpid(),iteration_number)])
    cmPath = os.path.sep.join([args["output"], "CM_{}-{}.png".format(os.getpid(),iteration_number)])

    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far

    print("[INFO]: Compiling model....")
    optimizer = SGD(lr=learningRate, decay=learningRate/decayEpoch, momentum=0.9, nesterov=True)
    model = LeNet.build(width=imageSize, height=imageSize, depth=3, classes=2)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])


    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    mc = ModelCheckpoint(modelPath, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), es, mc]

    # Train the network
    print("[INFO]: Training....")
    H = model.fit(data[train], labels[train], validation_data=(data[test], labels[test]), batch_size=8, epochs=epoch, class_weight=classWeight,callbacks=callbacks, verbose=2)

    # load the saved model
    saved_model = load_model(modelPath)

    # Test the network
    predictions = saved_model.predict(data[test], batch_size = 8)
    predictions[predictions <= 0.5] = 0.
    predictions[predictions > 0.5] = 1.
    print(classification_report(labels[test], predictions, target_names=['Mild','Zeros']))
    
    scores = saved_model.evaluate(data[test], labels[test])
    print("%s: %.2f%%" % (saved_model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    vis = Visualise(cmPath)
    vis.plot_confusion_matrix(labels[test],predictions,classes=class_names,normalize=True)

    # Acquiring all of the classification report scores
    acc = accuracy_score(labels[test], predictions)
    prec = precision_score(labels[test], predictions)
    rec = recall_score(labels[test], predictions)
    f1 = f1_score(labels[test], predictions)
    acc_arr.append(acc)
    prec_arr.append(prec)
    rec_arr.append(rec)
    f1_arr.append(f1)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print("Average Classification Report Metrics")
print("Mean Accuracy: {} (+/- {})".format(np.mean(acc_arr),np.std(acc_arr)))
print("Mean Precision: {} (+/- {})".format(np.mean(prec_arr),np.std(prec_arr)))
print("Mean Recall: {} (+/- {})".format(np.mean(rec_arr),np.std(rec_arr)))
print("Mean F1 Score: {} (+/- {})".format(np.mean(f1_arr),np.std(f1_arr)))

# # Test the network
# print('[INFO]: Evaluating the network....')
# predictions = saved_model.predict(testX, batch_size=8)
# predictions[predictions <= 0.5] = 0.
# predictions[predictions > 0.5] = 1.
# print(classification_report(testY, predictions, target_names=['ASD','TD']))

# cmPath = os.path.sep.join([args["output"], "CM_{}.png".format(os.getpid())])
# vis = Visualise(cmPath)
# vis.plot_confusion_matrix(testY,predictions,classes=class_names,normalize=True)

# print(accuracy_score(testY, predictions))
