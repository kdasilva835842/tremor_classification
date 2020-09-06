import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from utilities.preprocessing import SimplePreprocessor
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.datasets import SimpleDatasetLoader
# from utilities.nn.conv import ShallowNet
from utilities.nn.conv import LeNet
# from utilities.nn.conv import MiniVGGNet
# from utilities.nn.conv import AlexNet
# from utilities.nn.conv import ResNet
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
from keras.models import load_model
from sklearn.utils import class_weight
from imutils import paths
import argparse
import os
import numpy as np
import glob


# ap = argparse.ArgumentParser()
# ap.add_argument('-d', '--dataset', required=True,
#                 help='Path to input dataset')
# args = vars(ap.parse_args())

imageSize = 80
epoch = 4000
decayEpoch = 4000
learningRate = 1e-5
folds = 5
PID = 6678
datapath = "/Users/kelvi/Desktop/DATA/severityDivided/Train_Test/70_30_split/CRST_ZerosVsMild/val"
modelFolder = "/Users/kelvi/Downloads/trans_10"


# fix random seed for reproducibility
seed = 11
np.random.seed(seed)

# Grab the list of images
print('[INFO]: Loading images....')
#image_paths = list(paths.list_images(datapath))
image_paths = glob.glob(datapath+"/*/*.jpg")

print("[INFO] process ID: {}".format(PID))

# Initialise preprocessors
sp = SimplePreprocessor(imageSize,imageSize)
iap = ImageToArrayPreprocessor()

# Load dataset from file and scale pixel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels) = sdl.load(image_paths,verbose=500)
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

#for iteration_number, (train, test) in enumerate(kfold.split(data, labels)):
for iteration_number in range(5):

    print("[INFO]: Fold number: ", iteration_number)

    # Setting up paths
    modelPath = os.path.sep.join([modelFolder, "model-{}-{}.h5".format(PID,iteration_number)])

    # load the saved model
    saved_model = load_model(modelPath)

    # Test the network
    predictions = saved_model.predict(data, batch_size = 8)
    predictions[predictions <= 0.5] = 0.
    predictions[predictions > 0.5] = 1.
    print(classification_report(labels, predictions, target_names=['Mild','Zeros']))
    
    scores = saved_model.evaluate(data, labels)
    print("%s: %.2f%%" % (saved_model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
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
