import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from utilities.nn.conv import ShallowNet
from utilities.nn.conv import LeNet
from utilities.nn.conv import LeNet_MNIST
from utilities.nn.conv import MiniVGGNet
from utilities.nn.conv import AlexNet
from utilities.nn.conv import ResNet
from utilities.visualise import Visualise
from keras.utils import plot_model

imageSize = 64
epoch = 4000
decayEpoch = 4000
learningRate = 1e-5
folds = 5


print("[INFO]: Compiling model....")
optimizer = SGD(lr=learningRate, decay=learningRate/decayEpoch, momentum=0.9, nesterov=True)
model = LeNet_MNIST.build(width=imageSize, height=imageSize, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)
model.summary()