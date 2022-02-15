import os

#Data

INPUT_IMAGES = "data"
BASE_OUTPUT = "output"
IMAGES = os.path.sep.join([BASE_OUTPUT, "images"])
LABELS = os.path.sep.join([BASE_OUTPUT, "labels"])
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "srcnn.model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

#Train

BATCH_SIZE = 128
NUM_EPOCHS = 10
SCALE = 2.0
INPUT_DIM = 33
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.0)
STRIDE = 14
