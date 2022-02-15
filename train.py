from conf import sr_config as config
from features.build_features import CustomDataGen
from model.srcnn import SRCNN 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] compiling model...")
opt = Adam(lr = 0.001, decay=0.001 / config.NUM_EPOCHS)
model = SRCNN.build(width=config.INPUT_DIM, height=config.INPUT_DIM, depth=3)
model.compile(loss="mse", optimizer=opt)
data = CustomDataGen()

model.fit(data, epochs = config.NUM_EPOCHS, verbose=1)

model.save("srcnn.h5")
