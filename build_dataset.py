from conf import sr_config as config
from imutils import paths
from PIL import Image
import numpy as np
import shutil
import random
import PIL
import cv2
import os

# if the output directories do not exist, create them
for p in [config.IMAGES, config.LABELS]:
	if not os.path.exists(p):
		os.makedirs(p)

# grab the image paths and initialize the total number of crops
# processed
print("[INFO] creating temporary images...")
image_paths = list(paths.list_images(config.INPUT_IMAGES))
random.shuffle(image_paths)
total = 0

for image_path in image_paths:

    image = cv2.imread(image_path)
    (h,w) = image.shape[:2]
    w -= int(w % config.SCALE)
    h -= int(h % config.SCALE)
    image = image[0:h, 0:w]

    lowW = int(w * (1.0 / config.SCALE))
    lowH = int(h * (1.0 / config.SCALE))
    highW = int(lowW * (config.SCALE / 1.0))
    highH = int(lowH * (config.SCALE / 1.0))

    scaled = np.array(Image.fromarray(image).resize((lowW , lowH), resample=PIL.Image.BICUBIC))
    scaled = np.array(Image.fromarray(scaled).resize((highW, highH), resample=PIL.Image.BICUBIC))

    for y in range(0, h - config.INPUT_DIM + 1, config.STRIDE):
        for x in range(0, w - config.INPUT_DIM + 1, config.STRIDE):

            crop = scaled[y:y + config.INPUT_DIM , x:x + config.INPUT_DIM]

            target = image[y + config.PAD:y + config.PAD + config.LABEL_SIZE,
                       x + config.PAD:x + config.PAD + config.LABEL_SIZE]
        
            crop_path = os.path.sep.join([config.IMAGES, "{}.png".format(total)])
            target_path = os.path.sep.join([config.LABELS, "{}.png".format(total)])

            cv2.imwrite(crop_path, crop)
            cv2.imwrite(target_path, target)

            total+=1

'''
print("[INFO] building HDF5 datasets...")
input_paths = sorted(list(paths.list_images(config.IMAGES)))
output_paths = sorted(list(paths.list_images(condig.lABELS)))

input_writer = HDF5DatasetWriter((len(input_paths), config.INPUT_DIM, config.INPUT_DIM, 3), config.INPUTS_DB)
output_writer = HDF5DatasetWriter((len(output_paths), config.LABEL_SIZE, config.LABEL_SIZE, 3), config.OUTPUTS_DB)

for (input_path, output_path) in zip(input_paths, output_paths):

    input_image = cv2.imread(input_path)
    output_image = cv2.imread(output_path)
    input_writer.add([input_image], [-1])
    output_writer.add([output_image], [-1])

input_writer.close()
output_writer.close()

print("[INFO] cleaning up...")
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)
'''
























