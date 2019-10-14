import keras
import sys

# import modules
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.bin.train import makedirs

# import other stuff
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf


if __name__ == "__main__":
    BACKBONE_NAME = "resnet50"
    THRESHOLD = 0.45
    TEST_VIDEO = ["data/test/volvo_vid.avi"]

    gpu = 0
    setup_gpu(gpu)
    model_path = os.path.join(".", "snapshots", "whatever_the_weight_file_is.h5")

    model = models.load_model(model_path, backbone_name=BACKBONE_NAME)

    labels_to_names = {0: "wheel", 1: "cab", 2: "tipping body"}

    makedirs("output/")

    # load video
    for video in TEST_VIDEO:
        output_path = video.replace("data/test/", "output/")

        cap = cv2.VideoCapture(video)
        out = cv2.VideoWriter(output_path, -1, 20.0, (1920, 1080))

        if cap.isOpened() == False:
            print("Error cant open file {}".format(video))

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                #frame should be bgr format because its opencv
                draw = frame.copy()
                draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RBG)
                
                image = preprocess_image(frame)
                image, scale = resize_image(image)

                start = time.time()
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                print("processing time: {}".format(time.time() - start))

                boxes /= scale

                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    if score < THRESHOLD:
                        break
                    
                    color = label_color(label)
                    b = box.astype(int)
                    draw_box(draw, b, color=color)
                    caption = "{} {:.3f}".format(labels_to_names[label], score)
                    draw_caption(draw, b, caption)
                    # save img in output video
                    out.write(draw)
            else:
                print("Done with video {}".format(video))
            
        out.release()
            
        cap.release()
        cv2.destroyAllWindows()