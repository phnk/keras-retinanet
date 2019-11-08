import keras
import sys
import glob
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



# Running directly from the repository:
# keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5
if __name__ == "__main__":
    BACKBONE_NAME = "resnet50"
    THRESHOLD = 0.45
    TEST_VIDEO = glob.glob("/home/carbor/data/raw test videos/*.*")
    #TEST_VIDEO = ["/home/carbor/code/keras-retinanet/data/test/indoor_test_1.MP4", "/home/carbor/code/keras-retinanet/data/test/indoor_test_2.MP4", "/home/carbor/code/keras-retinanet/data/test/volvo_test.avi"]
    #TEST_VIDEO = ["/home/carbor/code/keras-retinanet/data/test/indoor_test_1.MP4"]


    gpu = 0
    setup_gpu(gpu)
    model_path = os.path.join(".", "resnet50_csv_inference.h5")

    model = models.load_model(model_path, backbone_name=BACKBONE_NAME)

    labels_to_names = {0: "wheel", 1: "cab", 2: "tipping body"}

    makedirs("output/")

    # load video
    for video in TEST_VIDEO:
        output_path = video.replace("data/raw test videos/", "code/keras-retinanet/output/").replace("avi", "mp4")

        codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        out_size = (1920, 1080)

        cap = cv2.VideoCapture(video)

        imgs = []

        if cap.isOpened() == False:
            print("Error cant open file {}".format(video))


        tt = 0

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                draw = frame.copy()
                
                image = preprocess_image(frame)
                image, scale = resize_image(image, min_side=800, max_side=800)

                start = time.time()
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                print("processing time: {}".format(time.time() - start))
                tt += time.time() - start
                boxes /= scale

                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    if score < THRESHOLD:
                        break
                    
                    color = label_color(label)
                    b = box.astype(int)
                    draw_box(draw, b, color=color)
                    caption = "{} {:.3f}".format(labels_to_names[label], score)
                    draw_caption(draw, b, caption)

                imgs.append(draw)

            else:
                print("Done with video {}".format(video))
                break

        out = cv2.VideoWriter(output_path, codec, 30, out_size)

        print("amount of frames {}".format(len(imgs)))

        print("mean inference time {}".format(tt / len(imgs)))
        
        exit(1)
        for im in imgs:
            res = cv2.resize(im, out_size)
            out.write(res)

        out.release()   
        cap.release()
