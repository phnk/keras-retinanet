# https://github.com/phnk/keras-retinanet/blob/master/keras_retinanet/utils/eval.py
# https://github.com/phnk/keras-retinanet/blob/master/keras_retinanet/preprocessing/csv_generator.py

from keras-retinanet.utils.eval import evaluate
from keras-retinanet.preprocessing.csv_generator import CSVGenerator

# check inference.py for how to do everything with models
from keras_retinanet import models

'''
def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
)
'''


    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

if __name__ == "__main__":
    # create a CSVGenerator with the correct arguments (https://github.com/fizyr/keras-retinanet#csv-datasets)
    # create a model to evaluate + load the model
    # run evaluate
    # save results

    pass