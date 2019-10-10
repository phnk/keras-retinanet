import keras
import keras_retinanet.models


if __name__ == "__main__":
    NUM_CLASSES = 3
    BACKBONE = "resnet50"
    LEARNING_RATE = 1e-5
    CLIP_NORM = 0.001 # What is this?

    model = keras_retinanet.models.backbone(BACKBONE).retinanet(NUM_CLASSES)
    
    print(model.summary())

    model.compile(
            loss={
                "regression": keras_retinanet.losses.smooth_l1(),
                "classification": keras_retinanet.losses.focal()
                },
            optimizer=keras.optimizers.adam(lr=LEARNING_RATE, clipnorm=CLIP_NORM)
            )
