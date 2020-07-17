from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Dropout, Dense

class MobilenetV2_adv:
    @staticmethod
    def build(classes):
        mobile = MobileNetV2(weights = "imagenet", include_top = False, input_shape = (224,224,3))
        for layer in mobile._layers:
            layer.trainable = False
        x = mobile.output
        x = Flatten(name = "flatten")(x)
        x = Dropout(0.5)(x)
        x = Dense(5, activation = 'softmax')(x)

        model = Model(inputs=mobile.input, outputs = x)


        return model