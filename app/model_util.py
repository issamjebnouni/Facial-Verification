import tensorflow as tf
from keras.layers import Layer

class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def load_model():

    siamese_model = tf.keras.models.load_model("siamesemodel.h5",
                                            custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy()}) 
    
    return siamese_model
