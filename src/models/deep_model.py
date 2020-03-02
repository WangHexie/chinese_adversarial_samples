from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
import tensorflow as tf


class SimpleCnn(Model):
    def __init__(self, maxlen, max_features, embedding_dims, embedding_matrix, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        model = tf.keras.Sequential()
        self.input_embedding_layer = tf.keras.layers.Embedding(max_features,
                                                               embedding_dims,
                                                               weights = [embedding_matrix],
                                                               embeddings_initializer='uniform',
                                                               embeddings_regularizer=None,
                                                               activity_regularizer=None, embeddings_constraint=None,
                                                               mask_zero=False, input_length=maxlen)
        model.add(self.input_embedding_layer)
        # self.input_embedding_layer.set_weights([self.word_vector.vector.vectors])
        self.input_embedding_layer.trainable = False
        model.add(tf.keras.layers.Conv1D(16, 4, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=4, strides=None, padding='valid', data_format='channels_last'))
        model.add(tf.keras.layers.Conv1D(16, 4, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=4, strides=None, padding='valid', data_format='channels_last'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))
        model.summary()
        model.compile(optimizer='adam',
                      loss="mean_absolute_error",
                      metrics=['accuracy'])
        self.model = model

    def export_middle_layers(self):
        decoder_input = tf.keras.layers.Input(shape=(self.maxlen, self.embedding_dims))
        next_input = decoder_input

        # get the decoder layers and apply them consecutively
        for layer in self.model.layers[1:]:
            next_input = layer(next_input)

        decoder = tf.keras.Model(inputs=decoder_input, outputs=next_input)
        return self.input_embedding_layer, decoder

    def call(self, inputs):
        return self.model(inputs)


class SimpleRNN(SimpleCnn):
    def __init__(self, maxlen, max_features, embedding_dims, embedding_matrix, *args, **kwargs):
        super().__init__(maxlen, max_features, embedding_dims, embedding_matrix, *args, **kwargs)
