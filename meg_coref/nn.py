import math
import numpy as np
import tensorflow as tf

for x in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(x, True)


class RasterSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y=None, batch_size=128):
        self.x = np.array(x)
        if y is None:
            self.y = None
        else:
            self.y = np.array(y)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        out = (batch_x,)
        if self.y is not None:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            out = out + (batch_y,)

        return out


class DNN(tf.keras.Model):
    def __init__(
            self,
            lab_map=None,
            learning_rate=0.0001,
            layer_type='rnn',
            n_layers=1,
            n_units=16,
            kernel_width=20,
            cnn_activation='gelu',
            n_outputs=300,
            dropout=None,
            reg_scale=1.,
            use_glove=False,
            continuous_outputs=False,
            project=True,
            layer_normalize=False,
            batch_normalize=False,
            **kwargs
    ):
        super(DNN, self).__init__(**kwargs)
        self.lab2ix = lab_map
        if lab_map:
            self.ix2lab = {self.lab2ix[y]: y for y in self.lab2ix}
        else:
            self.ix2lab = None
        self.learning_rate = learning_rate
        self.layer_type = layer_type.lower()
        self.n_layers = n_layers
        self.n_units = n_units
        self.kernel_width = kernel_width
        self.cnn_activation = cnn_activation
        self.n_outputs = n_outputs
        self.reg_scale = reg_scale
        self.dropout = dropout
        self.use_glove = use_glove
        self.continuous_outputs = continuous_outputs
        self.project = project
        self.layer_normalize = layer_normalize
        self.batch_normalize = batch_normalize

        if use_glove or continuous_outputs:
            loss = 'mean_squared_error'
            metrics = ['mse']
            output_activation = None
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['acc']
            if self.reg_scale:
                metrics = ['ce'] + metrics
            output_activation = 'softmax'

        if self.reg_scale:
            kernel_regularizer = tf.keras.regularizers.L2(self.reg_scale)
        else:
            kernel_regularizer = None

        layers = []
        if self.batch_normalize:
            layers.append(tf.keras.layers.BatchNormalization())
        if self.layer_normalize:
            layers.append(tf.keras.layers.LayerNormalization())
        for _ in range(self.n_layers):
            if layer_type == 'cnn':
                layers.append(
                    tf.keras.layers.Conv1D(
                        n_units,
                        kernel_width,
                        padding='causal',
                        kernel_regularizer=kernel_regularizer,
                        activation=cnn_activation
                    )
                )
                if self.batch_normalize:
                    layers.append(tf.keras.layers.BatchNormalization())
                if self.layer_normalize:
                    layers.append(tf.keras.layers.LayerNormalization())
                if dropout:
                    layers.append(tf.keras.layers.Dropout(dropout))
            elif layer_type == 'rnn':
                layers.append(
                    tf.keras.layers.GRU(
                        n_units,
                        kernel_regularizer=kernel_regularizer,
                        dropout=self.dropout if self.dropout else 0.,
                        return_sequences=True
                    )
                )
                if self.batch_normalize:
                    layers.append(tf.keras.layers.BatchNormalization())
                if self.layer_normalize:
                    layers.append(tf.keras.layers.LayerNormalization())
            else:
                raise ValueError('Unrecognized layer type: %s' % layer_type)
        if project:
            layers.append(
                # tf.keras.layers.Dense(n_outputs, kernel_regularizer=kernel_regularizer, activation=None)
                tf.keras.layers.Dense(n_outputs, activation=output_activation)
            )

        self._layers = layers

        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=self.learning_rate
        )

        self.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        return x

    def fit(self, *args, **kwargs):
        super(DNN, self).fit(*args, **kwargs)

        return self

    def classify(self, X, argmax=False, comparison_set=None, **kwargs):
        if self.use_glove:
            assert comparison_set is not None, 'Classification using GloVe requires a comparison set'

        outputs = self.predict(X, **kwargs)

        if self.use_glove:
            classes = np.array(sorted(list(comparison_set.keys())))
            glove_targ = np.stack([comparison_set[x] for x in classes], axis=1)
            glove_targ = zscore(glove_targ, axis=1)

            out = np.dot(outputs, glove_targ)
            if argmax:
                ix = np.argmax(out, axis=-1)
                out = classes[ix]
        else:
            out = np.argmax(outputs, axis=-1)

        out = np.vectorize(self.ix2lab.__getitem__)(out)

        return out
