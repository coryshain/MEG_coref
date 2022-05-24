import math
import pickle
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import tensorflow_addons as tfa

from .util import normalize

for x in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(x, True)


def dnn_classify(model, X, argmax=False, return_prob=False, comparison_set=None, **kwargs):
    if model.use_glove:
        assert comparison_set is not None, 'Classification using GloVe requires a comparison set'

    outputs = []
    for _X in X:
        _outputs = model.predict_on_batch(_X, **kwargs)
        if model.contrastive_loss_weight:
            _outputs, _ = _outputs
        outputs.append(_outputs)
    outputs = np.concatenate(outputs, axis=0)

    if model.use_glove:
        outputs = normalize(outputs, axis=-1)
        classes = np.array(sorted(list(comparison_set.keys())))
        glove_targ = np.stack([comparison_set[x] for x in classes], axis=1)
        glove_targ = normalize(glove_targ, axis=0)

        outputs = np.dot(outputs, glove_targ)
        if argmax:
            ix = np.argmax(outputs, axis=-1)
            pred = classes[ix]
        else:
            pred = outputs
        if return_prob:
            probs = np.max(outputs, axis=-1)
        else:
            prob = None
    else:
        if argmax:
            pred = np.argmax(outputs, axis=-1)
        else:
            pred = outputs
        pred = np.vectorize(lambda x: model.ix2lab.get(x, '<<OOV>>'))(pred)

    return pred


class ModelEval(tf.keras.callbacks.Callback):
    def __init__(
            self,
            save_freq=10,
            save_path=None,
            eval_data_generator=None,
            ylab=None,
            comparison_set=None,
            results_dict=None,
            results_path=None
    ):
        self.save_freq = save_freq
        self.save_path = save_path
        self.eval_data_generator = eval_data_generator
        self.ylab = ylab
        self.comparison_set = comparison_set
        self.results_dict = results_dict
        self.results_path = results_path

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.save_freq == 0:
            if self.save_path:
                self.model.save(self.save_path)

            if self.eval_data_generator is not None:
                ylab = self.ylab
                if self.results_dict is None:
                    results_dict = {}
                else:
                    results_dict = self.results_dict
                y_pred = self.model.classify(self.eval_data_generator, comparison_set=self.comparison_set)
                T = y_pred.shape[1]

                results_acc = np.zeros(T)
                results_f1 = np.zeros(T)
                for t in range(T):
                    acc = accuracy_score(ylab, y_pred[:, t])
                    f1 = f1_score(ylab, y_pred[:, t], average='macro')
                    results_acc[t] = acc
                    results_f1[t] = f1
                results_dict['acc'] = results_acc
                results_dict['f1'] = results_f1
                with open(self.results_path, 'wb') as f:
                    pickle.dump(results_dict, f)


class RasterData(tf.keras.utils.Sequence):
    def __init__(self, x, y=None, batch_size=128, shuffle=False, contrastive_sampling=False):
        self.x = np.array(x)
        if y is None:
            self.y = None
        else:
            self.y = np.array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.contrastive_sampling = contrastive_sampling
        if self.contrastive_sampling:
            y_uniq, y_ix_inv = np.unique(y, return_inverse=True, axis=0)
            n_y_uniq = len(y_uniq)
            self.y_uniq = y_uniq
            self.y_ix_inv = y_ix_inv
            self.n_y_uniq = n_y_uniq

        self.set_idx()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.ix[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[indices]
        out = batch_x
        if self.y is not None:
            batch_y = self.y[indices]
            tile_ix = [1, batch_x.shape[1]]
            if len(self.y.shape) == 2:
                tile_ix.append(1)
            batch_y = np.tile(np.expand_dims(batch_y, axis=1), tile_ix)
            if self.contrastive_sampling:
                y_ix_inv = self.y_ix_inv[indices]
                contrastive_ix = np.mod(y_ix_inv + np.random.randint(1, self.n_y_uniq), self.n_y_uniq)
                contrastive_targets = self.y_uniq[contrastive_ix]
                contrastive_targets = np.tile(np.expand_dims(contrastive_targets, axis=1), tile_ix)
                out = (batch_x, (batch_y, contrastive_targets))
            else:
                out = (batch_x, batch_y)

        return out

    def on_epoch_end(self):
        self.set_idx()

    def set_idx(self):
        ix = np.arange(len(self.x))
        if self.shuffle:
            ix = np.random.permutation(ix)
        self.ix = ix


class ResNetConv1DBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_width,
            n_layers=2,
            kernel_regularizer=None,
            inner_activation=None,
            activation=None,
            layer_normalize=False,
            batch_normalize=False,
            **kwargs
    ):
        super(ResNetConv1DBlock, self).__init__(**kwargs)
        self.kernel_width = kernel_width
        self.n_layers = n_layers
        self.kernel_regularizer = kernel_regularizer
        self.inner_activation = inner_activation
        self.activation = activation
        self.layer_normalize = layer_normalize
        self.batch_normalize = batch_normalize

    def build(self, input_shape):
        _x = tf.keras.Input(input_shape)
        n_units = _x.shape[-1]
        _layers = []
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                activation = self.inner_activation
            else:
                activation = self.activation
            _layers.append(
                tf.keras.layers.Conv1D(
                    n_units,
                    self.kernel_width,
                    padding='causal',
                    kernel_regularizer=self.kernel_regularizer,
                    activation=activation
                )
            )
            if self.batch_normalize:
                _layers.append(tf.keras.layers.BatchNormalization())
            if self.layer_normalize:
                _layers.append(tf.keras.layers.LayerNormalization())

        self._layers = _layers

        self._add = tf.keras.layers.Add()

        self.built = True

    def call(self, inputs, training=False):
        _x = inputs
        x = inputs
        for layer in self._layers:
            x = layer(x, training=training)

        x = self._add([_x, x])

        return x


class L2LayerNormalization(tf.keras.layers.Layer):
    def __init__(
            self,
            epsilon=0.001,
            center=True,
            scale=True,
            **kwargs
    ):
        super(L2LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        ndim = int(input_shape[-1])
        self.gamma = self.add_weight(
            name='gamma',
            shape=(ndim,),
            initializer='ones'
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(ndim,),
            initializer='zeros'
        )

        self.built = True

    def call(self, inputs, training=False):
        norm = tf.maximum(tf.linalg.norm(inputs, axis=-1, keepdims=True), self.epsilon)
        gamma = self.gamma
        while len(gamma.shape) < len(inputs.shape):
            gamma = gamma[None, ...]
        beta = self.beta
        while len(beta.shape) < len(inputs.shape):
            beta = beta[None, ...]
        return inputs / norm * gamma + beta


class SensorFilter(tf.keras.layers.Layer):
    def __init__(
            self,
            rate=None,
            **kwargs
    ):
        super(SensorFilter, self).__init__(**kwargs)

        self.rate = rate
        if self.rate:
            self.w_regularizer = tf.keras.regularizers.L1(self.rate)
        else:
            self.w_regularizer = None

    def build(self, input_shape):
        ndim = int(input_shape[-1])
        self.w = self.add_weight(
            name='filter_weights',
            shape=(ndim,),
            initializer='ones',
            regularizer=self.w_regularizer
        )

        self.built = True

    def call(self, inputs, training=False):
        x = inputs
        w = self.w
        w = tf.tanh(w)
        while len(w.shape) < len(x.shape):
            w = w[None, ...]
        x = x * w

        return x

@tf.keras.utils.register_keras_serializable()
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
            input_dropout=0.5,
            reg_scale=1.,
            sensor_filter_scale=None,
            use_glove=False,
            continuous_outputs=False,
            use_resnet=False,
            use_locally_connected=False,
            contrastive_loss_weight=False,
            project=True,
            batch_normalize=False,
            layer_normalize=False,
            l2_layer_normalize=False,
            **kwargs
    ):
        super(DNN, self).__init__(**kwargs)
        self.lab_map = lab_map
        self.lab2ix = self.lab_map
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
        self.sensor_filter_scale = sensor_filter_scale
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_glove = use_glove
        self.continuous_outputs = continuous_outputs
        self.use_resnet = use_resnet
        self.use_locally_connected = use_locally_connected
        self.contrastive_loss_weight = contrastive_loss_weight
        self.project = project
        self.batch_normalize = batch_normalize
        self.layer_normalize = layer_normalize
        self.l2_layer_normalize = l2_layer_normalize

        if use_glove:
            loss = 'mse'
            metrics = []
            if self.reg_scale:
                metrics.append('mse')
            metrics.append(tf.keras.metrics.CosineSimilarity(name='sim'))
            # loss = tf.keras.losses.CosineSimilarity(name='sim', axis=-1)
            # metrics = []
            # if self.reg_scale:
            #     metrics.append(tf.keras.metrics.CosineSimilarity(name='sim', axis=-1))
            output_activation = None
        elif continuous_outputs:
            loss = 'mse'
            metrics = []
            if self.reg_scale:
                metrics.append('mse')
            output_activation = None
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = []
            if self.reg_scale:
                metrics.append('ce')
            metrics.append('acc')
            output_activation = 'softmax'

        if self.reg_scale:
            kernel_regularizer = tf.keras.regularizers.L2(self.reg_scale)
        else:
            kernel_regularizer = None

        layers = []
        if self.sensor_filter_scale:
            layers.append(SensorFilter(rate=self.sensor_filter_scale))
        if self.input_dropout:
            layers.append(tf.keras.layers.Dropout(self.input_dropout))
        if self.use_locally_connected:
            layers.append(tf.keras.layers.ZeroPadding1D(padding=(kernel_width - 1, 0)))
        for _ in range(self.n_layers):
            if layer_type == 'cnn':
                if self.use_resnet:
                    layers.append(
                        ResNetConv1DBlock(
                            kernel_width,
                            kernel_regularizer=kernel_regularizer,
                            inner_activation=self.cnn_activation,
                            activation=None,
                            layer_normalize=self.layer_normalize,
                            batch_normalize=self.batch_normalize
                        )
                    )
                else:
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
                    if self.l2_layer_normalize:
                        layers.append(L2LayerNormalization())
                if dropout:
                    layers.append(tf.keras.layers.Dropout(dropout))
            elif layer_type == 'rnn':
                layers.append(
                    tf.keras.layers.GRU(
                        n_units,
                        kernel_regularizer=kernel_regularizer,
                        return_sequences=True
                    )
                )
                if self.batch_normalize:
                    layers.append(tf.keras.layers.BatchNormalization())
                if self.layer_normalize:
                    layers.append(tf.keras.layers.LayerNormalization())
                if self.l2_layer_normalize:
                    layers.append(L2LayerNormalization())
                if dropout:
                    layers.append(tf.keras.layers.Dropout(dropout))
            else:
                raise ValueError('Unrecognized layer type: %s' % layer_type)
        if self.use_locally_connected:
            if self.project:
                lc_units = n_units
                lc_activation = cnn_activation
                lc_regularizer = kernel_regularizer
            else:
                lc_units = n_outputs
                lc_activation = output_activation
                lc_regularizer = None
            layers.append(
                tf.keras.layers.LocallyConnected1D(
                    lc_units,
                    kernel_width,
                    padding='valid',
                    kernel_regularizer=lc_regularizer,
                    activation=lc_activation,
                    implementation=1
                )
            )
            if self.project:
                if self.batch_normalize:
                    layers.append(tf.keras.layers.BatchNormalization())
                if self.layer_normalize:
                    layers.append(tf.keras.layers.LayerNormalization())
                if self.l2_layer_normalize:
                    layers.append(L2LayerNormalization())
                if dropout:
                    layers.append(tf.keras.layers.Dropout(dropout))
        if project:
            layers.append(
                tf.keras.layers.Dense(n_outputs, kernel_regularizer=kernel_regularizer, activation=output_activation)
                # tf.keras.layers.Dense(n_outputs, activation=output_activation)
            )

        self._layers = layers

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        if self.contrastive_loss_weight:
            loss = [loss, tfa.losses.ContrastiveLoss()]
            metrics = [metrics, []]
            loss_weights = [1, self.contrastive_loss_weight]
        else:
            loss_weights = None

        self.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights
        )

        # self.classify = lambda *args, **kwargs: dnn_classify(self, *args, **kwargs)

    def call(self, inputs, training=False):
        x = inputs

        for layer in self._layers:
            x = layer(x, training=training)

        if self.contrastive_loss_weight:
            out = (x, x)
        else:
            out = x

        return out

    def fit(self, *args, **kwargs):
        super(DNN, self).fit(*args, **kwargs)

        return self

    def classify(self, X, argmax=True, return_prob=False, comparison_set=None, **kwargs):
        if self.use_glove:
            assert comparison_set is not None, 'Classification using GloVe requires a comparison set'

        outputs = []
        for _X in X:
            _outputs = self.predict_on_batch(_X, **kwargs)
            if self.contrastive_loss_weight:
                _outputs, _ = _outputs
            outputs.append(_outputs)
        outputs = np.concatenate(outputs, axis=0)

        if self.use_glove:
            outputs = normalize(outputs, axis=-1)
            classes = np.array(sorted(list(comparison_set.keys())))
            glove_targ = np.stack([comparison_set[x] for x in classes], axis=1)
            glove_targ = normalize(glove_targ, axis=0)

            outputs = np.dot(outputs, glove_targ)
            if argmax:
                ix = np.argmax(outputs, axis=-1)
                pred = classes[ix]
            else:
                pred = outputs
            if return_prob:
                probs = np.max(outputs, axis=-1)
            else:
                prob = None
        else:
            if argmax:
                pred = np.argmax(outputs, axis=-1)
            else:
                pred = outputs
            pred = np.vectorize(lambda x: self.ix2lab.get(x, '<<OOV>>'))(pred)

        return pred

    def get_config(self):
        config = {
            'lab_map': self.lab_map,
            'learning_rate': self.learning_rate,
            'layer_type': self.layer_type,
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'kernel_width': self.kernel_width,
            'cnn_activation': self.cnn_activation,
            'n_outputs': self.n_outputs,
            'dropout': self.dropout,
            'input_dropout': self.input_dropout,
            'reg_scale': self.reg_scale,
            'sensor_filter_scale': self.sensor_filter_scale,
            'use_glove': self.use_glove,
            'continuous_outputs': self.continuous_outputs,
            'use_resnet': self.use_resnet,
            'use_locally_connected': self.use_locally_connected,
            'project': self.project,
            'contrastive_loss_weight': self.contrastive_loss_weight,
            'batch_normalize': self.batch_normalize,
            'layer_normalize': self.layer_normalize,
            'l2_layer_normalize': self.l2_layer_normalize
        }

        return config

    @classmethod
    def from_config(cls, config):
        out = cls(**config)
        return out