from gradient_accumulator import GradientAccumulatorModel

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout


class CustomClasifier(GradientAccumulatorModel):

    def __init__(self, num_classes, num_grad_accum=1, **kargs):
        super(CustomClasifier, self).__init__(num_accum=num_grad_accum, **kargs)
        self.num_classes = num_classes
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy()
        self.val_acc_tacker = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='loss')
        
        def __conv_bn_act(filters):
            layers = []
            layers.append(Conv2D(filters, (3, 3), padding='same'))
            layers.append(BatchNormalization())
            layers.append(ReLU())
            return layers

        self.seq = tf.keras.Sequential(
            __conv_bn_act(32) + __conv_bn_act(32) +
            [MaxPool2D((2, 2), (2, 2))] +
            __conv_bn_act(64) + __conv_bn_act(64) +
            [MaxPool2D((2, 2), (2, 2))] +
            __conv_bn_act(128) + __conv_bn_act(128) +
            [MaxPool2D((2, 2), (2, 2))] +
            __conv_bn_act(256) + __conv_bn_act(256) +
            [GlobalAveragePooling2D(), Dropout(0.3)] +
            [Dense(1024), BatchNormalization(), ReLU()] + 
            [Dense(self.num_classes)])

        # self.seq = tf.keras.Sequential(
        #     __conv_bn_act(16) +
        #     [MaxPool2D((2, 2), (2, 2))] +
        #     __conv_bn_act(32) +
        #     [MaxPool2D((2, 2), (2, 2))] +
        #     [Flatten(), Dropout(0.2)] +
        #     [Dense(32), BatchNormalization(), ReLU()] + 
        #     [Dense(self.num_classes)])

    def compile(self, **kargs):
        super(CustomClasifier, self).compile(**kargs)

    def call(self, x, training=False):
        return self.seq(x, training=training)

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_true = tf.one_hot(y_true, self.num_classes)
            y_pred = self(x, training=True)
            probs = tf.nn.softmax(y_pred, axis=-1)
            total_loss = tf.keras.losses.categorical_crossentropy(y_true, probs)
            total_loss = tf.math.reduce_mean(total_loss)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.accumulate_grads_and_apply(grads)
        self.loss_tracker.update_state(total_loss)
        self.acc_tracker.update_state(y_true, probs)
        return {'loss': self.loss_tracker.result(),
            'accuracy': self.acc_tracker.result()}

    def test_step(self, data):
        x, y_true = data
        y_pred = self(x, training=False)
        val_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
        self.val_loss_tracker.update_state(val_loss)
        self.val_acc_tacker.update_state(y_true, y_pred)
        test_logs = {
            self.val_loss_tracker.name: self.val_loss_tracker.result(),
            self.val_acc_tacker.name: self.val_acc_tacker.result()}
        return test_logs

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def summary(self):
        self.seq.summary()
