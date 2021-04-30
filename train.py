import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from custom_model import CustomClasifier

import tensorflow as tf
import tensorflow_datasets as tfds


def parse_option():
    parser = argparse.ArgumentParser('TF2-Keras Gradient Accumulator Example')
    parser.add_argument('--batch', type=int, help="batch size for single GPU")
    parser.add_argument('--grad_accum', type=int, help="gradient accumulation steps")
    args = parser.parse_args()
    return args

def get_dataset(batch_size):

    def __parse_data(data):
        img = data['image']
        label = data['label']
        return img, label

    def __preprocessing(x, y):
        x = tf.cast(x, tf.float32)
        return x / 255., y

    train_ds, test_ds = tfds.load('Cifar100', split=['train', 'test'])

    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.map(__parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(__preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(__parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.map(__preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds

def get_custom_model(batch_size, num_classes, grad_accum):
    model = CustomClasifier(num_classes, grad_accum)
    dummy_input = tf.random.uniform((batch_size, 32, 32, 3))
    model(dummy_input) # dummy call for building the model
    return model

class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, grad_accum, log_dir='./logs'):
        super(LogCallback, self).__init__()
        self.log_dir = log_dir
        self.grad_accum = grad_accum
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.count = 0
        self.global_step = 0

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.global_step += 1
        self.count += 1
        if self.count % self.grad_accum == 0:
            with self.writer.as_default():
                step = int(self.global_step / self.grad_accum)
                tf.summary.scalar('loss', logs['loss'], step=step)
                tf.summary.scalar('accuracy', logs['accuracy'], step=step)
                self.writer.flush()
            self.count = 0

if __name__ == '__main__':
    args = parse_option()
    num_classes = 100
    log_dir = 'logs/grad_accum_{}'.format(args.grad_accum)
    train_ds, test_ds = get_dataset(args.batch)
    model = get_custom_model(args.batch, num_classes, args.grad_accum)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
    model.summary()
    model.fit(train_ds, validation_data=test_ds, epochs=5 * args.grad_accum, verbose=1,
        callbacks=[LogCallback(args.grad_accum, log_dir=log_dir)],
        workers=tf.data.AUTOTUNE)
