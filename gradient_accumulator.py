import tensorflow as tf

class GradientAccumulatorModel(tf.keras.Model):

    def __init__(self, num_accum, **kargs):
        super(GradientAccumulatorModel, self).__init__(**kargs)
        self.num_accum = tf.constant(num_accum, dtype=tf.int32)
        self.step_count = tf.Variable(0, dtype=tf.int32, trainable=False)

    def compile(self, **kargs):
        super(GradientAccumulatorModel, self).compile(**kargs)
        if self.num_accum == 1:
            self.accum_func = self.__apply_now
        else:
            self.accum_func = self.__apply_accum
            self.grad_accum = [
                tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
                    for v in self.trainable_variables]

    def accumulate_grads_and_apply(self, step_grads):
        self.accum_func(step_grads)

    def __apply_now(self, step_grads):
        self.optimizer.apply_gradients(zip(step_grads, self.trainable_variables))

    def __apply_accum(self, step_grads):
        self.step_count.assign_add(1)
        for i in range(len(step_grads)):
            avg_grad = step_grads[i] / tf.cast(self.num_accum, tf.float32)
            self.grad_accum[i].assign_add(avg_grad)
        tf.cond(tf.equal(self.step_count, self.num_accum),
            self.__apply_grads_and_init, lambda: None)

    def __apply_grads_and_init(self):
        self.optimizer.apply_gradients(zip(self.grad_accum, self.trainable_variables))
        self.step_count.assign(0)
        for i in range(len(self.grad_accum)):
            self.grad_accum[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
