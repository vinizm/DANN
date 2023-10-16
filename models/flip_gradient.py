import tensorflow as tf


class FlipGradientBuilder:
    
    def __init__(self):
        pass
    
    @tf.custom_gradient
    def grad_reverse(self, x, l):
        y = tf.identity(x)
        def custom_grad(dy):
            l_shrink = tf.math.reduce_mean(l)
            return - l_shrink * dy, 0. * dy
        return y, custom_grad
    
    def __call__(self, x, l):
        return self.grad_reverse(x, l)
    
flip_gradient = FlipGradientBuilder()