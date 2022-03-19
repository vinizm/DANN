import tensorflow as tf


class FlipGradientBuilder:
    
    def __init__(self):
        pass
    
    @tf.custom_gradient
    def grad_reverse(self, x, l = 1.):
        y = tf.identity(x)
        def custom_grad(dy):
            return -l * dy
        return y, custom_grad
    
    def __call__(self, x, l = 1.):
        return self.grad_reverse(x, l = l)
    
flip_gradient = FlipGradientBuilder()