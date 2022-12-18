import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from models.deeplabv3plus import DeepLabV3Plus
from models.layers import GradientReversalLayer
from models.discriminators import DomainDiscriminatorPixelwise


class DomainAdaptationModel(Model):

    def __init__(self, input_shape: tuple = (256, 256, 1), num_class: int = 2, output_stride: int = 8, skip_conn: bool = True, **kwargs):
        super(DomainAdaptationModel, self).__init__(**kwargs)

        self.main_network = DeepLabV3Plus(input_shape = input_shape, num_class = num_class, output_stride = output_stride, skip_conn = skip_conn, domain_adaptation = True)
        self.gradient_reversal_layer = GradientReversalLayer()
        self.domain_discriminator = DomainDiscriminatorPixelwise(zero_mean = False)

        self.inputs = [Input(shape = input_shape), Input(shape = (1,))]
        self.outputs = self.call(self.inputs)

        self.build()

    def call(self, inputs):
        input_img, l = inputs

        segmentation_output, feature_output = self.main_network(input_img)
        discriminator_input = self.gradient_reversal_layer([feature_output, l])
        discriminator_output = self.domain_discriminator(discriminator_input)

        return segmentation_output, discriminator_output

    def get_config(self):
        config = super(DomainAdaptationModel, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

    def build(self):
        super(DomainAdaptationModel, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)