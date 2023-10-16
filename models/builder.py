import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from models.deeplabv3plus import DeepLabV3Plus
from models.layers import GradientReversalLayer
from models.discriminators import DomainDiscriminatorFullyConnected


class DomainAdaptationModel(Model):

    def __init__(self, input_shape: tuple = (256, 256, 1), num_class: int = 2, output_stride: int = 8, units: int = 1024, skip_conn: bool = True, **kwargs):
        super(DomainAdaptationModel, self).__init__(**kwargs)

        self.main_network = DeepLabV3Plus(input_shape = input_shape, num_class = num_class, output_stride = output_stride, skip_conn = skip_conn)
        self.domain_discriminator = DomainDiscriminatorFullyConnected(units = units)

        self.inputs = [Input(shape = input_shape), Input(shape = (1,))]
        self.outputs = self.call(self.inputs)

        self.build()

    def call(self, inputs):

        bottleneck_features, skip_connect = self.main_network.encoder(inputs)
        segmentation_output = self.main_network.decoder([bottleneck_features, skip_connect])
        discriminator_output = self.domain_discriminator(bottleneck_features)

        return segmentation_output, discriminator_output

    def get_config(self):
        config = super(DomainAdaptationModel, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)

    def build(self):
        super(DomainAdaptationModel, self).build(self.inputs.shape if tf.is_tensor(self.inputs) else self.inputs)
        self.call(self.inputs)
