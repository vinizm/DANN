from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation


class DomainDiscriminator(Model):

    def __init__(self, units: int, **kwargs):
        super(DomainDiscriminator, self).__init__(**kwargs)
        self.units = units

        self.flat = Flatten()
        self.batch_norm_1 = BatchNormalization(center = False, scale = False, axis = -1, epsilon = 1.e-32)
        self.dense_1 = Dense(units = units)
        self.activ_1 = Activation('relu')
        self.dense_2 = Dense(units = units)
        self.activ_2 = Activation('relu')
        self.dense_3 = Dense(units = 2)
        self.proba = Activation('softmax')

    def call(self, x):

        x = self.flat(x)
        x = self.batch_norm_1(x, training = True)
        x = self.dense_1(x)
        x = self.activ_1(x)
        x = self.dense_2(x)
        x = self.activ_2(x)
        x = self.dense_3(x)
        x = self.proba(x)

        return x

    def get_config(self):
        config = super(DomainDiscriminator, self).get_config()
        config.update({'units': self.units})
        return config

    def from_config(cls, config):
        return cls(**config)