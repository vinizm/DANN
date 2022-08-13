from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D


class DomainDiscriminatorFullyConnected(Model):

    def __init__(self, units: int, **kwargs):
        super(DomainDiscriminatorFullyConnected, self).__init__(**kwargs)
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
        config = super(DomainDiscriminatorFullyConnected, self).get_config()
        config.update({'units': self.units})
        return config

    def from_config(cls, config):
        return cls(**config)


class DomainDiscriminatorLeNet5(Model):

    def __init__(self, **kwargs):
        super(DomainDiscriminatorLeNet5, self).__init__(**kwargs)
        
        self.conv_1 = Conv2D(filters = 192, kernel_size = 5, strides = 1, padding = 'same')
        self.activ_1 = Activation('relu')
        
        self.conv_2 = Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same')
        self.activ_2 = Activation('relu')
        
        self.conv_3 = Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'valid')
        self.activ_3 = Activation('relu')
        
        self.avg_pool_1 = AveragePooling2D(pool_size = 2, strides = 2)
        
        self.conv_4 = Conv2D(filters = 128, kernel_size = 5, strides = 1)
        self.activ_4 = Activation('relu')
        
        self.avg_pool_2 = AveragePooling2D(pool_size = 2, strides = 2)
        
        self.conv_5 = Conv2D(filters = 128, kernel_size = 5, strides = 1)
        self.activ_5 = Activation('relu')
        
        self.flat = Flatten()
        self.dense_1 = Dense(units = 96)
        self.activ_6 = Activation('relu')
        
        self.dense_2 = Dense(units = 32)
        self.activ_7 = Activation('relu')        
        
        self.dense_3 = Dense(units = 2)
        self.activ_8 = Activation('softmax')

    def call(self, x):
        
        x = self.conv_1(x)
        x = self.activ_1(x)
        
        x = self.conv_2(x)
        x = self.activ_2(x)
        
        x = self.conv_3(x)
        x = self.activ_3(x)
        
        x = self.avg_pool_1(x)
        
        x = self.conv_4(x)
        x = self.activ_4(x)
        
        x = self.avg_pool_2(x)
        
        x = self.conv_5(x)
        x = self.activ_5(x)
        
        x = self.flat(x)
        x = self.dense_1(x)
        x = self.activ_6(x)
        
        x = self.dense_2(x)
        x = self.activ_7(x)
        
        x = self.dense_3(x)
        x = self.activ_8(x)

        return x

    def get_config(self):
        config = super(DomainDiscriminatorLeNet5, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)
    
class DomainDiscriminatorHybridv1(Model):

    def __init__(self, **kwargs):
        super(DomainDiscriminatorHybridv1, self).__init__(**kwargs)
        
        self.batch_norm = BatchNormalization(epsilon = 1.e-32)
        
        self.conv_1 = Conv2D(filters = 256, kernel_size = 5, strides = 1, padding = 'same')
        self.activ_1 = Activation('relu')

        self.conv_2 = Conv2D(filters = 192, kernel_size = 5, strides = 1, padding = 'valid')
        self.activ_2 = Activation('relu')

        self.conv_3 = Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'valid')
        self.activ_3 = Activation('relu')
        
        self.conv_4 = Conv2D(filters = 96, kernel_size = 5, strides = 1, padding = 'valid')
        self.activ_4 = Activation('relu')

        self.conv_5 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'valid')
        self.activ_5 = Activation('relu')

        self.flat = Flatten()
        
        self.dense_1 = Dense(units = 192)
        self.activ_6 = Activation('relu')

        self.dense_2 = Dense(units = 96)
        self.activ_7 = Activation('relu')

        self.dense_3 = Dense(units = 2)
        self.activ_8 = Activation('softmax')

    def call(self, x):
        
        x = self.batch_norm(x)
        
        x = self.conv_1(x)
        x = self.activ_1(x)
        
        x = self.conv_2(x)
        x = self.activ_2(x)
        
        x = self.conv_3(x)
        x = self.activ_3(x)
        
        x = self.conv_4(x)
        x = self.activ_4(x)
        
        x = self.conv_5(x)
        x = self.activ_5(x)
        
        x = self.flat(x)
        x = self.dense_1(x)
        x = self.activ_6(x)
        
        x = self.dense_2(x)
        x = self.activ_7(x)
        
        x = self.dense_3(x)
        x = self.activ_8(x)

        return x

    def get_config(self):
        config = super(DomainDiscriminatorHybridv1, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)