from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout

from models.constraints import ZeroMeanFilter


class DomainDiscriminatorFullyConnected(Model):

    def __init__(self, units: int = 1024, **kwargs):
        super(DomainDiscriminatorFullyConnected, self).__init__(**kwargs)

        self.flat = Flatten()
        
        self.dense_1 = Dense(units = units)
        self.activ_1 = Activation('relu')
        self.batch_norm_1 = BatchNormalization(epsilon = 1.e-6)
        
        self.dense_2 = Dense(units = units)
        self.activ_2 = Activation('relu')
        self.batch_norm_2 = BatchNormalization(epsilon = 1.e-6)
                
        self.dense_3 = Dense(units = 2)
        self.proba = Activation('softmax')               
        
    def call(self, x):

        x = self.flat(x)
        
        x = self.dense_1(x)
        x = self.activ_1(x)
        x = self.batch_norm_1(x)

        x = self.dense_2(x)
        x = self.activ_2(x)
        x = self.batch_norm_2(x)

        x = self.dense_3(x)
        x = self.proba(x)

        return x

    def get_config(self):
        config = super(DomainDiscriminatorFullyConnected, self).get_config()
        return config

    def from_config(cls, config):
        return cls(**config)


# class DomainDiscriminatorFullyConnected(Model):

#     def __init__(self, units: int = 1024, **kwargs):
#         super(DomainDiscriminatorFullyConnected, self).__init__(**kwargs)
        
#         self.conv_1 = Conv2D(filters = 128, kernel_size = 1, strides = 1, padding = 'valid')
#         self.conv_2 = Conv2D(filters = 64, kernel_size = 1, strides = 1, padding = 'valid')
        
#         self.flat = Flatten()
        
#         self.dense_1 = Dense(units = units)
#         self.activ_1 = Activation('relu')
        
#         self.dense_2 = Dense(units = units)
#         self.activ_2 = Activation('relu')
                
#         self.dense_3 = Dense(units = 2)
#         self.proba = Activation('softmax')               
        
#     def call(self, x):
        
#         x = self.conv_1(x)
#         x = self.conv_2(x)
        
#         x = self.flat(x)
        
#         x = self.dense_1(x)
#         x = self.activ_1(x)

#         x = self.dense_2(x)
#         x = self.activ_2(x)

#         x = self.dense_3(x)
#         x = self.proba(x)

#         return x

#     def get_config(self):
#         config = super(DomainDiscriminatorFullyConnected, self).get_config()
#         return config

#     def from_config(cls, config):
#         return cls(**config)


class DomainDiscriminatorPixelwise(Model):
    
    def __init__(self, zero_mean: bool, **kwargs):
        super(DomainDiscriminatorPixelwise, self).__init__(**kwargs)
        
        self.zero_mean = zero_mean
        
        constraint_function = None
        if self.zero_mean:
            constraint_function = ZeroMeanFilter()
        
        self.conv_1 = Conv2D(filters = 512, kernel_size = 1, strides = 1, padding = 'valid', kernel_constraint = constraint_function)
        self.batch_norm_1 = BatchNormalization(epsilon = 1.e-6)
        self.activ_1 = LeakyReLU(0.2)
        
        self.conv_2 = Conv2D(filters = 512, kernel_size = 1, strides = 1, padding = 'valid', kernel_constraint = constraint_function)
        self.batch_norm_2 = BatchNormalization(epsilon = 1.e-6)
        self.activ_2 = LeakyReLU(0.2)
        
        self.conv_3 = Conv2D(filters = 512, kernel_size = 1, strides = 1, padding = 'valid', kernel_constraint = constraint_function)
        self.batch_norm_3 = BatchNormalization(epsilon = 1.e-6)
        self.activ_3 = LeakyReLU(0.2)
        
        self.conv_4 = Conv2D(filters = 512, kernel_size = 1, strides = 1, padding = 'valid', kernel_constraint = constraint_function)
        self.batch_norm_4 = BatchNormalization(epsilon = 1.e-6)
        self.activ_4 = LeakyReLU(0.2)
        
        self.conv_final = Conv2D(filters = 1, kernel_size = 1, strides = 1, padding = 'valid')
        self.activ_final = Activation('sigmoid')    

    def call(self, x):
        
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.activ_1(x)
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.activ_2(x)
        
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.activ_3(x)

        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.activ_4(x)

        x = self.conv_final(x)
        output_proba = self.activ_final(x)

        return output_proba
    
    def get_config(self):
        config = super(DomainDiscriminatorPixelwise, self).get_config()
        config.update({'zero_mean': self.zero_mean})
        return config

    def from_config(cls, config):
        return cls(**config)