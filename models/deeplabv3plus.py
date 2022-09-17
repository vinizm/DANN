from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation

from models.layers import ReshapeTensor, ExpandDimensions
from models.blocks import SepConv_BN, _conv2d_same, _xception_block


def DeepLabV3Plus(input_shape: tuple = (256, 256, 1), num_class: int = 2, output_stride: int = 8, activation: str = 'softmax',
                  backbone_size: int = 16, custom_atrous_rates: tuple = None, domain_adaptation: bool = False):
    """ Instantiates the Deeplabv3+ architecture
    """
    img_input = Input(shape = input_shape)

    if output_stride == 8:
        entry_block3_stride = 1
        middle_block_rate = 2 # not mentioned in paper; but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36) if custom_atrous_rates is None else custom_atrous_rates
        print(atrous_rates)

    elif output_stride == 16:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18) if custom_atrous_rates is None else custom_atrous_rates
        print(atrous_rates)

    x = Conv2D(32, (3, 3), strides = (2, 2), name = 'entry_flow_conv1_1', use_bias = False, padding = 'same')(img_input)
    x = BatchNormalization(name = 'entry_flow_conv1_1_BN', epsilon = 1.e-32)(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size = 3, stride = 1)
    x = BatchNormalization(name = 'entry_flow_conv1_2_BN', epsilon = 1.e-32)(x)
    x = Activation('relu')(x)

    x, skip_0 = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                               skip_connection_type = 'conv', stride = 2,
                               depth_activation = False, return_skip = True)
    
    # new skip connection
    x = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type = 'conv', stride = 2,
                               depth_activation = False, return_skip = False)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type = 'conv', stride = entry_block3_stride,
                        depth_activation = False)
    
    for i in range(backbone_size):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type = 'sum', stride = 1, rate = middle_block_rate,
                            depth_activation = False)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type = 'conv', stride = 1, rate = exit_block_rates[0],
                        depth_activation = False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type = 'none', stride = 1, rate = exit_block_rates[1],
                        depth_activation = True)

    # end of feature extractor
    # branching for Atrous Spatial Pyramid Pooling (ASPP)
    # image feature branch
    b4 = GlobalAveragePooling2D()(x)

    # from (batch_size, channels) -> (batch_size, 1, 1, channels)
    b4 = ExpandDimensions(axis = 1)(b4)
    b4 = ExpandDimensions(axis = 1)(b4)

    b4 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'image_pooling')(b4)
    b4 = BatchNormalization(name = 'image_pooling_BN', epsilon = 1.e-32)(b4)
    b4 = Activation('relu')(b4)

    # upsample; have to use compat because of the option align_corners
    size_before = tuple(x.shape)
    b4 = ReshapeTensor(size_before[1:3], factor = 1, method = 'bilinear', align_corners = True)(b4)
    
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'aspp0')(x)
    b0 = BatchNormalization(name = 'aspp0_BN', epsilon = 1.e-32)(b0)
    b0 = Activation('relu', name = 'aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2; not sure why
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1', rate = atrous_rates[0], depth_activation = True, epsilon = 1.e-32)
    
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2', rate = atrous_rates[1], depth_activation = True, epsilon = 1.e-32)
    
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3', rate = atrous_rates[2], depth_activation = True, epsilon = 1.e-32)

    # concatenate ASPP branches and project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding = 'same', use_bias = False, name = 'concat_projection')(x)
    x = BatchNormalization(name = 'concat_projection_BN', epsilon = 1.e-32)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # DeepLabv3+ decoder
    # feature projection
    size_before_2 = tuple(x.shape)
    x = ReshapeTensor(size_before_2[1:3], factor = output_stride // 4, method = 'bilinear', align_corners = True)(x)

    x = SepConv_BN(x, 256, 'decoder_conv2', depth_activation = True, epsilon = 1.e-32)
    x = SepConv_BN(x, 256, 'decoder_conv3', depth_activation = True, epsilon = 1.e-32)

    x = ReshapeTensor(size_before_2[1:3], factor = output_stride // 2, method = 'bilinear', align_corners = True)(x)

    dec_skip_0 = Conv2D(48, (1, 1), padding = 'same', use_bias = False, name = 'feature_projection0')(skip_0)
    dec_skip_0 = BatchNormalization(name = 'feature_projection0_BN', epsilon = 1.e-32)(dec_skip_0)
    dec_skip_0 = Activation('relu')(dec_skip_0)
    discriminator_spot = Concatenate()([x, dec_skip_0])

    x = SepConv_BN(discriminator_spot, 128, 'decoder_conv0', depth_activation = True, epsilon = 1.e-32)
    x = SepConv_BN(x, 128, 'decoder_conv1', depth_activation = True, epsilon = 1.e-32)
    x = Conv2D(num_class, (1, 1), padding = 'same', name = 'custom_logits_semantic')(x)

    size_before_3 = tuple(img_input.shape)
    x = ReshapeTensor(size_before_3[1:3], factor = 1, method = 'bilinear', align_corners = True)(x)

    if activation in ['softmax', 'sigmoid']:
        outputs = Activation(activation)(x)

    if domain_adaptation:
        outputs = [outputs, discriminator_spot]

    inputs = img_input
    model = Model(inputs = inputs, outputs = outputs, name = 'deeplabv3plus')
    return model