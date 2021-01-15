from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import regularizers


def conv_norm(inputs, filters):
    activation = 'elu'
    kernel_initializer='he_normal'
    padding='same'
    #kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)
    #bias_regularizer=regularizers.l2(l2)
    #activity_regularizer=regularizers.l2(l2)

    conv = Conv2D(filters, (3, 3),
              activation=activation,
              kernel_initializer=kernel_initializer,
              padding=padding) (inputs)
              #kernel_regularizer=kernel_regularizer,
              #bias_regularizer=bias_regularizer,
              #activity_regularizer=activity_regularizer)(inputs)

    return BatchNormalization() (conv)

def conv_step(inputs, filters, dropout = 0.1):

    conv = conv_norm(inputs=inputs, filters=filters)
    conv = Dropout(dropout) (conv)
    conv = conv_norm(inputs=conv, filters=filters)

    return conv


def convolutional_model(inputs, skip_connections = 4, filters_first = 8, input_dropout = 0, hidden_dropout = 0.1, normalize_input=False):
    
    denominator = 2**skip_connections
    assert(inputs[0].shape[0]%denominator == 0)
    assert(inputs[0].shape[1]%denominator == 0)
    
    filters_last = filters_first * denominator

    assert(filters_last%(2**skip_connections) == 0)

    #filters_first = filters_last/(2**skip_connections)
    filters_list = [filters_first*2**i for i in range(skip_connections)]

    conv_list = []

    if normalize_input == True:
        last = Lambda(lambda x: x / 255) (inputs)
    else:
        last = inputs

    last = Dropout(input_dropout) (last)

    for filters in filters_list:
        last = conv_step(inputs=last, filters=filters, dropout=hidden_dropout)
        conv_list.append(last)
        last = MaxPooling2D(pool_size=(2, 2)) (last)

    last = conv_step(last, filters_last)

    n = len(filters_list)

    for i in range(n):
        filters = filters_list[n-1-i]
        upsample = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (last)
        conv = conv_list[n-1-i]
        upsample = concatenate([upsample, conv])

        last = conv_step(upsample, filters)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (last)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model