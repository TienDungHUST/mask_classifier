import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def mobilenetv1(IMG_HEIGHT=128,IMG_WIDTH=128):
    base_model  = keras.applications.MobileNet(
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3),
    include_top = False,
    weights     = 'imagenet'
    )
    base_model.trainable = False
    inputs = keras.Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training = False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = keras.layers.Dense(2)(x)
    outputs = layers.Activation('softmax')(x)
    model   = keras.Model(inputs, outputs)
    model.compile(  optimizer = 'adam',
                    loss    = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics =['accuracy'])
    return model
def base(IMG_HEIGHT=128,IMG_WIDTH=128):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters= 16,kernel_size= (3,3), strides= 2, padding= 'valid',
                            activation=  'relu', input_shape= (IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Conv2D(filters= 32,kernel_size= (3,3), strides= 2, padding= 'valid',
                            activation=  'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Conv2D(filters= 64,kernel_size= (3,3), strides= 2, padding= 'valid',
                            activation=  'relu'))      
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))

    # sgd= tf.keras.optimizers.SGD(learning_rate=0.1, name='SGD2')

    model.compile(optimizer= 'adam', 
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'] 
                )
    return model