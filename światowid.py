import numpy as np
import tensorflow as tf
from keras.layers import Activation, Add, Conv2D, MaxPooling2D,Conv2D,Concatenate,Flatten,Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.vis_utils import plot_model

seed=777
np.random.seed(seed) 
tf.random.set_seed(seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    #Restrict Tensorflow to only allocate 6gb of memory on the first GPU
   try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
       #virtual devices must be set before GPUs have been initialized
        print(e)

train = tf.keras.preprocessing.image_dataset_from_directory(
  directory=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/",
  labels="inferred",
  label_mode="categorical",
  
  color_mode="rgb", 
  image_size=(600, 450),
  shuffle=True,
  seed=seed+123,
  validation_split=0.4,
  subset='training',
  batch_size=32
  )
val = tf.keras.preprocessing.image_dataset_from_directory(
  directory=r"/home/geniusz/nn/swiatowidCNN/HAM10000_images_part_1/",
  labels="inferred",
  label_mode="categorical",
  
  color_mode="rgb",
  batch_size=32, 
  image_size=(600, 450),
  shuffle=True,
  seed=seed+123,
  validation_split=0.4,
  subset='training',
  )



def światowid(n,classes,BatchSize):
    inputs=tf.keras.layers.Input(shape=(600,450,3),batch_size=BatchSize)

    x1=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(inputs)
    x2=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(inputs)
    x3=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(inputs)
    x4=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(inputs)

    add1=Add()([x1,x2,x3,x4])


    x1=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(add1)
    x2=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(add1)
    x3=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(add1)
    x4=Conv2D(n,(3,3),(2,2),padding="same",activation="relu")(add1)

    add2=Add()([x1,x2,x3,x4])

    x1=MaxPooling2D((2,2),(1,1))(add2)
    x2=MaxPooling2D((2,2),(1,1))(add2)
    x3=MaxPooling2D((2,2),(1,1))(add2)
    x4=MaxPooling2D((2,2),(1,1))(add2)

    x1=Conv2D(2*n,(3,3),(1,1),padding="same",activation="relu")(x1)
    x2=Conv2D(2*n,(3,3),(1,1),padding="same",activation="relu")(x2)
    x3=Conv2D(2*n,(3,3),(1,1),padding="same",activation="relu")(x3)
    x4=Conv2D(2*n,(3,3),(1,1),padding="same",activation="relu")(x4)

    add3=Add()([x1,x2,x3,x4])

    x1=Conv2D(2*n,(3,3),(2,2),padding="same",activation="relu")(add3)
    x2=Conv2D(2*n,(3,3),(2,2),padding="same",activation="relu")(add3)
    x3=Conv2D(2*n,(3,3),(2,2),padding="same",activation="relu")(add3)
    x4=Conv2D(2*n,(3,3),(2,2),padding="same",activation="relu")(add3)

    add4=Add()([x1,x2,x3,x4])

    x1=MaxPooling2D((2,2),(1,1))(add4)
    x2=MaxPooling2D((2,2),(1,1))(add4)
    x3=MaxPooling2D((2,2),(1,1))(add4)
    x4=MaxPooling2D((2,2),(1,1))(add4)

    x1=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(x1)
    x2=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(x2)
    x3=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(x3)
    x4=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(x4)

    add5=Add()([x1,x2,x3,x4])

    x1=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(add5)
    x2=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(add5)
    x3=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(add5)
    x4=Conv2D(4*n,(3,3),(1,1),padding="same",activation="relu")(add5)

    add6=Add()([x1,x2,x3,x4])


    x1=Conv2D(4*n,(3,3),(2,2),padding="same",activation="relu")(add6)
    x2=Conv2D(4*n,(3,3),(2,2),padding="same",activation="relu")(add6)
    x3=Conv2D(4*n,(3,3),(2,2),padding="same",activation="relu")(add6)
    x4=Conv2D(4*n,(3,3),(2,2),padding="same",activation="relu")(add6)

    add_final=Add()([x1,x2,x3,x4])

    x=MaxPooling2D((2,2),(1,1))(add_final)
    x=Flatten()(x)

    d1=Dense(8*classes,activation="relu")(x)
    d2=Dense(8*classes,activation="relu")(d1)

    # dense_add=Add()([d1,d2])

    outputs=Dense(classes,activation="softmax")(d2)

    return tf.keras.Model(inputs,outputs)
with tf.device('/device:GPU:0'):
  model=światowid(1,7,32)
  model.summary()
  # plot_model(model, to_file=r"C:/Users/TK/Desktop/swiatowid.png", show_shapes=True, show_layer_names=True)
  opti=tf.keras.optimizers.AdamW(0.001,0.001)

  model.compile(optimizer=opti, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.IoU(num_classes=7,target_class_ids=[0,1,2,3,4,5,6]),tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

  my_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=r'/home/geniusz/nn/catcross.1b.2n.Epoka-{epoch:02d}_loss-{val_loss:.4f}_IoU-{val_io_u:.4f}_binAcc-{val_categorical_accuracy:.4f}.h5',verbose=1,monitor="val_categorical_accuracy",save_weights_only=False,mode="max",save_best_only=True)
  model.fit(train, batch_size=32, epochs=400, verbose=1,callbacks=my_callbacks,validation_data=val,validation_batch_size=32)