#!/usr/bin/env python
import numpy as np, os, sys, joblib
import joblib
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import tensorflow_addons as tfa






def create_model():
    inputA = tf.keras.layers.Input(shape=(5000,12)) 
    
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8,input_shape=(5000,12), padding='same')(inputA)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    model1 = keras.Model(inputs=inputA, outputs=gap_layer)

    conv1 = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(inputA)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
    conv1 = keras.layers.Dropout(rate=0.2)(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    # conv block -2
    conv2 = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
    conv2 = keras.layers.Dropout(rate=0.2)(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    # conv block -3
    conv3 = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.Dropout(rate=0.2)(conv3)
    # split for attention
    attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
    # attention mechanism
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
    # last layer
    dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
    dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
    # output layer
    flatten_layer = keras.layers.Flatten()(dense_layer)
    model2 = keras.Model(inputs=inputA, outputs=flatten_layer)

    combined = keras.layers.concatenate([model1.output, model2.output])
    final_layer = keras.layers.Dense(27, activation="sigmoid")(combined)
    model = keras.models.Model(inputs=inputA, outputs=final_layer)



    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), 
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', dtype=None, threshold=0.5)])
  return model


def run_12ECG_classifier(data,header_data,loaded_model):
    


    threshold = np.array([0.12820681, 0.06499375, 0.13454682, 0.16845625, 0.1470617 ,
0.2161416 , 0.16106858, 0.1051053 , 0.16673433, 0.21358207,
0.17808011, 0.05360209, 0.0879685 , 0.06232401, 0.11914249,
0.00379602, 0.15083605, 0.20306677, 0.15644205, 0.13406455,
0.17194449, 0.11921279, 0.21419376, 0.16725275, 0.17113625,
0.08283495, 0.09289312])


    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model
    padded_signal = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
    reshaped_signal = padded_signal.reshape(1,5000,12)

    #Rule-based model
    avg_hr = 0
    peaks = 0
    try:
        peaks = DetectRWithPanTompkins(data[1],int(header_data[0].split()[2]))
        
        try:
            peaks = R_correction(data[0], peaks)
        except:
            print("Did not manage to do R_correction")
        
    except:
        print("Did not manage to find any peaks using Pan Tomkins")

          
    try:
        rr_interval, avg_hr = heartrate(peaks,int(header_data[0].split()[2]))
    except:
        print("not able to calculate heart rate")
        rr_interval = 0
        avg_hr = 0

    gender = header_data[14][6:-1]
    age=header_data[13][6:-1]
    if gender == "Male":
        gender = 0
    elif gender == "male":
        gender = 0
    elif gender =="M":
        gender = 0
    elif gender == "Female":
        gender = 1
    elif gender == "female":
        gender = 1
    elif gender == "F":
        gender = 1
    elif gender =="NaN":
        gender = 2

    # Age processing - replace with nicer code later
    if age == "NaN":
        age = -1
    else:
        age = int(age)

    demo_data = np.asarray([age,gender])
    reshaped_demo_data = demo_data.reshape(1,2)

    #combined_data = [reshaped_signal,reshaped_demo_data]


    score  = model.predict(reshaped_signal)[0]
    
    binary_prediction = score > threshold
    binary_prediction = binary_prediction * 1

    if avg_hr != 0:     # bare gjør disse endringene dersom vi klarer å beregne puls
        if 60 < avg_hr < 100:
            binary_prediction[16] = 0
            binary_prediction[14] = 0
            binary_prediction[13] = 0
        elif avg_hr < 60 & binary_prediction[15] == 1:
            binary_prediction[13] = 1
        elif avg_hr < 60 & binary_prediction[15] == 0:
            binary_prediction[14] = 1
        elif avg_hr > 100:
            binary_prediction[16] = 1


    classes = ['10370003','111975006','164889003','164890007','164909002','164917005','164934002','164947007','17338001',
        '251146004','270492004','284470004','39732003','426177001','426627000','426783006','427084000','427172004','427393009','445118002','47665007','59118001',
        '59931005','63593006','698252002','713426002','713427006']

    return binary_prediction, score, classes

def load_12ECG_model(model_input):
    model = create_model()
    f_out='model.h5'
    filename = os.path.join(model_input,f_out)
    model.load_weights(filename)

    return model
