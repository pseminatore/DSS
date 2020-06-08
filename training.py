import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from matplotlib import pyplot
from twilio.rest import Client
import numpy
import auth


_TRAINING_SIZE = 1000
_VALIDATION_SIZE = 500

def main():
    try:
        print("main -> Loading...")
        dataset = loadFromTFRecord()
        
        print("main -> Initializing...")
        model = initializeModel()
        
        print("main -> Fitting...")
        # fit model
        history = model.fit(dataset['trainingLabels'], dataset['trainingSamples'], validation_data=(dataset['validationLabels'], dataset['validationSamples']), batch_size = 1000, epochs=50, verbose=1)
        
        print("main -> Evaluating...")
        # evaluate the model
        train_acc = model.evaluate(dataset['trainingLabels'],  dataset['trainingSamples'], verbose=0)
        test_acc = model.evaluate(dataset['validationLabels'], dataset['validationSamples'], verbose=0)
        print("main -> Done")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        
        #sendStatusUpdate("Your results are here!\nTrain: %.3f, Test: %.3f" % (train_acc[0], test_acc[0]))
        print('Train: %.3f, Test: %.3f' % (train_acc[0], test_acc[0]))
        print(history.history)
        
        
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(history.history['accuracy'], label='train')
        pyplot.plot(history.history['val_accuracy'], label='test')
        pyplot.legend()
        
        pyplot.show()
    except Exception as err:
        print(err)
        sendStatusUpdate(err)
        pass
    
def loadFromTFRecord():
    
    raw_training_dataset = tf.data.TFRecordDataset('data/nsynth-train.tfrecord')
    raw_validation_dataset = tf.data.TFRecordDataset('data/nsynth-valid.tfrecord')
    
    
    # Convert features into tensors
    features = {
    "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64),
    "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32)
    }

    # Parsing function
    parse_function = lambda example_proto: tf.io.parse_single_example(example_proto,features)
    
    # Map parsing function to each dataset to extract features
    training_dataset = raw_training_dataset.map(parse_function)
    validation_dataset = raw_validation_dataset.map(parse_function)
    
    trainX = numpy.empty  # Training label, in this case, Audio
    trainY = numpy.empty  # Training value, in this case, Pitch
    
    validX = numpy.empty   # Validation label, in this case, Audio
    validY = numpy.empty   # Validation value, in this case, Pitch
    
    print("loadFromTFRecord ------------------> Extracting Training...")
    # Extract label and audio from each training Example
    example_num = 0
    for record in training_dataset.take(_TRAINING_SIZE):
        #example = tf.train.Example()
        #example.ParseFromString(record)
        #trainX.append(list(example.features.feature['pitch'].int64_list.value))
        #trainY.append(list(example.features.feature['audio'].float_list.value))
        trainX = numpy.append(trainX, record['audio'])
        trainY = numpy.append(trainY, record['pitch'])
        example_num += 1
        print("loadFromTFRecord ------------------> Extracted Example {0}... -->{1}".format(example_num, record['pitch']))
        
    print("loadFromTFRecord ------------------> Extracting Validation...")   
    # Extract label and audio from each validation Example
    example_num = 1
    for record in validation_dataset.take(_VALIDATION_SIZE):
        print("loadFromTFRecord ------------------> Extracting Example {0}...".format(example_num))
        validX = numpy.append(validX, record['audio'])
        validY = numpy.append(validY, record['pitch'])
        example_num += 1
    
    # Drop random 'Special Function' that gets prepended to each numpy arr for some reason
    trainX = trainX[1:]
    validX = validX[1:]
    trainY = trainY[1:]
    validY = validY[1:]
    
    # Reshape audio arrays
    trainX = numpy.reshape(trainX, [_TRAINING_SIZE, 64000])
    validX = numpy.reshape(validX, [_VALIDATION_SIZE, 64000])
    
    #compile sets into parsed dataset
    parsed_dataset = {
        'trainingLabels' : tf.convert_to_tensor(trainX, numpy.int64),
        'trainingSamples' : tf.convert_to_tensor(trainY, numpy.float32),
        'validationLabels' : tf.convert_to_tensor(validX, numpy.int64),
        'validationSamples' : tf.convert_to_tensor(validY, numpy.float32),
    }
    
    print("loadFromTFRecord ------------------> Done")
    return parsed_dataset

    
def initializeModel():
    model = Sequential()
    
    # Adds a densely-connected layer with 64 units to the model:
    model.add(Dense(64, activation='tanh'))
    
    # Add another:
    model.add(Dense(64, activation='tanh'))

    # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
    model.add(Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01)))

    # Add an output layer with 7 output units, one for each note:
    model.add(Dense(7, activation='softmax'))
    
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def sendStatusUpdate(status_message):
    
    # Twilio Account SID and Auth Token
    client = Client(auth.ACCT_SID, auth.AUTH_TOKEN)
    # change the "from_" number to your Twilio number and the "to" number
    # to the phone number you signed up for Twilio with, or upgrade your
    # account to send SMS to any phone number
    client.messages.create(to=auth.USER_PHONE, 
                        from_=auth.CLIENT_PHONE, 
                        body=status_message)
        
if __name__ == '__main__':
    main()