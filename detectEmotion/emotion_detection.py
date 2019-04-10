
#django frontend scripts
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.models import model_from_json
from keras import backend as b

#create dictionary with filepath, label
#training data -> labeled

#divide img by 255
#shuffle images randomly = True
#use unlabed images as training data
#VGGNet16, ResNet, VGGNet19


        
def resizeImage(img_file):
        img = image.load_img(img_file, target_size=(64, 64))
        resized_img = image.img_to_array(img) #numpy array
        resized_img = np.expand_dims(img, axis=0)
        #print(np.shape(resized_img))
        resized_img =  resized_img / float(255)
        #print(resized_img)
        return resized_img
                

def inputDataSummary(inputs,labels):
    print(np.shape(inputs))
    print(np.shape(labels))




def getEmotion(emotion):
    if emotion == 0:
        return "neutral"
    elif emotion == 1:
        return "anger"
    elif emotion == 2:
        return "contempt"
    elif emotion == 3:
        return "disgust"
    elif emotion == 4:
        return "fear"
    elif emotion == 5:
        return "happy"
    elif emotion == 6:
        return "sadness"
    elif emotion == 7:
        return "surprise"


def loadModel(model_name):
    model_folder = "modelWeights/" 
    model_url = "detectEmotion/static/detectEmotion/" + model_folder + model_name + ".json"
    print("m " + model_url)
    json_file = open(model_url,"r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("detectEmotion/static/detectEmotion/" + model_folder + model_name + ".h5")
    print("loading model and recompiling...")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    loaded_model._make_predict_function()

    return loaded_model
    

def runModel(model,model_name, inputs, labels, test_data):
    filepath = "../saved/"+ model_name + "/" + model_name + "_weights-improvement-{epoch:02d}.h5"
    print("filepath " + filepath)
    checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(inputs, labels,validation_split=0.05, epochs=1, batch_size=32, shuffle=True, callbacks=callbacks_list) #TODO: set to 500
    model.summary()
    score = model.evaluate(inputs, labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    


def predictOnImage(loaded_model,image):
    preds = loaded_model.predict(image,verbose=0)
    mylist = []
    for i in preds:
        for g in i:
            mylist.append(round(g*100,4))
    return mylist
      
#    preds = loaded_model.predict_classes(image,verbose=0)
#    print("preds val " + str(preds))
#    prediction = loaded_model.predict(image)
#    print("predict " + str(prediction))
#    pred_prob = loaded_model.predict_proba(image)
#    print("X=%s, Predicted=%s" % (image, pred_prob))
#    #b.clear_session() 
#
#    print("preds " + str(preds))
#    return getEmotion(preds)
        

def evaluateLoadedModel(loaded_model,inputs,labels):
    score = loaded_model.evaluate(inputs, labels, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

