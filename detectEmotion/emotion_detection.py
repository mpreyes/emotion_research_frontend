
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

def traverseFolders(rootdir,extension,labels):
    labeled_data = []
    test_data = []
    list_subdirs = [] #should be 123 "outer dirs in cohn-kanade-imgs"
    for root, dirs, files, in os.walk(rootdir):
        for d in dirs:
            if not d.startswith("S"): #grabbing the inner dirs, should be 593
                list_subdirs.append(os.path.join(root,d))

    files_seq = 0
    for i in list_subdirs:
        for rootdir, dirs, files in os.walk(i): #traversing through subdirs to grab files
            for f in files:
                temp_arr = []
                if f.endswith(extension) and not f.startswith("."): #-1 is if the sequence is not labeled
                    temp_arr.append(labels[files_seq])
                    temp_arr.append(os.path.join(i,f))
                    if labels[files_seq] != -1:
                        labeled_data.append(temp_arr)
                    elif labels[files_seq] == -1:
                         test_data.append(temp_arr)
                   
        files_seq += 1

    return labeled_data, test_data
        


def getLabels(rootdir,extension): #grabbing labels from labels folder
    labels = []
    for root, dirs, files, in os.walk(rootdir):
        if len(files) != 0: 
            labels.append(-1) #meaning no label was assigned for this sequence of images 
            for f in files:
                if f.endswith(extension) and not f.startswith("."):
                    f = open(os.path.join(root,f),"r")
                    labels.append(int(float(f.readline().strip("\n"))))
    return labels


def resizeImages(img_files,x,y):
        for i in range(len(img_files)): 
            img = image.load_img(img_files[i][1], target_size=(224, 224))
            resized_img = image.img_to_array(img) #numpy array
            x.append(resized_img)
            y.append(img_files[i][0])

        
def resizeImage(img_file):
        img = image.load_img(img_file, target_size=(224, 224))
        resized_img = image.img_to_array(img) #numpy array
        resized_img = np.expand_dims(img, axis=0)
        print(np.shape(resized_img))
        return resized_img
                

def inputDataSummary(inputs,labels):
    print(np.shape(inputs))
    print(np.shape(labels))


def sequentialModel():
    model = Sequential()
    model.add(Flatten(input_shape=(224, 224, 3)))
    model.add(Dense(units=224, activation='relu'))
    model.add(Dense(units=8, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return model

#increase img size 224 * 224, scale and divide by 256

#TODO: fix this boi
def conv2DModel(): #add dropout layers, 
    model = Sequential()
    model.add(Conv2D(224, kernel_size=(5, 5), strides=(1, 1), activation='relu',data_format = "channels_last",input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(224, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(224, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


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

def saveModel(model,model_name):
    model_json = model.to_json()

    with open( model_name + ".json","w") as json_file:
        json_file.write(model_json)  
    model.save_weights(model_name + ".h5")  
    print("saving model...")

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
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    loaded_model._make_predict_function()
    return loaded_model
    

def runModel(model,model_name, inputs, labels, test_data):
    filepath = "../saved/"+ model_name + "/" + model_name + "_weights-improvement-{epoch:02d}.h5"
    print("filepath " + filepath)
    checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(inputs, labels, epochs=1, batch_size=32, shuffle=True, callbacks=callbacks_list) #TODO: set to 500
    model.summary()
    score = model.evaluate(inputs, labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #probs = model.predict(inputs)
    #y_classes = probs.argmax(axis=-1)
    #print(y_classes)
    #class_labels = [0,1,2,3,4,5,6,7,8]
    #spred = model.argmax(class_labels, axis=-1)
    #print(class_labels[pred[0]])
    


def notMyModel(model,inputs,labels,test_data):
    newModel = Sequential()
    newModel.add(model)
    newModel.add(Flatten(input_shape=(224, 224, 3)))
    newModel.add(Dense(224, activation='relu'))
    newModel.add(Dense(8, activation='softmax'))
    newModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return newModel


def predictOnImage(loaded_model,image):
    
    preds = loaded_model.predict_classes(image,verbose=0)
    print("preds val " + str(preds))
    #b.clear_session() 
    return getEmotion(preds)
        

def evaluateLoadedModel(loaded_model,inputs,labels):
    score = loaded_model.evaluate(inputs, labels, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

