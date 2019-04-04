from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from .emotion_detection import *
from django.conf import settings
# Create your views here.

# vgg16Loaded = loadModel("saved_vgg16")
# vgg19Loaded = loadModel("saved_vgg19")
# resNetLoaded = loadModel("saved_resNet")
# conv2DLoaded = loadModel("saved_conv2D")


        
def index(request):

    
    if request.method == "POST" and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        img = "detectEmotion/images/happy.png"
        resizedImage = resizeImage(settings.MEDIA_ROOT + "/" + filename)
        
                
        
        vgg16Result = ""
        vgg19Result = ""
        resNetResult = ""
        conv2DResult = ""
        
        # vgg16Result = predictOnImage(vgg16Loaded,resizedImage)
        # vgg19Result = predictOnImage(vgg19Loaded,resizedImage)
        # resNetResult = predictOnImage(resNetLoaded,resizedImage)
        # conv2DResult = predictOnImage(conv2DLoaded,resizedImage)
        
    
        vgg16Obj = {"model":"vgg16", "emotion": vgg16Result, "image": img}
        vgg19Obj = {"model":"vgg19", "emotion": vgg19Result, "image": img}
        resnetObj = {"model":"resNet", "emotion": resNetResult, "image": img}
        conv2DObj = {"model":"conv2D", "emotion": conv2DResult, "image": img}
        
        context = {'uploaded_file_url': uploaded_file_url, 'vgg16': vgg16Obj,"vgg19": vgg19Obj, "resNet": resnetObj,"conv2D": conv2DObj}
        
        return render(request, 'detectEmotion/show_emotion.html',context) 
        
    clear_media_folder()
    
    return render(request, 'detectEmotion/index.html')


def show_emotion(request):
    return render(request, 'detectEmotion/show_emotion.html')



def clear_media_folder():
    folder = "media/"
    for f in os.listdir(folder):
        file_path = os.path.join(folder,f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
