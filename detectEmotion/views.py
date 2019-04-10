from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from .emotion_detection import *
from django.conf import settings
# Create your views here.

vgg16Loaded = loadModel("vgg16_end")
vgg19Loaded = loadModel("vgg19_end")
resNetLoaded = loadModel("resNet_end")
conv2DLoaded = loadModel("conv2D_end")


        
def index(request):

    
    if request.method == "POST" and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        img = "detectEmotion/images/happy.png"
        resizedImage = resizeImage(settings.MEDIA_ROOT + "/" + filename)
        
        
        
        vgg16Result = predictOnImage(vgg16Loaded,resizedImage)
        vgg19Result = predictOnImage(vgg19Loaded,resizedImage)
        resNetResult = predictOnImage(resNetLoaded,resizedImage)
        conv2DResult = predictOnImage(conv2DLoaded,resizedImage)
        emotions = ["neutral", "anger","contempt", "disgust","fear","happiness","sadness","surprise"]

        vgg16zip = zip(emotions,vgg16Result)
        vgg19zip = zip(emotions,vgg19Result)
        resNetzip = zip(emotions,resNetResult)
        ourModelzip = zip(emotions,conv2DResult)
        
        context = {'uploaded_file_url': uploaded_file_url, 'vgg16': vgg16zip,"vgg19": vgg19zip, "resNet": resNetzip,"conv2D": ourModelzip}
        
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
