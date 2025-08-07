from django.shortcuts import render

from .models import predict
from django.shortcuts import render
import numpy as np


class_names = np.array(['fire','No Fire'])
# Create your views here.
def home(request):
	return render(request,'index.html')
def input(request):
	return render(request,'input.html')
def output(request):
    algo=request.POST.get('algo')
    img=request.FILES['file']
    #print(row)
    out=predict(img,algo)
    print(out)
    out=np.argmax(out)
    classes = class_names[out]
    print(class_names)
    return render(request,'output.html',{'out':classes})