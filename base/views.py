from django.shortcuts import render,redirect

# Create your views here.
# views.py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.http import HttpResponse,JsonResponse
import altair as alt

from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm

from  . utils import calling_func,get_device_statuses,plot_pie_chart,devices,single_device_plot,all_devices_locations,calling_model



def loginPage(request):

    page = 'login'

    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get("username")
        password = request.POST.get("password")

        try:
            user = User.objects.get(username = username)
        except:
            messages.error(request,'User does not exist')

        user = authenticate(request, username = username,password = password)

        if user is not None:
            login(request,user)
            return redirect('dashboard')
        else:
            messages.error(request,'User or password does not exist')
        
        
    context ={'page':page}
    return render(request,"login.html",context)

def logoutUser(request):
    logout(request)
    return redirect('dashboard')


def registerUser(request):
    page = 'register'
    form = UserCreationForm()

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()

            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')

            user = authenticate(username=username, password=raw_password)

            if user is not None:
                login(request, user)

            return redirect('dashboard')
        else:
            messages.error(request,"An error occured during registration")

    context = {'form':form}
    return render(request,'login.html',context)



@login_required(login_url ='login')
def dashboard(request):
    
    img_str1 = calling_func()
    status,active_count,total_count = get_device_statuses()
    img_str2 = plot_pie_chart()
    all_locations,locat = all_devices_locations()
    # output_pred = calling_model()

    # Render the template with the plot image
    return render(request, 'dashboard.html', {'plot_img1': img_str1,'plot_img2':img_str2,'status':status,'active_count':active_count,'total_count':total_count,'all_locations':all_locations,'locat':locat})


@login_required(login_url ='login')
def analytics(request):
    devicest = devices()
    context = {
        'devicest': devicest,
    }
    return render(request,'analytics.html',context)


def about_page(request):
    return render(request,"about.html")




def send_jsonresponse(request):
    if request.method == 'POST':
        option = request.POST.get('option')
        
        # Generate plot based on the selected option
        img_str,temp_str,latt,long  = single_device_plot(option)
        print(latt,long)
        return JsonResponse({'image': img_str,'temper':temp_str,'latitude':latt,'longitude':long})


def devices_prediction(request):
    output_pred,devices_all = calling_model()
    fuwork = zip(output_pred,devices_all)
    # range(len(devices))
    context = {"output_pred":output_pred,"devices_all":devices_all,"fuwork":fuwork}
    return render(request,"Devicestable.html",context)