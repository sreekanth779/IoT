
from django.urls import path
from . import views
urlpatterns = [
    path("",views.dashboard,name="dashboard"),
    path("analytics/",views.analytics,name="analytics"),
    path("ajax/",views.send_jsonresponse,name="ajaxreq"),
    path("login/",views.loginPage,name = "login"),
    path("logout/",views.logoutUser,name = "logout"),
    path("register/",views.registerUser,name = "register"),
    path("about/",views.about_page,name = "about_page"),
    path("devicesprediction/",views.devices_prediction,name = "devices_prediction"),

]
