"""House URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.shortcuts import redirect

from Houseweb import views

def root_redirect(request):
    return redirect('/home')

urlpatterns = [
    # path('admin/', admin.site.urls),
    # Redirect root to home page
    path('', root_redirect),

    path('index/LoadTestBoundary', views.LoadTestBoundary),
    path('index/NumSearch/', views.NumSearch),
    path('index/LoadTrainHouse/', views.LoadTrainHouse),
    path('index/TransGraph/', views.TransGraph),
    path('index/TransGraph_net/', views.TransGraph_net),
    path('index/Init/', views.Init),
    path('index/AdjustGraph/', views.AdjustGraph),
    path('index/AutoAdjustGraph/', views.AutoAdjustGraph),
    path('index/GraphSearch/', views.GraphSearch),
    path('index/RelBox/', views.RelBox),
    path('index/Save_Editbox/', views.Save_Editbox),
    path('index/Refine_Floorplan/', views.Refine_Floorplan),  # Manual refinement button
    path('index/Export_DXF/', views.Export_DXF),  # Manual DXF export button
    path('index/Log_Boundaries/', views.Log_Boundaries),  # Boundary diagnostic button
    path('index/Download_Logs/', views.Download_Logs),   # Download last refinement log

    path('home', views.home),


]
