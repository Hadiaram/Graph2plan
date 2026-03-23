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
    path(r'index/LoadTrainHouse/', views.LoadTrainHouse),
    path(r'index/TransGraph/', views.TransGraph),
    path(r'index/TransGraph_net/', views.TransGraph_net),
    path(r'index/Init/', views.Init),
    path(r'index/AdjustGraph/', views.AdjustGraph),
    path(r'index/AutoAdjustGraph/', views.AutoAdjustGraph),
    path(r'index/GraphSearch/', views.GraphSearch),
    path(r'index/RelBox/', views.RelBox),
    path(r'index/Save_Editbox/', views.Save_Editbox),
    path(r'index/OptimizeLayout/', views.OptimizeLayout),
    path(r'index/ExpandLivingRoom/', views.ExpandLivingRoom),
    path('index/Export_DXF/', views.Export_DXF),
    path('index/AlignWalls/', views.AlignWalls),
    path('index/FillWallGaps/', views.FillWallGaps),
    path('index/SnapRooms/', views.SnapRooms),
    path('index/FillLivingRoom/', views.FillLivingRoom),
    path('index/Log_Boundaries/', views.Log_Boundaries),
    path('index/Log_Graph/', views.Log_Graph),
    path('index/FixRooms/', views.FixRooms),
    path('index/AdjustBoundary/', views.AdjustBoundary),
    path('index/UpdateBoundary/', views.UpdateBoundary),
    path('index/SetScale/', views.SetScale),


    path('home', views.home),


]
