"""
URL configuration for insurtech_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat, name='chat'),
    path('ai-response/', views.ai_response, name='ai_response'),
    path('we-do/', views.we_do_page, name='we_do'),
    path('products/', views.products_page, name='products'),
    path('claims-and-services/', views.claims_services_page, name='claims_services'),
    path('work-with-us/', views.work_with_us_page, name='work_with_us'),
    path('about-us/', views.about_us_page, name='about_us'),
]
