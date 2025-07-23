from django.urls import path 
from .views import index, upload, capture, train , login_view ,cards,save_form,form_submit,check_phone ,detect_face 

urlpatterns = [
    path("", index, name='index'),
    path("upload/", upload, name='upload'),
    path("capture/", capture, name='capture'),
    path("training/", train, name='train'),
    path("detect_face/", detect_face, name='detect_face'),
    path("login_view/", login_view, name='login_view'),
    path('cards/', cards, name='cards'),
    path('save_form/', save_form, name='save_form'),
    path('form_submit/', form_submit, name='form_submit'),
    path('check_phone/',check_phone,name='check_phone'),
]