"""django_rest_framework_test URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Import the include() function: from django.conf.urls import url, include
    3. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import url, include
from django.contrib import admin

from blog import views
from blog.urls import router as blog_router

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^api/', include(blog_router.urls)),
    url(r'^AbilityTest/$', views.hello_template, name='hello_template'),  # 追加する
    url(r'^api/index/$', views.ApiIndexView.as_view({'get': 'list'})),
    url(r'^api/AbilityGameReport/$', views.ApiRecord.as_view({'get': 'list'})),
    url(r'^api/ApiRound2RecStart/$', views.ApiRound2Rec.as_view({'get': 'list'})),
    url(r'^api/ApiRound3RecStart/$', views.ApiRound3Rec.as_view({'get': 'list'})),
    url(r'^api/ApiRound3ScoreGet/$', views.ApiRound3Get.as_view({'get': 'list'})),
    url(r'^api/ApiResultGet/$', views.ApiRoundResult.as_view({'get': 'list'})),
    url(r'^api/UUU/$', views.PostViewSet.as_view({'get': 'list'})),


]
