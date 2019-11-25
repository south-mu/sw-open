from django.http.response import HttpResponse
from django.shortcuts import render # 追加する

from django.http import HttpResponse
from django_filters.views import FilterView

from app.models import Item


class soya(FilterView):
    i = 0
    model = Item

    def yaso(request):
        i = 0