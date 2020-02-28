# coding: utf-8

import django_filters
from django.views.decorators.csrf import csrf_protect
from rest_framework import viewsets, filters, status, permissions
from rest_framework.decorators import detail_route
from rest_framework.decorators import list_route
from rest_framework.response import Response

# from blog.eval import deftest
from .models import User, Entry, Results, GameRound
from .serializer import  EntrySerializer, ResultSerializer, GameRoundSerializer
from .eval.deftest import Deftest
from django.http.response import HttpResponse
from django.shortcuts import render  # 追加する
from django.shortcuts import render_to_response
import time
import json
import urllib.request
import numpy as np
from .models import Userd, Post
from .serializer import UserSerializer, PostSerializer

class UserViewSet(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)
    queryset = Userd.objects.all()
    serializer_class = UserSerializer

class GameRoundViewSet(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)
    queryset = GameRound.objects.all()
    serializer_class = GameRoundSerializer

class ResultViewSet(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)
    queryset = Results.objects.all()
    serializer_class = ResultSerializer


#NOT USE
class PostViewSet(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)

    def post(self, request, format=None):
        queryset = Post.objects.all()
        serializer_class = PostSerializer(data=request.data)
        if serializer_class.is_valid():
            serializer_class.save()
            return Response(str("GOOD"))

        return Response(str(serializer_class))



#NOT USE
class UserdViewSet(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)
    queryset = User.objects.all()

    def esu9(self):
        User(name="hakoishssi", mail="yamada@rail.jp").save()

    serializer_class = UserSerializer

    # @csrf_protect
    @list_route(methods=['get'])
    def set_password(self, request, pk=None):
        # c = {csrf(request)}
        # user = self.get_object()
        deftest = Deftest()

        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            print(request)
            print("UserMethod" + str(deftest.AddOne(5)))
            # c = {csrf(request)}
            # return render_to_response({'status': 'password set111111'}, c)
            return Response({'status': 'password set111111'})
        else:
            print(request)
            # self.esu9()
            print("UserMethod" + str(deftest.AddOne(5)))
            return Response({'status': 'password SOYASAMA'})

    @detail_route(methods=['post'])
    def set_mail(self, request, pk=2):
        user = self.get_object()
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            # user.set_password(serializer.data['password'])
            # user.save()
            return Response({'status': 'password set'})
        else:
            return Response({'status': 'password SOYASAMAWWWWWWWW'})


#NOT USE
class EntryViewSet(viewsets.ModelViewSet):
    queryset = Entry.objects.all()
    serializer_class = EntrySerializer
    filter_fields = ('author', 'status')


#NOT USE
class ApiIndexView(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)

    def esu9(self):
        User(name="matsuba111", mail="kaku@rail.jp").save()

    def esu10(self, se):
        User(se.data['name'], se.data['mail']).save()

    def post(self, request, format=None):
        # self.esu9()
        serializer1 = ResultSerializer(data=request.data)
        if serializer1.is_valid():
            serializer1.save()
            print(serializer1.data['RoundScoreV'])
            # self.esu10(serializer1)
            time.sleep(2)
            a = np.random()
            us = Results.objects.filter(CellGameID=serializer1.data['CellGameID']).values()
            s = 'OK ' + serializer1.data['RoundScoreV'] + str(us)
            return Response({'res': str(s)})
        return Response("DEFAULT")

class ApiRoundResult(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)

    def esu10(self, se):
        User(se.data['name'], se.data['mail']).save()

    def post(self, request, format=None):
        # self.esu9()
        serializer1 = GameRoundSerializer(data=request.data)
        print("dsfdfdsfds")
        print(serializer1.fields)
        if serializer1.is_valid():
            # print(serializer1)
            serializer1.save()
            print(serializer1.data['RoundScore'])
            us = Results.objects.filter(CellGameID=serializer1.data['CellGameID']).values()
            score = np.random.rand() * 100
            return Response({'flag': '0', 'score': str(score)})
        return Response(str(serializer1.errors))


class ApiRound3Rec(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)
    def esu10(self, se):
        User(se.data['name'], se.data['mail']).save()

    def post(self, request, format=None):
        # self.esu9()
        serializer1 = GameRoundSerializer(data=request.data)
        print("dsfdfdsfds")
        print(serializer1.fields)
        if serializer1.is_valid():
            # print(serializer1)
            serializer1.save()
            print(serializer1.data['RoundScore'])
            # deftest.Audio(4)
            #def 音のスコアだけupdate
            # def SAVE round3
            score = np.random.rand() * 50
            return Response({'flag': '0','score': str(score)})
        return Response(str(serializer1.errors))

class ApiRound3Get(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)
    def esu10(self, se):
        User(se.data['name'], se.data['mail']).save()

    def post(self, request, format=None):
        # self.esu9()
        serializer1 = GameRoundSerializer(data=request.data)
        print("dsfdfdsfds")
        print(serializer1.fields)
        if serializer1.is_valid():
            # print(serializer1)
            #serializer1.save()
            print(serializer1.data['RoundScore'])
            # deftest.Audio(4)
            #def 音のスコアだけupdate
            # def SAVE round3
            us = GameRound.objects.filter(CellGameID=serializer1.data['CellGameID']).values()
            return Response(us)
        return Response(str(serializer1.errors))

# def トータル計算
# def ResultをRound４として保存

class ApiRound2Rec(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)

    def esu10(self, se):
        User(se.data['name'], se.data['mail']).save()

    def post(self, request, format=None):
        # self.esu9()
        serializer1 = GameRoundSerializer(data=request.data)
        print("dsfdfdsfds")
        print(serializer1.fields)
        if serializer1.is_valid():
            # print(serializer1)
            serializer1.save()
            print(serializer1.data['RoundScore'])
            # deftest.Audio(4)
            # 音のスコアだけupdate
            us = GameRound.objects.filter(CellGameID=serializer1.data['CellGameID']).values()
            s = 'OK ' + str(serializer1.data) + str(us)
            return Response({'flag': '0','score': '-1.0'})
        return Response(str(serializer1.errors))


#NOT USE
class ApiRecord(viewsets.ModelViewSet):
    permission_classes = (permissions.AllowAny,)

    def esu9(self):
        User(name="matsuba111", mail="kaku@rail.jp").save()

    def esu10(self, se):
        User(se.data['name'], se.data['mail']).save()

    def post(self, request, format=None):
        # self.esu9()
        deftest = Deftest()
        print("sssss")
        serializer1 = ResultSerializer(data=request.data)
        if serializer1.is_valid():
            serializer1.save()
            print(serializer1.data['RoundScoreV'])
            # deftest.Audio(4)
            # res = deftest.Rockon('dummy')
            us = GameRound.objects.filter(CellGameID=serializer1.data['CellGameID']).values()
            s = 'OK ' + str(serializer1.data['RoundScoreV']) + str(us)  # + str(res)
            return Response({'res': str(s)})
        return Response(str(serializer1.errors))


#NOT USE
def hello_template(request):
    return render(request, 'Soya.html')
