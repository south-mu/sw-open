# coding: utf-8

from rest_framework import serializers

from blog.eval.deftest import Deftest
from .models import GameRound, Results
from .models import Userd, Post, User, Entry


class UserdSerializer(serializers.ModelSerializer):
    class Meta:
        model = Userd
        fields = ('id', 'name',)

        # idの読み込み専用をオフ
        extra_kwargs = {'id': {'read_only': False}}


class PostSerializer(serializers.ModelSerializer):
    author = UserdSerializer()

    class Meta:
        model = Post
        fields = ('id', 'title', 'body', 'author', 'created_at', 'updated_at')
        read_only_fields = ('created_at', 'updated_at')

    def save(self):
        id = self.validated_data['id']
        title = self.validated_data['name']
        body = self.validated_data['body']
        author = "aaa"
        created_at = self.validated_data['created_at']
        updated_at = self.validated_data['updated_at']

    # ネストされたモデルのPOSTに対応する
    def create(self, validated_data):
        user_data = validated_data.pop('author')
        user = Userd.objects.get(pk=user_data['id'])
        post = Post.objects.create(author=user, **validated_data)
        return post


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        # deftest = Deftest()
        # print("UserMethod" + str(deftest.AddOne(5)))
        model = User
        fields = ('name', 'mail')


class GameRoundSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameRound
        fields = ('id', 'CellGameID', 'RoundNum',
                  'RoundScore',
                  'RoundFlag',
                  'Errcode')


class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Results
        fields = ('id','CellGameID' ,'RoundNum',
                  'Score1',
                  'Score2',
                  'Errcode')


class EntrySerializer(serializers.ModelSerializer):
    author = UserSerializer(read_only=True)
    deftest = Deftest()
    print("EnMethod" + str(deftest.AddOne(5)))

    class Meta:
        model = Entry
        fields = ('id', 'title', 'body', 'created_at', 'status', 'author')
