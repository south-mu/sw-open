from django.db import models


class User(models.Model):
    name = models.CharField(max_length=32)
    mail = models.EmailField()

    def __repr__(self):
        return "{}: {}".format(self.pk, self.name)

    __str__ = __repr__

# マリエッタが送信するじょうほう

class GameRound(models.Model):
    CellGameID = models.IntegerField()
    RoundNum = models.IntegerField()
    RoundScore = models.FloatField()#1ラウンド=距離、2ラウンド=NULL、3=距離
    RoundFlag = models.IntegerField()
    Errcode = models.IntegerField(default=0)

    def __repr__(self):
        return "{}: {}".format(self.pk, self.CellGameID)

    __str__ = __repr__

#島村システムが送信する情報

class Results(models.Model):
    CellGameID = models.IntegerField()
    RoundNum = models.IntegerField()
    Score1 = models.FloatField()
    Score2 = models.FloatField()
    Errcode = models.IntegerField(default=0)

    def __repr__(self):
        return "{}: {}".format(self.pk, self.Score1)

#TEST

class Entry(models.Model):
    STATUS_DRAFT = "draft"
    STATUS_PUBLIC = "public"
    STATUS_SET = (
        (STATUS_DRAFT, "下書き"),
        (STATUS_PUBLIC, "公開中"),
    )
    title = models.CharField(max_length=128)
    body = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(choices=STATUS_SET, default=STATUS_DRAFT, max_length=8)
    author = models.ForeignKey(User, related_name='entries', on_delete=models.CASCADE)

    def __repr__(self):
        return "{}: {}".format(self.pk, self.title)

    __str__ = __repr__

class Userd(models.Model):
  name = models.CharField(max_length=64)

class Post(models.Model):
  title = models.CharField(max_length=128)
  body = models.TextField()
  author = models.ForeignKey(Userd, on_delete=models.CASCADE)
  created_at = models.DateTimeField(auto_now_add=True)
  updated_at = models.DateTimeField(auto_now=True)