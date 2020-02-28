# coding: utf-8

from rest_framework import routers
from .views import UserViewSet, EntryViewSet ,GameRoundViewSet ,ResultViewSet


router = routers.DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'entries', EntryViewSet)
router.register(r'round', GameRoundViewSet)
router.register(r'result', ResultViewSet)


