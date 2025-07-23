from rest_framework import serializers
from .models import UserProfile

class UserProfileSerializer(serializers.Serializer):
    image = serializers.CharField()
    username = serializers.CharField(max_length=100)
    email = serializers.EmailField()
    address = serializers.CharField()
    phone = serializers.CharField()
    birth_date = serializers.DateField()

    def create(self, validated_data):
        return UserProfile(**validated_data).save()