from rest_framework import serializers
from disindb.models import Report


class ReportSerializer(serializers.ModelSerializer):

    brand = serializers.SlugRelatedField(
        many=False,
        read_only=True,
        slug_field='id'
    )

    class Meta:
        model = Report
        fields = '__all__'