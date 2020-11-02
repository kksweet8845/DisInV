from django.db import models
from django.utils import timezone as tz
from .brand_model import Brand


class Report(models.Model):
    title   = models.CharField(
        max_length=200,
        blank=False,
        default=''
    )
    content = models.CharField(
        max_length=6000,
        blank=False,
        default=''
    )
    author  = models.CharField(
        max_length=40,
        blank=True,
        null=True
    )
    brand   = models.ForeignKey(Brand, on_delete=models.CASCADE)
    date    = models.DateField(
        auto_now=False,
        auto_now_add=False
    )
    update_time = models.DateTimeField(
        default=tz.now
    )
    url     = models.CharField(
        max_length=1000,
        default=''
    )

    def __str__(self):
        return ""

    class Meta:
        db_table = "reports"
