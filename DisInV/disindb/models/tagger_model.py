from django.db import models
from .report_model import Report
from datetime import date


class Tagger(models.Model):
    report  = models.ForeignKey(Report, on_delete=models.CASCADE)
    split   = models.TextField()
    date = models.DateField(
        auto_now=False,
        auto_now_add=False,
        default=date.today().isoformat()
    )

    class Meta:
        db_table = "taggers"