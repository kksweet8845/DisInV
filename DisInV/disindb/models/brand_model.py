from django.db import models


class Brand(models.Model):
    brand_name = models.CharField(
        max_length=20,
        blank=False,
        default=''
    )


    def __str__(self):
        return "brand_name: {}".format(self.brand_name)

    class Meta:
        db_table = "brands"