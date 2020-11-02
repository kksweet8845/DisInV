# Generated by Django 3.0.6 on 2020-09-19 18:04

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Brand',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('brand_name', models.CharField(default='', max_length=20)),
            ],
            options={
                'db_table': 'brands',
            },
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(default='', max_length=200)),
                ('content', models.CharField(default='', max_length=6000)),
                ('author', models.CharField(blank=True, max_length=15, null=True)),
                ('date', models.DateField()),
                ('update_time', models.DateTimeField(default=datetime.datetime(2020, 9, 19, 18, 4, 21, 92716))),
                ('url', models.CharField(default='', max_length=1000)),
                ('brand', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='disindb.Brand')),
            ],
            options={
                'db_table': 'reports',
            },
        ),
        migrations.CreateModel(
            name='Tagger',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('split', models.TextField()),
                ('date', models.DateField(default='2020-09-19')),
                ('report', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='disindb.Report')),
            ],
            options={
                'db_table': 'taggers',
            },
        ),
    ]