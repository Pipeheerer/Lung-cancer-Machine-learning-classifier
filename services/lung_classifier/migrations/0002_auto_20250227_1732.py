# Generated by Django 3.2.25 on 2025-02-27 17:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('lung_classifier', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='covidmodel',
            name='image_url',
            field=models.CharField(blank=True, max_length=2250),
        ),
        migrations.AlterField(
            model_name='covidmodel',
            name='output',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AlterField(
            model_name='lungcancerctmodel',
            name='image_url',
            field=models.CharField(blank=True, max_length=2250),
        ),
        migrations.AlterField(
            model_name='lungcancerctmodel',
            name='output',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]
