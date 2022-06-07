# Generated by Django 4.0.5 on 2022-06-07 21:22

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_writer_alt_names'),
    ]

    operations = [
        migrations.AlterField(
            model_name='writer',
            name='alt_names',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=300, unique=True), default=list, size=None),
        ),
    ]
