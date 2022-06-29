# Generated by Django 4.0.5 on 2022-06-29 02:14

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0025_ngram_stresses_attachment'),
    ]

    operations = [
        migrations.CreateModel(
            name='Line',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.CharField(max_length=500, unique=True)),
                ('ipa', models.CharField(blank=True, max_length=500, null=True)),
                ('phones', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), blank=True, db_index=True, null=True, size=None)),
                ('stresses', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), blank=True, db_index=True, null=True, size=None)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
    ]
