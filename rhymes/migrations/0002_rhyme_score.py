# Generated by Django 4.2.4 on 2023-09-03 22:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("rhymes", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="rhyme",
            name="score",
            field=models.FloatField(blank=True, db_index=True, null=True),
        ),
    ]
