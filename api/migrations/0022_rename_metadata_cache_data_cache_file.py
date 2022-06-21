# Generated by Django 4.0.5 on 2022-06-20 22:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0021_cache'),
    ]

    operations = [
        migrations.RenameField(
            model_name='cache',
            old_name='metadata',
            new_name='data',
        ),
        migrations.AddField(
            model_name='cache',
            name='file',
            field=models.FileField(blank=True, null=True, upload_to='data/cache'),
        ),
    ]