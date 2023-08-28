# Generated by Django 4.2.4 on 2023-08-27 20:56

import songs.managers
import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion
import songs.models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("contenttypes", "0002_remove_content_type_name"),
    ]

    operations = [
        migrations.CreateModel(
            name="Artist",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=300, unique=True)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Song",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("is_new", models.BooleanField(default=True)),
                ("title", models.CharField(db_index=True, max_length=300)),
                ("lyrics", models.TextField(blank=True, null=True)),
                ("rhymes_raw", models.TextField(blank=True, null=True)),
                ("spotify_id", models.SlugField()),
                ("jaxsta_id", models.SlugField(blank=True, null=True)),
                ("youtube_id", models.SlugField(blank=True, null=True, unique=True)),
                (
                    "audio_file",
                    models.FileField(blank=True, null=True, upload_to="data/audio"),
                ),
                ("metadata", models.JSONField(blank=True, null=True)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True, null=True)),
                (
                    "artists",
                    models.ManyToManyField(
                        blank=True, related_name="songs", to="songs.artist"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Tag",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("value", models.SlugField()),
                ("label", models.CharField(max_length=100)),
                (
                    "category",
                    models.CharField(
                        choices=[("song", "Song"), ("snippet", "Snippet")],
                        default="song",
                        max_length=100,
                    ),
                ),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True, null=True)),
            ],
            options={
                "unique_together": {("category", "value")},
            },
        ),
        migrations.CreateModel(
            name="Writer",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=300, unique=True)),
                (
                    "alt_names",
                    django.contrib.postgres.fields.ArrayField(
                        base_field=models.CharField(max_length=300, unique=True),
                        blank=True,
                        default=list,
                        size=None,
                    ),
                ),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="TaggedText",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("text", models.TextField(db_index=True)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True, null=True)),
                (
                    "song",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="tagged_texts",
                        to="songs.song",
                    ),
                ),
                (
                    "tag",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="texts",
                        to="songs.tag",
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="song",
            name="tags",
            field=models.ManyToManyField(
                blank=True, related_name="songs", to="songs.tag"
            ),
        ),
        migrations.AddField(
            model_name="song",
            name="writers",
            field=models.ManyToManyField(
                blank=True, related_name="songs", to="songs.writer"
            ),
        ),
        migrations.CreateModel(
            name="Attachment",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("object_id", models.PositiveIntegerField()),
                ("attachment_type", models.SlugField()),
                (
                    "file",
                    models.FileField(
                        blank=True,
                        null=True,
                        upload_to=songs.models.attachment_upload_path,
                    ),
                ),
                (
                    "content_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="contenttypes.contenttype",
                    ),
                ),
            ],
            options={
                "unique_together": {
                    ("content_type", "object_id", "attachment_type", "file")
                },
            },
            managers=[
                ("objects", songs.managers.AttachmentManager()),
            ],
        ),
    ]