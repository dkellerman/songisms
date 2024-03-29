from django.db import transaction
from django.db import models
from django.db.models import Count, Q
from django.contrib.postgres.fields import ArrayField
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.utils.safestring import mark_safe
from songisms import utils
from songs.managers import ArtistManager, WriterManager, TagManager, AttachmentManager, SongManager


class Artist(models.Model):
    name = models.CharField(max_length=300, unique=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = ArtistManager()

    def __str__(self):
        return f'{self.name}'

    def natural_key(self):
        return self.name,


class Writer(models.Model):
    name = models.CharField(max_length=300, unique=True)
    alt_names = ArrayField(models.CharField(max_length=300, unique=True), default=list, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = WriterManager()

    def __str__(self):
        return f'{self.name}'

    def natural_key(self):
        return self.name,


class Tag(models.Model):
    TAG_CATEGORY_CHOICES = (
        ('song', 'Song'),
        ('snippet', 'Snippet'),
    )
    value = models.SlugField()
    label = models.CharField(max_length=100)
    category = models.CharField(max_length=100, choices=TAG_CATEGORY_CHOICES, default='song')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = TagManager()

    def __str__(self):
        return f'{self.label}'

    def natural_key(self):
        return self.category, self.value

    class Meta:
        unique_together = [('category', 'value')]


class TaggedText(models.Model):
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE, related_name='texts')
    text = models.TextField(db_index=True)
    song = models.ForeignKey('Song', null=True, blank=True, related_name='tagged_texts', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)

    def __str__(self):
        return f'{self.song.title} [{self.tag.label}] (len={len(self.text)})'


def attachment_upload_path(instance, filename):
    if type(instance.content_object) == Song:
        key = instance.content_object.spotify_id
    else:
        key = str(instance.object_id)
    return f'data/{instance.content_type.model.lower()}/{key}/{instance.attachment_type}'


class Attachment(models.Model):
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')
    attachment_type = models.SlugField()
    file = models.FileField(upload_to=attachment_upload_path, blank=True, null=True)
    objects = AttachmentManager()

    class Meta:
        unique_together = [
            ['content_type', 'object_id', 'attachment_type', 'file']]

    def __str__(self):
        return f'{self.content_object} [{self.attachment_type}]'

    def natural_key(self):
        return self.content_type.id, self.object_id, self.attachment_type, self.file.name


class Song(models.Model):
    is_new = models.BooleanField(default=True)
    title = models.CharField(max_length=300, db_index=True)
    artists = models.ManyToManyField(Artist, related_name='songs', blank=True)
    writers = models.ManyToManyField(Writer, related_name='songs', blank=True)
    tags = models.ManyToManyField(Tag, related_name='songs', blank=True)
    lyrics = models.TextField(blank=True, null=True)
    rhymes_raw = models.TextField(blank=True, null=True)
    spotify_id = models.SlugField()
    jaxsta_id = models.SlugField(blank=True, null=True)
    youtube_id = models.SlugField(blank=True, null=True, unique=True)
    audio_file = models.FileField(upload_to='data/audio', blank=True, null=True)
    metadata = models.JSONField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    attachments = GenericRelation(Attachment)
    objects = SongManager()

    def __str__(self):
        return f'{self.title}'

    def natural_key(self):
        return self.spotify_id,

    def clean(self):
        self.lyrics = self.lyrics.replace('\r\n', '\n')
        self.rhymes_raw = self.rhymes_raw.replace('\r\n', '\n')

    @property
    def audio_file_url(self):
        return self.audio_file.url if self.audio_file else None

    def audio_blob(self):
        return utils.get_storage_blob(self.audio_file.name)

    def audio_file_exists(self):
        blob = self.audio_blob()
        return blob and blob.exists()

    def has_attachment(self, t):
        return self.attachments.filter(attachment_type=t).exists()

    def get_attachment(self, t):
        return self.attachments.filter(attachment_type=t).first()

    @property
    def jaxsta_url(self):
        return f'https://jaxsta.com/recording/{self.jaxsta_id}' if self.jaxsta_id else None

    @property
    def youtube_url(self):
        return f'https://youtube.com/watch?v={self.youtube_id}' if self.youtube_id else None

    @property
    def spotify_url(self):
        return f'https://open.spotify.com/track/{self.spotify_id}' if self.spotify_id else None

    @property
    def spotify_player(self):
        return mark_safe(f'''<iframe
            className="spotify"
            src="https://open.spotify.com/embed/track/{self.spotify_id}"
            height="80"
            frameBorder="0"
            allow="clipboard-write; encrypted-media; fullscreen; picture-in-picture"
        ></iframe>''') if self.spotify_id else None

    @property
    def youtube_player(self):
        return mark_safe(f'''
            <iframe
                width="560"
                height="315"
                src="https://www.youtube.com/embed/{self.youtube_id}"
                title="{self.title} - YouTube video player"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media;
                       gyroscope; picture-in-picture"
                allowfullscreen></iframe>
        ''') if self.youtube_id else None

    def set_artists(self, names=[]):
        if type(names) == str:
            names = [n.strip() for n in names.split(';')]
        names = [n.strip() for n in names]

        with transaction.atomic():
            new_artists = [
                Artist.objects.get_or_create(name=name)[0]
                for name in names
            ]
            self.artists.set(new_artists)

    def set_writers(self, names=[]):
        if type(names) == str:
            names = names.split(';')
        names = [n.strip() for n in names]

        with transaction.atomic():
            new_writers = [
                Writer.objects.get_or_create(
                    Q(name=name) | Q(alt_names__contains=name))[0]
                for name in names
            ]
            self.writers.set(new_writers)

    def set_song_tags(self, tags=[]):
        if type(tags) == str:
            tags = tags.split(';')
        tags = [t.strip() for t in tags]

        with transaction.atomic():
            self.tags.set([
                Tag.objects.get_or_create(
                    value=tag,
                    defaults=dict(label=tag, category='song')
                )[0]
                for tag in tags
            ])

    def set_snippet_tags(self, tags={}):  # {tag: {'content': 'text'}}
        tagged_texts = set()
        for tag, values in (tags or {}).items():
            for value in values:
                tagged_texts.add((tag, value['content'],))

        self.tagged_texts.set([
            TaggedText.objects.get_or_create(
                text=text,
                tag=Tag.objects.get_or_create(
                    value=tag,
                    category='snippet',
                    defaults=dict(label=tag)
                )[0]
            )[0] for tag, text in tagged_texts
        ])


def prune():
    artists = Artist.objects.annotate(song_ct=Count('songs')).filter(song_ct=0)
    for a in artists:
        print("[PRUNE ARTIST]", a.pk, a.name)
        a.delete()

    writers = Writer.objects.annotate(song_ct=Count('songs')).filter(song_ct=0)
    for w in writers:
        print("[PRUNE WRITER]", w.pk, w.name)
        w.delete()
