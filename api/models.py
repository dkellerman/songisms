import reversion
from django.db import transaction
from django.db.models import Count
from django.contrib.postgres.fields import ArrayField
from django.utils.safestring import mark_safe
from .cloud_utils import get_storage_blob
from .managers import *


@reversion.register()
class Artist(models.Model):
    name = models.CharField(max_length=300, unique=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = ArtistManager()

    def __str__(self):
        return f'{self.name}'

    def natural_key(self):
        return (self.name,)


@reversion.register()
class Writer(models.Model):
    name = models.CharField(max_length=300, unique=True)
    alt_names = ArrayField(models.CharField(max_length=300, unique=True), default=list, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = WriterManager()

    def __str__(self):
        return f'{self.name}'

    def natural_key(self):
        return (self.name,)


@reversion.register()
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
        return f'{self.label} [{self.category}]'

    def natural_key(self):
        return (self.category, self.value)

    class Meta:
        unique_together = [('category', 'value')]


@reversion.register()
class TaggedText(models.Model):
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE, related_name='texts')
    text = models.TextField(db_index=True)
    song = models.ForeignKey('Song', null=True, blank=True, related_name='tagged_texts', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)

    def __str__(self):
        return f'{self.song.title} [{self.tag.label}] (len={len(self.text)})'


@reversion.register()
class NGram(models.Model):
    text = models.CharField(max_length=500, unique=True)
    n = models.PositiveIntegerField(db_index=True)
    rhymes = models.ManyToManyField('self', through='Rhyme')
    stresses = models.CharField(max_length=50, blank=True, null=True)
    ipa = models.CharField(max_length=500, blank=True, null=True)
    phones = models.CharField(max_length=500, blank=True, null=True)
    formants = ArrayField(ArrayField(models.IntegerField(), size=4), blank=True, null=True)
    pct = models.FloatField(blank=True, null=True)
    adj_pct = models.FloatField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = NGramManager()

    class Meta:
        verbose_name = 'NGram'
        verbose_name_plural = 'NGrams'

    def __str__(self):
        return f'{self.text}'

    def natural_key(self):
        return (self.text,)


@reversion.register()
class Rhyme(models.Model):
    from_ngram = models.ForeignKey(NGram, on_delete=models.CASCADE, related_name='rhymed_from')
    to_ngram = models.ForeignKey(NGram, on_delete=models.CASCADE, related_name='rhymed_to')
    song = models.ForeignKey('Song', on_delete=models.CASCADE, related_name='rhymes')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = RhymeManager()

    class Meta:
        unique_together = [['from_ngram', 'to_ngram', 'song']]

    def __str__(self):
        return f'{self.from_ngram.text} => {self.to_ngram.text} [{self.song.title}]'


@reversion.register()
class SongNGram(models.Model):
    ngram = models.ForeignKey('NGram', on_delete=models.CASCADE, related_name='song_ngrams')
    song = models.ForeignKey('Song', on_delete=models.CASCADE, related_name='song_ngrams')
    count = models.PositiveIntegerField()
    objects = RhymeManager()

    class Meta:
        unique_together = [['ngram', 'song']]

    def __str__(self):
        return f'{self.ngram.text} [{self.count}x IN {self.song.title}]'


@reversion.register()
class Song(models.Model):
    title = models.CharField(max_length=300, db_index=True)
    artists = models.ManyToManyField(Artist, related_name='songs', blank=True)
    writers = models.ManyToManyField(Writer, related_name='songs', blank=True)
    tags = models.ManyToManyField(Tag, related_name='songs', blank=True)
    lyrics = models.TextField(blank=True, null=True)
    lyrics_raw = models.TextField(blank=True, null=True)
    lyrics_ipa = models.TextField(blank=True, null=True)
    ngrams = models.ManyToManyField(NGram, through='SongNGram', blank=True, related_name='songs')
    rhymes_raw = models.TextField(blank=True, null=True)
    spotify_id = models.SlugField()
    jaxsta_id = models.SlugField(blank=True, null=True)
    youtube_id = models.SlugField(blank=True, null=True, unique=True)
    audio_file = models.FileField(upload_to='data/audio', blank=True, null=True)
    metadata = models.JSONField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = SongManager()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f'{self.title}'

    def natural_key(self):
        return (self.spotify_id,)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    @property
    def audio_file_path(self):
        return f'data/audio/{self.spotify_id}.webm'

    def audio_blob(self):
        return get_storage_blob(self.audio_file_path)

    def audio_file_exists(self):
        return self.audio_blob().exists()

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
    def audio_file_url(self):
        return self.audio_file.url if self.audio_file else None

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
            names = names.split(';')

        with transaction.atomic():
            new_artists = [
                Artist.objects.get_or_create(name=name.strip())[0]
                for name in (names or [])
            ]
            self.artists.set(new_artists)

    def set_writers(self, names=[]):
        if type(names) == str:
            names = names.split(';')

        with transaction.atomic():
            new_writers = [
                Writer.objects.get_or_create(name=name.strip())[0]
                for name in (names or [])
            ]
            self.writers.set(new_writers)

    def set_rhymes(self, rhymes=[]):
        if type(rhymes) == str:
            from .nlp_utils import get_rhyme_pairs
            rhymes = get_rhyme_pairs(rhymes)

        with transaction.atomic():
            for text1, text2 in rhymes:
                n1 = len(text1.split())
                n2 = len(text2.split())
                ngram1, _ = NGram.objects.get_or_create(text=text1, n=n1)
                ngram2, _ = NGram.objects.get_or_create(text=text2, n=n2)
                Rhyme.objects.get_or_create(
                    from_ngram=ngram1,
                    to_ngram=ngram2,
                    song=self,
                )
                Rhyme.objects.get_or_create(
                    from_ngram=ngram2,
                    to_ngram=ngram1,
                    song=self,
                )

    def set_song_tags(self, tags=[]):
        if type(tags) == str:
            tags = tags.split(';')

        with transaction.atomic():
            self.tags.set([
                Tag.objects.get_or_create(
                    value=tag,
                    defaults=dict(label=tag, category='song'
                                  ))[0]
                for tag in (tags or [])
            ])

    def set_snippet_tags(self, tags={}):  # {tag: {'content': text}}
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

    ngrams = NGram.objects.annotate(song_ct=Count('song_ngrams'), rhyme_ct=Count('rhymes')) \
        .filter(song_ct=0, rhyme_ct=0)
    print("[PRUNE NGRAMS]", ngrams.count())
    ngrams.delete()
