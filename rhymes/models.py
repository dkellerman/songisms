from django.db import models
from django.contrib.postgres.fields import ArrayField
from .managers import NGramManager, RhymeManager, CacheManager


class NGram(models.Model):
    text = models.CharField(max_length=500, unique=True)
    n = models.PositiveIntegerField(db_index=True)
    rhymes = models.ManyToManyField('self', through='Rhyme')
    ipa = models.CharField(max_length=500, blank=True, null=True)
    phones = ArrayField(ArrayField(models.FloatField()), null=True, blank=True, db_index=True)
    stresses = ArrayField(models.IntegerField(), null=True, blank=True, db_index=True)
    mscore = models.FloatField(blank=True, null=True, db_index=True)
    count = models.PositiveIntegerField(blank=True, null=True, db_index=True)
    song_count = models.PositiveIntegerField(blank=True, null=True, db_index=True)
    pct = models.FloatField(blank=True, null=True, db_index=True)
    adj_pct = models.FloatField(blank=True, null=True, db_index=True)
    song_pct = models.FloatField(blank=True, null=True, db_index=True)
    title_pct = models.FloatField(blank=True, null=True, db_index=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = NGramManager()

    class Meta:
        verbose_name = 'NGram'
        verbose_name_plural = 'NGrams'

    def __str__(self):
        return f'{self.text}'

    def natural_key(self):
        return self.text,


class Rhyme(models.Model):
    from_ngram = models.ForeignKey(NGram, on_delete=models.CASCADE, related_name='rhymed_from')
    to_ngram = models.ForeignKey(NGram, on_delete=models.CASCADE, related_name='rhymed_to')
    song_uid = models.SlugField(blank=True, null=True)
    level = models.PositiveIntegerField(default=1, db_index=True)
    score = models.FloatField(blank=True, null=True, db_index=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = RhymeManager()

    class Meta:
        unique_together = [['from_ngram', 'to_ngram', 'song_uid']]

    def __str__(self):
        return f'{self.from_ngram.text} => {self.to_ngram.text} [L{self.level}]'


class SongNGram(models.Model):
    ngram = models.ForeignKey('NGram', on_delete=models.CASCADE, related_name='song_ngrams')
    song_uid = models.SlugField()
    count = models.PositiveIntegerField(db_index=True)
    objects = RhymeManager()

    class Meta:
        unique_together = [['ngram', 'song_uid']]

    def __str__(self):
        return f'{self.ngram.text} [{self.count}x IN {self.song.title}]'


class Cache(models.Model):
    key = models.CharField(max_length=500)
    version = models.PositiveIntegerField(default=1)
    data = models.JSONField(blank=True, null=True)
    file = models.FileField(upload_to='data/cache', blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = CacheManager()

    class Meta:
        unique_together = [('key', 'version',)]

    def __str__(self):
        return f'{self.key} [v{self.version}]'

    def natural_key(self):
        return self.key, self.version

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def get(self, key, getter=None, save=False):
        if key in (self.data or {}):
            return self.data[key]
        self.data = self.data or {}
        if getter:
            self.data[key] = getter(key)
            if save:
                self.save()
            return self.data[key]
        else:
            return None

    def clear(self, save=True):
        self.data = None
        if save:
            self.save()
