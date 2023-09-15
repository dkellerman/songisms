from django.db import models
from .managers import NGramManager, RhymeManager, CacheManager, VoteManager


class NGram(models.Model):
    text = models.CharField(max_length=500, unique=True)
    n = models.PositiveIntegerField(db_index=True)
    rhymes = models.ManyToManyField('self', through='Rhyme')
    frequency = models.PositiveIntegerField(db_index=True)
    pct = models.FloatField(db_index=True)
    adj_pct = models.FloatField(db_index=True)
    song_pct = models.FloatField(db_index=True)
    title_pct = models.FloatField(db_index=True)
    mscore = models.FloatField(db_index=True)
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
    uid = models.SlugField(primary_key=True)
    from_ngram = models.ForeignKey(NGram, on_delete=models.CASCADE, related_name='rhymed_from')
    to_ngram = models.ForeignKey(NGram, on_delete=models.CASCADE, related_name='rhymed_to')
    frequency = models.PositiveIntegerField(db_index=True)
    song_ct = models.PositiveIntegerField(db_index=True)
    score = models.FloatField(db_index=True)
    uscore = models.FloatField(db_index=True)
    source = models.SlugField()
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = RhymeManager()

    class Meta:
        unique_together = [['from_ngram', 'to_ngram',]]

    def __str__(self):
        return self.uid


class Vote(models.Model):
    '''User feedback from RLHF or straight up/down voting
    '''
    LABEL_CHOICES = (
        # for RLHF
        ('alt1', 'Alt1'),
        ('alt2', 'Alt2'),
        ('both', 'Both'),
        ('neither', 'Neither'),
        ('flagged', 'Flagged'),
        # for straight voting
        ('good', 'Good'),
        ('bad', 'Bad'),
    )

    anchor = models.CharField(max_length=200)
    alt1 = models.CharField(max_length=200, blank=True, null=True)
    alt2 = models.CharField(max_length=200, blank=True, null=True)
    voter_uid = models.SlugField()
    label = models.CharField(max_length=20, choices=LABEL_CHOICES)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True, blank=True, null=True)
    objects = VoteManager()

    class Meta:
        unique_together = [['voter_uid', 'created']]

    def __str__(self):
        return f'{self.anchor} [{self.label}]]'

    def natural_key(self):
        return self.voter_uid, self.created


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
