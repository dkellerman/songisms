import re
from django.contrib.contenttypes.models import ContentTypeManager
from django.db.models import Q, F, Count
from django_pandas.managers import DataFrameManager
from songisms import utils


class BaseManager(DataFrameManager):
    pass


class SongManager(BaseManager):
    filter_map = {
        'lyrics': ('lyrics__icontains', None),
        'rhymes': ('rhymes_raw__icontains', None),
        'writer': ('writers__name__icontains', None),
        'artist': ('artists__name__icontains', None),
        'tag': ('tags__value__iexact', None),
        'title': ('title__icontains', None),
        'is_new': ('is_new', None),
    }

    def get_by_natural_key(self, spotify_id):
        return self.get(spotify_id=spotify_id)

    def query(self, q=None):
        songs = self.by_popularity().prefetch_related('artists', 'tags')

        if q:
            includes = {}
            excludes = {}
            matches = re.findall(r'(~?[^\s]+:)?([^\s]+)', q.lower())
            order_by = None
            for field, qstr in matches:
                qstr = re.sub(r'\+', ' ', qstr)
                field = (field or 'title:')[:-1]
                reverse = field[0] == '~'
                if reverse:
                    field = field[1:]
                if field == 'is':
                    field = f'is_{qstr}'
                    qstr = True
                if field == 'has':
                    excludes[qstr] = None
                elif field == 'sort':
                    order_by = qstr
                else:
                    include, exclude = self.filter_map.get(field, (None, None))
                    if include:
                        includes[include] = qstr
                    if exclude:
                        excludes[exclude] = qstr
                if reverse:
                    includes, excludes = excludes, includes
                songs = songs.filter(**includes).exclude(**excludes)
            if order_by:
                songs = songs.order_by(order_by)

        return songs


    def with_words(self, *words, variants=False, or_=True, pct=False, title_only=False):
        words = set(words)
        if variants:
            for word in list(words):
                for syn in utils.make_variants(word):
                    words.add(syn)

        qs = self.exclude(is_new=True)
        total = qs.count() if pct else 0

        q = Q()
        for word in list(words):
            cond = Q(title__iregex=r"\y{0}\y".format(word))
            if not title_only:
                cond = cond | Q(lyrics__iregex=r"\y{0}\y".format(word))
            q = q | cond if or_ else q & cond
        qs = qs.filter(q)
        if pct:
            return qs.count() / total
        return qs


    def by_popularity(self):
        return self.annotate(popularity=F('metadata__spotify__track__popularity')).order_by('-popularity')


class ArtistManager(BaseManager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class WriterManager(BaseManager):
    def query(self, q=None, ordering=None):
        writers = self.prefetch_related('songs').annotate(
            song_ct=Count('songs', distinct=True),
        )
        if q:
            writers = writers.filter(name__icontains=q)
            ordering = ordering or ['name']
        else:
            ordering = ordering or ['-song_ct']
        return writers.order_by(*ordering)

    def get_by_natural_key(self, name):
        return self.get(name=name)


class TagManager(BaseManager):
    def get_by_natural_key(self, category, value):
        return self.get(category=category, value=value)


class CacheManager(BaseManager):
    def get_by_natural_key(self, key, version):
        return self.get(key=key, version=version)


class AttachmentManager(ContentTypeManager):
    def get_by_natural_key(self, content_type_id, object_id, attachment_type, file_name):
        return self.get(content_type__id=content_type_id, object_id=object_id, attachment_type=attachment_type, file=file_name)
