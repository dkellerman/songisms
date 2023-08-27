import re
from django.contrib.contenttypes.models import ContentTypeManager
from django.db import models, connection
from django.db.models import Q, F
from django.core.cache import cache
from django.conf import settings
from django_pandas.managers import DataFrameManager
from .utils.text import make_variants, get_stresses, get_vowel_vector, normalize_lyric


class BaseManager(DataFrameManager):
    pass


class RhymeManager(BaseManager):
    HARD_LIMIT = 100
    USE_SUGGESTIONS = True


    def top_rhymes(self, offset=0, limit=100):
        offset = offset or 0
        cache_key = f'top_rhymes_{offset}_{offset + limit}'
        qs = cache.get(cache_key) if settings.USE_QUERY_CACHE else None

        if not qs:
            from api.models import NGram
            qs = NGram.objects.annotate(
                frequency=models.Count('rhymed_from__song__id', distinct=True),
                ngram=models.F('text'),
                type=models.Value('rhyme'),
            ).filter(frequency__gt=0) \
             .filter(rhymed_from__level=1) \
             .order_by('-frequency', 'text') \
             .values('ngram', 'frequency', 'type')

            qs = qs[offset:min(self.HARD_LIMIT, offset+limit)]
            if settings.USE_QUERY_CACHE:
                cache.set(cache_key, qs)

        return qs


    def query(self, q, offset=0, limit=100):
        if not q:
            return self.top_rhymes(offset, limit)

        offset = offset or 0
        qkey = re.sub(' ', '_', q)
        cache_key = f'query_{qkey}'

        if settings.USE_QUERY_CACHE:
            vals = cache.get(cache_key)
            if vals:
                return vals[offset:min(self.HARD_LIMIT, offset+limit)]

        q = normalize_lyric(q)
        n = len(q.split())
        variants = make_variants(q)
        vec = get_vowel_vector(q) or None
        stresses = get_stresses(q)
        all_q = [q.upper()] + [s.upper() for s in variants]

        rhymes_sql = f'''
            SELECT DISTINCT ON (ngram)
                rto.text AS ngram,
                rto.n AS n,
                r.level AS level,
                COUNT(r.song_id) AS frequency,
                0 AS vec_distance,
                {f'CUBE(%(stresses)s) <-> CUBE(n.stresses) AS stresses_distance' if len(stresses)
                 else '0 AS stresses_distance'},
                n.adj_pct AS adj_pct,
                n.song_pct AS song_pct,
                n.title_pct AS title_pct,
                ABS(rto.n - %(n)s) AS ndiff,
                0 AS mscore,
                n.song_count AS song_count
            FROM
                api_ngram n
            INNER JOIN
                api_rhyme r ON r.from_ngram_id = n.id
            INNER JOIN
                api_ngram rfrom ON rfrom.id = r.from_ngram_id
            INNER JOIN
                api_ngram rto ON rto.id = r.to_ngram_id
            WHERE
                UPPER(n.text) = ANY(%(q)s)
                AND NOT (UPPER(rto.text) = ANY(%(q)s))
            GROUP BY ngram, rto.n, level, vec_distance, stresses_distance,
                     n.adj_pct, n.song_pct, n.title_pct, ndiff, n.mscore, n.song_count
        '''

        suggestions_sql = f'''
            SELECT
                n.text AS ngram,
                n.n AS n,
                CAST(NULL AS bigint) AS level,
                CAST(NULL AS bigint) AS frequency,
                CUBE(%(vec)s) <-> CUBE(n.phones) AS vec_distance,
                CUBE(%(stresses)s) <-> CUBE(n.stresses) AS stresses_distance,
                n.adj_pct AS adj_pct,
                n.song_pct AS song_pct,
                n.title_pct AS title_pct,
                ABS(n - %(n)s) AS ndiff,
                n.mscore AS mscore,
                n.song_count AS song_count
            FROM
                api_ngram n
            WHERE
                NOT (UPPER(n.text) = ANY(%(q)s))
                AND n.phones IS NOT NULL
                AND CUBE(%(vec)s) <-> CUBE(n.phones) <= 2.5
                AND adj_pct >= 0.00005
                AND n.mscore > 4
                AND n.song_count > 2
            GROUP BY ngram, n, level, frequency, vec_distance, stresses_distance,
                     adj_pct, song_pct, title_pct, ndiff, mscore, song_count
        ''' if self.USE_SUGGESTIONS and vec and len(vec) and stresses and len(stresses) else ''

        with connection.cursor() as cursor:
            cursor.execute(f'''
                WITH results AS (
                    {rhymes_sql}
                    {f'UNION ALL {suggestions_sql}' if suggestions_sql else ''}
                )
                    SELECT
                        ngram,
                        frequency,
                        CASE
                            WHEN level = 1 THEN 'rhyme'
                            WHEN level = 2 THEN 'rhyme-l2'
                            ELSE 'suggestion'
                        END AS type
                    FROM (SELECT DISTINCT ON (ngram) * FROM results ORDER BY ngram, level) uniq_results
                    ORDER BY
                        level NULLS LAST,
                        frequency DESC NULLS LAST,
                        vec_distance,
                        stresses_distance,
                        mscore DESC NULLS LAST,
                        ndiff - (adj_pct * 10000),
                        adj_pct DESC NULLS LAST,
                        title_pct DESC NULLS LAST,
                        song_pct DESC NULLS LAST
                    OFFSET 0
                    LIMIT %(limit)s
                ;
            ''', dict(q=all_q, qstr=q, n=n, vec=vec, stresses=stresses, limit=self.HARD_LIMIT))

            columns = [col[0] for col in cursor.description]
            vals = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
            if settings.USE_QUERY_CACHE:
                cache.set(cache_key, vals)
            vals = vals[offset:min(self.HARD_LIMIT, offset+limit)]

            return vals


class LineManager(BaseManager):
    def get_by_natural_key(self, text):
        return self.get(text=text)


class NGramManager(BaseManager):
    def get_by_natural_key(self, text):
        return self.get(text=text)

    def completions(self, q, ct=20):
        if not q:
            return []

        qkey = re.sub(' ', '_', q)
        cache_key = f'completion_{qkey}'
        qs = cache.get(cache_key) if settings.USE_QUERY_CACHE else None
        spaces = sum([1 for c in q if c == ' '])

        if not qs:
            from api.models import NGram
            qs = NGram.objects.filter(text__istartswith=q)
            qs = qs.annotate(rhyme_ct=models.Count('rhymes')).filter(rhyme_ct__gt=0)
            if spaces == 0:
                qs = qs.filter(n=1)
            else:
                qs = qs.filter(n__gt=spaces)
            qs = qs.order_by('-rhyme_ct')[:ct]
            cache.set(cache_key, qs)
        return qs

    def by_query(self, q):
        from api.models import NGram
        qs = NGram.objects.filter(text__istartswith=q).order_by(F('song_count').desc(nulls_last=True))
        return qs

    def top(self):
        from api.models import NGram
        return NGram.objects.all().order_by(F('song_count').desc(nulls_last=True))


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

    def with_words(self, *words, variants=True, or_=True, pct=False, title_only=False):
        words = set(words)
        if variants:
            for word in list(words):
                for syn in make_variants(word):
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
        return self.annotate(popularity=models.F('metadata__spotify__track__popularity')).order_by('-popularity')


class ArtistManager(BaseManager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class WriterManager(BaseManager):
    def query(self, q=None, ordering=None):
        writers = self.prefetch_related('songs').annotate(
            song_ct=models.Count('songs', distinct=True),
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
