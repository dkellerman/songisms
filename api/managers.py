import re
from django.contrib.contenttypes.models import ContentTypeManager
from django.db import models, connection
from django.db.models import Q
from django.core.cache import cache
from django.conf import settings
from django_pandas.managers import DataFrameManager
from .utils.text import make_synonyms, get_vowel_vectors, tokenize_lyric_line, get_stresses


class BaseManager(DataFrameManager):
    pass


class RhymeManager(BaseManager):
    HARD_LIMIT = 200

    def top_rhymes(self, offset=0, limit=50):
        offset = offset or 0
        limit = limit or 50
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

    def query(self, q, offset=0, limit=50):
        if not q:
            return self.top_rhymes(offset, limit)

        offset = offset or 0
        limit = limit or 50
        qkey = re.sub(' ', '_', q)
        cache_key = f'query_{qkey}'

        if settings.USE_QUERY_CACHE:
            vals = cache.get(cache_key)
            if vals:
                return vals[offset:min(self.HARD_LIMIT, offset+limit)]

        q = ' '.join(tokenize_lyric_line(q))
        syns = make_synonyms(q)
        qphones = get_vowel_vectors(q, try_syns=tuple(syns), pad_to=10) or None
        qstresses = get_stresses(q)
        qn = len(q.split())
        all_q = [q.upper()] + [s.upper() for s in syns]

        rhymes_sql = f'''
            SELECT DISTINCT ON (ngram)
                rto.text AS ngram,
                rto.n AS n,
                r.level AS level,
                COUNT(r.song_id) AS frequency,
                0 AS phones_distance,
                CUBE(%(qstresses)s) <-> CUBE(n.stresses) AS stresses_distance,
                n.adj_pct AS adj_pct,
                0 AS ndiff,
                0 AS mscore,
                0 AS song_ct
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
            GROUP BY ngram, rto.n, level, phones_distance, stresses_distance, n.adj_pct, ndiff, n.mscore, song_ct
        '''

        suggestions_sql = f'''
            SELECT
                n.text AS ngram,
                n.n AS n,
                CAST(NULL AS bigint) AS level,
                CAST(NULL AS bigint) AS frequency,
                CUBE(%(qphones)s) <-> CUBE(n.phones) AS phones_distance,
                CUBE(%(qstresses)s) <-> CUBE(n.stresses) AS stresses_distance,
                n.adj_pct AS adj_pct,
                ABS(n - %(qn)s) AS ndiff,
                n.mscore AS mscore,
                COUNT(sn.song_id) AS song_ct
            FROM
                api_ngram n
            FULL OUTER JOIN
                api_songngram sn ON sn.ngram_id = n.id
            WHERE
                NOT (UPPER(n.text) = ANY(%(q)s))
                AND n.phones IS NOT NULL
                AND CUBE(%(qphones)s) <-> CUBE(n.phones) <= 2.5
                AND adj_pct >= 0.00005
                AND n.mscore > 4
            GROUP BY ngram, n, level, frequency, phones_distance, stresses_distance, adj_pct, ndiff, mscore
            HAVING COUNT(sn.song_id) > 2
        ''' if qphones and len(qphones) else ''

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
                        phones_distance,
                        stresses_distance,
                        mscore DESC NULLS LAST,
                        ndiff - (adj_pct * 10000),
                        adj_pct DESC NULLS LAST
                    OFFSET 0
                    LIMIT %(limit)s
                ;
            ''', dict(q=all_q, qstr=q, qn=qn, qphones=qphones, qstresses=qstresses, limit=self.HARD_LIMIT))

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

    def suggest(self, q, ct=20):
        if not q:
            return []

        qkey = re.sub(' ', '_', q)
        cache_key = f'suggest_{qkey}'
        qs = cache.get(cache_key) if settings.USE_QUERY_CACHE else None
        if not qs:
            from api.models import NGram
            qs = NGram.objects.filter(text__istartswith=q)
            qs = qs.annotate(rhyme_ct=models.Count('rhymes')).filter(rhyme_ct__gt=0)
            qs = qs.order_by('-rhyme_ct')[:ct]
            cache.set(cache_key, qs)
        return qs


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
        songs = self.prefetch_related('artists', 'tags')

        if q:
            includes = {}
            excludes = {}
            matches = re.findall(r'(~?[^\s]+:)?([^\s]+)', q.lower())
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
                else:
                    include, exclude = self.filter_map.get(field, (None, None))
                    if include:
                        includes[include] = qstr
                    if exclude:
                        excludes[exclude] = qstr
            if reverse:
                includes, excludes = excludes, includes
            songs = songs.filter(**includes).exclude(**excludes)
        return songs

    def with_words(self, *words, syns=True, or_=True, pct=False):
        words = set(words)
        if syns:
            for word in list(words):
                for syn in make_synonyms(word):
                    words.add(syn)

        qs = self.exclude(is_new=True)
        total = qs.count() if pct else 0

        q = Q()
        for word in list(words):
            cond = Q(lyrics__iregex=r"\y{0}\y".format(word))
            q = q | cond if or_ else q & cond
        qs = qs.filter(q)
        if pct:
            return qs.count() / total
        return qs


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
    pass
