import re
from django.db import models, connection
from django.db.models import F
from django.core.cache import cache
from django.conf import settings
from songisms import utils


class BaseManager(models.Manager):
    pass


class RhymeManager(BaseManager):
    HARD_LIMIT = 100
    USE_SUGGESTIONS = True

    def top_rhymes(self, offset=0, limit=100, q=None):
        offset = offset or 0
        cache_key = f'top_rhymes_{offset}_{offset + limit}'

        if not q and settings.USE_QUERY_CACHE:
            qs = cache.get(cache_key) if settings.USE_QUERY_CACHE else None
            if qs:
                return qs

        from rhymes.models import NGram
        qs = NGram.objects.all()
        if q:
            starts_with, ends_with = q.split('*')
            if starts_with:
                qs = qs.filter(text__istartswith=starts_with)
            if ends_with:
                qs = qs.filter(text__iendswith=ends_with)

        qs = qs.annotate(
            frequency=models.Count('rhymed_from__song__id', distinct=True),
            ngram=models.F('text'),
            type=models.Value('rhyme'),
        )

        qs = qs.filter(frequency__gt=0) \
            .filter(rhymed_from__level=1) \
            .order_by('-frequency', 'text') \
            .values('ngram', 'frequency', 'type')

        qs = qs[offset:min(self.HARD_LIMIT, offset+limit)]
        if not q and settings.USE_QUERY_CACHE:
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

        q = utils.normalize_lyric(q)
        n = len(q.split())
        variants = utils.make_variants(q)
        vec = utils.get_vowel_vector(q) or None
        stresses = utils.get_stresses_vector(q)
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
                rhymes_ngram n
            INNER JOIN
                rhymes_rhyme r ON r.from_ngram_id = n.id
            INNER JOIN
                rhymes_ngram rfrom ON rfrom.id = r.from_ngram_id
            INNER JOIN
                rhymes_ngram rto ON rto.id = r.to_ngram_id
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
                rhymes_ngram n
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


class NGramManager(BaseManager):
    def get_by_natural_key(self, text):
        return self.get(text=text)

    def completions(self, q, ct=20):
        if not q:
            return []

        qkey = re.sub(' ', '_', q)
        cache_key = f'completion_{qkey}'
        qs = cache.get(cache_key) if settings.USE_QUERY_CACHE else None

        if not qs:
            from rhymes.models import NGram
            qs = NGram.objects.filter(text__istartswith=q)
            qs = qs.annotate(rhyme_ct=models.Count('rhymes')).filter(rhyme_ct__gt=0)
            qs = qs.order_by('n', '-rhyme_ct')[:ct]
            if settings.USE_QUERY_CACHE:
                cache.set(cache_key, qs)
        return qs

    def by_query(self, q):
        from rhymes.models import NGram
        qs = NGram.objects.filter(text__istartswith=q).order_by(F('song_count').desc(nulls_last=True))
        return qs

    def top(self):
        from rhymes.models import NGram
        return NGram.objects.all().order_by(F('song_count').desc(nulls_last=True))


class CacheManager(BaseManager):
    def get_by_natural_key(self, key, version):
        return self.get(key=key, version=version)
