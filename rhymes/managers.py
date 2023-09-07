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

    def query(self, q, offset, limit, voter_uid=None):
        '''Main rhymes query, handles rhymes and suggestions
        '''
        if not q:
            return []

        offset = offset
        qkey = re.sub(' ', '_', q)
        cache_key = f'query_{qkey}'
        use_cache = settings.USE_QUERY_CACHE and not voter_uid

        if use_cache:
            vals = cache.get(cache_key)
            if vals:
                return vals[offset:min(self.HARD_LIMIT, offset+limit)]

        q = utils.normalize_lyric(q)
        n = len(q.split())
        variants = utils.make_variants(q)
        vec = utils.get_vowel_vector(q) or None
        stresses = utils.get_stresses_vector(q)
        all_q = [q.upper()] + [v.upper() for v in variants]

        rhymes_sql = f'''
            SELECT DISTINCT ON (ngram)
                rto.text AS ngram,
                rto.n AS n,
                r.level AS level,
                r.score AS score,
                COUNT(r.song_uid) AS frequency,
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
                AND (r.level != 2 OR r.score >= 0.4)
            GROUP BY ngram, rto.n, level, score, vec_distance, stresses_distance,
                     n.adj_pct, n.song_pct, n.title_pct, ndiff, n.mscore, n.song_count
        '''

        suggestions_sql = f'''
            SELECT
                n.text AS ngram,
                n.n AS n,
                CAST(NULL AS bigint) AS level,
                CAST(NULL AS float) AS score,
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
                AND CUBE(%(vec)s) <-> CUBE(n.phones) >= 3.0
                AND CUBE(%(vec)s) <-> CUBE(n.phones) <= 3.2
                AND adj_pct >= 0.00012
                AND n.mscore >= 4
                AND n.song_count > 2
            GROUP BY ngram, n, level, frequency, vec_distance, stresses_distance,
                     adj_pct, song_pct, title_pct, ndiff, mscore, song_count
        ''' if self.USE_SUGGESTIONS and vec and len(vec) and stresses and len(stresses) else ''

        vote_sql = f'''
            SELECT
                label
            FROM
                rhymes_vote v
            WHERE (
                v.anchor = %(qstr)s AND v.alt1 = ngram
                OR (v.anchor = ngram AND v.alt1 = %(qstr)s)
            ) AND v.voter_uid = %(voter_uid)s
        ''' if voter_uid else ''

        with connection.cursor() as cursor:
            cursor.execute(f'''
                WITH results AS (
                    {rhymes_sql}
                    {f'UNION ALL {suggestions_sql}' if suggestions_sql else ''}
                )
                    SELECT
                        ngram,
                        frequency,
                        score,
                        {f'({vote_sql})' if vote_sql else 'NULL'} AS vote,
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
            ''', dict(qstr=q, q=all_q, n=n, vec=vec, stresses=stresses,
                      limit=self.HARD_LIMIT, voter_uid=voter_uid))

            columns = [col[0] for col in cursor.description]
            vals = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]

            if use_cache:
                cache.set(cache_key, vals)

            vals = vals[offset:min(self.HARD_LIMIT, offset+limit)]
            return vals


    def top_rhymes(self, offset, limit, q=None):
        '''Retrieve most-rhymed words
        '''
        offset = offset
        cache_key = f'top_rhymes_{offset}_{offset + limit}'
        use_cache = not q and settings.USE_QUERY_CACHE

        if use_cache:
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
            frequency=models.Count('rhymed_from__song_uid', distinct=True),
            ngram=models.F('text'),
            type=models.Value('rhyme'),
        )

        qs = qs.filter(frequency__gt=0) \
            .filter(rhymed_from__level=1) \
            .order_by('-frequency', 'text') \
            .values('ngram', 'frequency', 'type')

        qs = qs[offset:min(self.HARD_LIMIT, offset+limit)]

        if use_cache:
            cache.set(cache_key, qs)

        return qs


class NGramManager(BaseManager):
    def get_by_natural_key(self, text):
        return self.get(text=text)

    def completions(self, q, limit):
        '''Auto-suggest completions for a given query
        '''
        if not q:
            return []

        qkey = re.sub(' ', '_', q)
        cache_key = f'completion_{qkey}'
        use_cache = settings.USE_QUERY_CACHE
        qs = cache.get(cache_key) if use_cache else None

        if not qs:
            from rhymes.models import NGram
            qs = NGram.objects.filter(text__istartswith=q)
            qs = qs.annotate(rhyme_ct=models.Count('rhymes')).filter(rhyme_ct__gt=0)
            qs = qs.order_by('n', '-rhyme_ct')[:limit]

            if use_cache:
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


class VoteManager(BaseManager):
    def get_by_natural_key(self, voter_uid, created):
        return self.get(voter_uid=voter_uid, created=created)
