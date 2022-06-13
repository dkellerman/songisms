import re
from django.db import models, connection
from django.core.cache import cache
from .nlp_utils import make_synonyms, get_phones, tokenize_lyric_line


class RhymeManager(models.Manager):
    HARD_LIMIT = 200

    def top_rhymes(self, offset=0, limit=50):
        offset = offset or 0
        limit = limit or 50
        cache_key = f'top_rhymes_{offset}_{offset+limit}'
        qs = cache.get(cache_key)
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
            cache.set(cache_key, qs)

        return qs

    def query(self, q, offset=0, limit=50):
        if not q:
            return self.top_rhymes(offset, limit)

        offset = offset or 0
        limit = limit or 50
        qkey = re.sub(' ', '_', q)
        cache_key = f'query_{qkey}'
        vals = cache.get(cache_key)
        if vals:
            return vals[offset:min(self.HARD_LIMIT, offset+limit)]

        q = ' '.join(tokenize_lyric_line(q))
        syns = make_synonyms(q)
        qphones = get_phones(q, vowels_only=True, include_stresses=False, try_syns=syns)
        qn = len(q.split())
        all_q = [q.upper()] + [s.upper() for s in syns]

        rhymes_sql = f'''
            SELECT DISTINCT ON (ngram)
                rto.text AS ngram,
                rto.n AS n,
                r.level AS level,
                COUNT(r.song_id) AS frequency,
                0 AS distance,
                n.adj_pct AS adj_pct
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
            GROUP BY
                ngram,
                rto.n,
                level,
                distance,
                n.adj_pct
        '''

        suggestions_sql = f'''
            SELECT
                n.text AS ngram,
                n.n AS n,
                NULL AS level,
                NULL AS frequency,
                LEVENSHTEIN(n.phones, %(qphones)s) AS distance,
                n.adj_pct AS adj_pct
            FROM
                api_ngram n
            WHERE
                NOT (UPPER(n.text) = ANY(%(q)s))
                AND LEVENSHTEIN(n.phones, %(qphones)s) <= 3
                AND adj_pct >= 0.0005
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
                        END AS type,
                        ABS(n - 1) AS ndiff
                    FROM (SELECT DISTINCT ON (ngram) * FROM results ORDER BY ngram, level) uniq_results
                    ORDER BY
                        level NULLS LAST,
                        frequency DESC NULLS LAST,
                        adj_pct DESC NULLS LAST,
                        distance,
                        ndiff
                    OFFSET 0
                    LIMIT %(limit)s
                ;
            ''', dict(q=all_q, qn=qn, qphones=qphones, limit=self.HARD_LIMIT))

            columns = [col[0] for col in cursor.description]
            vals = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
            cache.set(cache_key, vals)
            vals = vals[offset:min(self.HARD_LIMIT, offset+limit)]
            return vals


class NGramManager(models.Manager):
    def get_by_natural_key(self, text):
        return self.get(text=text)


class SongManager(models.Manager):
    def get_by_natural_key(self, spotify_id):
        return self.get(spotify_id=spotify_id)


class ArtistManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class WriterManager(models.Manager):
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


class TagManager(models.Manager):
    def get_by_natural_key(self, category, value):
        return self.get(category=category, value=value)
