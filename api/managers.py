from django.db import models, connection
from .nlp_utils import make_synonyms, get_phones


class RhymeManager(models.Manager):
    def top_rhymes(self, limit=None, offset=0, n_min=None, n_max=None):
        from api.models import NGram
        n_filters = dict()
        if n_min:
            n_filters['n__gte'] = n_min
        if n_max:
            n_filters['n__lte'] = n_max
        qs = NGram.objects.annotate(
            frequency=models.Count('rhymed_from__song__id', distinct=True),
            ngram=models.F('text'),
            type=models.Value('rhyme'),
        ).filter(**n_filters) \
         .filter(frequency__gt=0) \
         .order_by('-frequency', 'text') \
         .values('ngram', 'frequency', 'type')

        if limit:
            qs = qs[offset:offset+limit]

        return qs

    def top_suggestions(self, limit=None, offset=0, n_min=None, n_max=None):
        from api.models import NGram
        n_filters = dict()
        if n_min:
            n_filters['n__gte'] = n_min
        if n_max:
            n_filters['n__lte'] = n_max
        qs = NGram.objects.annotate(
            frequency=models.Count('songs', distinct=True),
            ngram=models.F('text'),
            type=models.Value('suggestion'),
        ).filter(**n_filters) \
         .order_by(models.F('adj_pct').desc(nulls_last=True), '-frequency') \
         .values('ngram', 'frequency', 'type')

        if limit:
            qs = qs[offset:offset+limit]

        return qs

    def query(self, q, limit=None, offset=0, n_min=None, n_max=None):
        if not q:
            return self.top_rhymes(limit, offset)

        syns = make_synonyms(q)

        with connection.cursor() as cursor:
            cursor.execute(f'''
                WITH RECURSIVE rhymes AS (
                    SELECT
                        from_ngram.text AS from_ngram_text,
                        to_ngram.text AS to_ngram_text,
                        to_ngram.n as to_ngram_n,
                        count(to_ngram.id) AS frequency,
                        to_ngram.adj_pct AS adj_pct,
                        1 AS level
                    FROM api_rhyme r
                    INNER JOIN api_ngram from_ngram ON from_ngram.id = r.from_ngram_id
                    INNER JOIN api_ngram to_ngram ON to_ngram.id = r.to_ngram_id
                    WHERE UPPER(from_ngram.text) ILIKE ANY(%(q)s)
                    GROUP BY from_ngram.text, to_ngram.text, to_ngram.n, to_ngram.adj_pct, level

                    UNION ALL

                    SELECT
                        from_ngram.text AS from_ngram_text,
                        to_ngram.text AS to_ngram_text,
                        to_ngram.n AS to_ngram_n,
                        0 AS frequency,
                        to_ngram.adj_pct AS adj_pct,
                        rhymes.level + 1 AS level2
                    FROM rhymes, api_rhyme r2
                    INNER JOIN api_ngram from_ngram ON from_ngram.id = r2.from_ngram_id
                    INNER JOIN api_ngram to_ngram ON to_ngram.id = r2.to_ngram_id
                    WHERE
                        UPPER(from_ngram.text) = UPPER(rhymes.to_ngram_text)
                        AND NOT UPPER(to_ngram.text) ILIKE ANY(%(q)s)
                        AND to_ngram.text != rhymes.to_ngram_text
                        AND rhymes.level <= 1
                    GROUP BY from_ngram.text, to_ngram.text, to_ngram.n, to_ngram.adj_pct, rhymes.level
                )
                    SELECT * from (
                        SELECT
                            DISTINCT ON (to_ngram_text)
                            to_ngram_text AS ngram,
                            to_ngram_n AS n,
                            frequency,
                            adj_pct,
                            CASE
                              WHEN frequency = 0 THEN 'rhyme-l2'
                              ELSE 'rhyme'
                            END AS type
                        FROM rhymes
                        WHERE to_ngram_text NOT ILIKE ANY(%(q)s)
                        {f'AND n >= %(n_min)s' if n_min else ''}
                        {f'AND n <= %(n_max)s' if n_max else ''}
                        ORDER BY to_ngram_text, level
                    ) results
                        ORDER BY frequency DESC, adj_pct DESC NULLS LAST
                        -- {f'LIMIT %(limit)s OFFSET %(offset)s' if limit else ''}
                ;
                ''', dict(q=[q] + syns, limit=limit, offset=offset, n_min=n_min, n_max=n_max)
            )

            columns = [col[0] for col in cursor.description]
            vals = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
            if limit:
                vals = vals[offset:offset+limit]
            return vals

    def suggest(self, q, limit=None, offset=0, n_min=None, n_max=None):
        if not q:
            return self.top_suggestions(limit, offset, n_min, n_max)

        syns = make_synonyms(q)
        qphones = get_phones(q, vowels_only=True, include_stresses=False)
        qn = len(q.split())

        with connection.cursor() as cursor:
            cursor.execute(f'''
                WITH results AS (
                    SELECT
                        ngram.text AS ngram_text,
                        ngram.n AS n,
                        ngram.phones,
                        ngram.adj_pct AS adj_pct,
                        COUNT(sn.song_id) as song_ct,
                        LEVENSHTEIN(%(qphones)s, ngram.phones) AS phones_distance
                    FROM
                        api_ngram ngram
                    INNER JOIN
                        api_songngram sn ON sn.ngram_id = ngram.id
                    WHERE
                        NOT (ngram.text = ANY(%(q)s))
                        AND ngram.n <= 2
                        AND adj_pct > 0
                    GROUP BY
                        ngram.text, ngram.phones, ngram.adj_pct, ngram.n, phones_distance
                    HAVING
                        COUNT(sn.song_id) > 1
                        AND LEVENSHTEIN(%(qphones)s, ngram.phones) <= 4
                    ORDER BY ngram.text
                )
                    SELECT
                        ngram_text AS ngram,
                        n,
                        phones_distance,
                        adj_pct,
                        ABS(n - %(qn)s) AS ndiff,
                        0 AS frequency,
                        'suggestion' AS type
                    FROM results
                    WHERE ngram_text NOT ILIKE ANY(%(q)s)
                    AND adj_pct > 0
                    {f'AND n >= %(n_min)s' if n_min else ''}
                    {f'AND n <= %(n_max)s' if n_max else ''}
                    ORDER BY phones_distance, adj_pct DESC NULLS LAST, ndiff
                    -- {f'LIMIT %(limit)s OFFSET %(offset)s' if limit else ''}
                ;
            ''', dict(q=[q] + syns, qn=qn, qphones=qphones, limit=limit, offset=offset,
                      n_min=n_min, n_max=n_max))

            columns = [col[0] for col in cursor.description]
            vals = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
            if limit:
                vals = vals[offset:offset+limit]
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
