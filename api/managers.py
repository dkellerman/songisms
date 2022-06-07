from django.db import models, connection
from .utils import make_synonyms, get_phones


class RhymeManager(models.Manager):
    def top(self, limit=None):
        from api.models import NGram
        qs = NGram.objects.annotate(
            frequency=models.Count('rhymed_from__song__id', distinct=True),
            ngram=models.F('text'),
            type=models.Value('top'),
        ).filter(frequency__gt=0) \
         .order_by('-frequency', 'text') \
         .values('ngram', 'frequency', 'type')

        if limit:
            qs = qs[:limit]

        return qs

    def query(self, q, limit=None):
        if not q:
            return []

        syns = make_synonyms(q)
        q = [q] + syns

        with connection.cursor() as cursor:
            cursor.execute(f'''
                WITH RECURSIVE rhymes AS (
                    SELECT
                        from_ngram.text AS from_ngram_text,
                        to_ngram.text AS to_ngram_text,
                        count(to_ngram.id) AS frequency,
                        1 AS level
                    FROM api_rhyme r
                    INNER JOIN api_ngram from_ngram ON from_ngram.id = r.from_ngram_id
                    INNER JOIN api_ngram to_ngram ON to_ngram.id = r.to_ngram_id
                    WHERE UPPER(from_ngram.text) ILIKE ANY(%(q)s)
                    GROUP BY from_ngram.text, to_ngram.text, level

                    UNION ALL

                    SELECT
                        from_ngram.text AS from_ngram_text,
                        to_ngram.text AS to_ngram_text,
                        0 AS frequency,
                        rhymes.level + 1 AS level2
                    FROM rhymes, api_rhyme r2
                    INNER JOIN api_ngram from_ngram ON from_ngram.id = r2.from_ngram_id
                    INNER JOIN api_ngram to_ngram ON to_ngram.id = r2.to_ngram_id
                    WHERE
                        UPPER(from_ngram.text) = UPPER(rhymes.to_ngram_text)
                        AND NOT UPPER(to_ngram.text) ILIKE ANY(%(q)s)
                        AND to_ngram.text != rhymes.to_ngram_text
                        AND rhymes.level <= 1
                    GROUP BY from_ngram.text, to_ngram.text, rhymes.level
                )
                    SELECT * from (
                        SELECT
                            DISTINCT ON (to_ngram_text)
                            to_ngram_text AS ngram,
                            frequency,
                            CASE
                              WHEN frequency = 0 THEN 'rhyme-l2'
                              ELSE 'rhyme'
                            END AS type
                        FROM rhymes
                        ORDER BY to_ngram_text
                    ) results
                        ORDER BY frequency DESC
                        {f'LIMIT %(limit)s' if limit else ''}
                ;
                ''', dict(q=q, limit=limit)
            )

            columns = [col[0] for col in cursor.description]
            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]

    def suggest(self, q, limit=None):
        if not q:
            return []

        syns = make_synonyms(q)
        qphones = get_phones(q, vowels_only=True, include_stresses=False)
        qn = len(q.split())
        q = [q] + syns

        with connection.cursor() as cursor:
            cursor.execute(f'''
                WITH results AS (
                    SELECT
                        ngram.text AS ngram_text,
                        ngram.n AS n,
                        ngram.phones,
                        COUNT(sn.song_id) as song_ct,
                        LEVENSHTEIN(%(qphones)s, ngram.phones) AS phones_distance
                    FROM
                        api_ngram ngram
                    INNER JOIN
                        api_song_ngrams sn ON sn.ngram_id = ngram.id
                    WHERE
                        NOT (ngram.text = ANY(%(q)s))
                        AND ngram.n <= 2
                    GROUP BY
                        ngram.text, ngram.phones, ngram.n, phones_distance
                    HAVING
                        COUNT(sn.song_id) > 1
                        AND LEVENSHTEIN(%(qphones)s, ngram.phones) <= 4
                    ORDER BY ngram.text
                )
                    SELECT
                        ngram_text AS ngram,
                        phones_distance,
                        ABS(n - %(qn)s) AS ndiff,
                        0 AS frequency,
                        'suggestion' AS type
                    FROM results
                    ORDER BY phones_distance
                    {f'LIMIT %(limit)s' if limit else ''}
                ;
            ''', dict(q=q, qn=qn, qphones=qphones, limit=limit))

            columns = [col[0] for col in cursor.description]
            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]


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
    def get_by_natural_key(self, name):
        return self.get(name=name)


class TagManager(models.Manager):
    def get_by_natural_key(self, category, value):
        return self.get(category=category, value=value)
