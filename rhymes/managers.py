from django.db import models, connection
from django.db.models import F
from songisms import utils


class BaseManager(models.Manager):
    pass


class RhymeManager(BaseManager):
    def query(self, q, offset, limit, voter_uid=None):
        '''Main rhymes query
        '''
        if not q:
            return []

        q = utils.normalize_lyric(q)
        variants = utils.get_variants(q)
        all_q = [q] + variants

        target_ngram = '''
            CASE
                WHEN nfrom.text = ANY(%(q)s) THEN nto.text
                ELSE nfrom.text
            END
        '''

        votes_sql = f'''
            SELECT
                label
            FROM
                rhymes_vote v
            WHERE (
                v.anchor = %(qstr)s AND v.alt1 = {target_ngram}
                OR (v.anchor = {target_ngram} AND v.alt1 = %(qstr)s)
            ) AND v.voter_uid = %(voter_uid)s
            ORDER BY v.created DESC
            LIMIT 1
        '''

        rhymes_sql = f'''
            SELECT
                {target_ngram} AS ngram,
                CASE
                    WHEN nfrom.text = %(qstr)s OR nto.text = %(qstr)s
                    THEN rhyme.frequency
                    ELSE 0
                END AS frequency,
                rhyme.score AS score,
                rhyme.source AS source,
                {f'({votes_sql})' if voter_uid else 'NULL'} AS vote,
                'rhyme' AS type
            FROM
                rhymes_rhyme rhyme
            INNER JOIN
                rhymes_ngram nfrom ON rhyme.from_ngram_id = nfrom.id
            INNER JOIN
                rhymes_ngram nto ON rhyme.to_ngram_id = nto.id
            WHERE
                nfrom.text = ANY(%(q)s)
                OR nto.text = ANY(%(q)s)
        '''

        with connection.cursor() as cursor:
            cursor.execute(f'''
                WITH results AS ({rhymes_sql})
                SELECT
                    ngram,
                    frequency,
                    score,
                    source,
                    vote,
                    type
                FROM (SELECT DISTINCT ON (ngram) * FROM results ORDER BY ngram, frequency) uniq
                WHERE ngram != %(qstr)s
                ORDER BY frequency DESC, score DESC
                OFFSET %(offset)s
                LIMIT %(limit)s
            ''', dict(qstr=q, q=all_q, voter_uid=voter_uid, limit=limit, offset=offset))

            columns = [col[0] for col in cursor.description]
            vals = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]

            return vals


    def top_rhymes(self, offset, limit, q=None):
        '''Retrieve most-rhymed words
        '''
        from rhymes.models import NGram
        qs = NGram.objects.all()
        if q:
            starts_with, ends_with = q.split('*')
            if starts_with:
                qs = qs.filter(text__istartswith=starts_with)
            if ends_with:
                qs = qs.filter(text__iendswith=ends_with)

        qs = qs.annotate(
            ngram=models.F('text'),
            rfrequency=models.Count('rhymed_from__pk', distinct=True) +
                       models.Count('rhymed_to__pk', distinct=True)
        )
        qs = qs.filter(rfrequency__gt=0).order_by('-rfrequency', 'text')
        qs = qs[offset:offset+limit].values('ngram', 'rfrequency',)

        return qs


class NGramManager(BaseManager):
    def get_by_natural_key(self, text):
        return self.get(text=text)

    def completions(self, q, limit):
        '''Auto-suggest completions for a given query
        '''
        if not q:
            return []

        from rhymes.models import NGram
        qs = NGram.objects.filter(text__istartswith=q)
        qs = qs.annotate(rhyme_ct=models.Count('rhymes')).filter(rhyme_ct__gt=0)
        qs = qs.order_by('n', '-rhyme_ct')[:limit]

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
