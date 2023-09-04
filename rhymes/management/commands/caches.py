import argparse
from tqdm import tqdm
from django.core.cache import cache
from django.core.management.base import BaseCommand
from rhymes.models import NGram, Rhyme


class Command(BaseCommand):
    help = 'Caches'

    def add_arguments(self, parser):
        parser.add_argument('--reset', '-r', action=argparse.BooleanOptionalAction)


    def handle(self, *args, **options):
        reset = options['reset']

        if reset:
            reset_caches()
            return


def reset_caches():
    print('clearing caches')
    cc = cache._cache.get_client()
    keys = cc.keys('sism:*:completions_*') + \
           cc.keys('sism:*:query_*') + \
           cc.keys('sism:*:top_*')
    for key in keys:
        key = str(key).split(':')[-1][:-1]
        cache.delete(key)

    topn = 100
    qsize = 100
    sug_size = 20

    print("Top", topn)
    top = Rhyme.objects.top_rhymes(0, topn)

    for ngram in tqdm(top, desc='queries & completions cache'):
        val = ngram['ngram']
        Rhyme.objects.query(val, 0, qsize)
        for i in range(0, len(val) + 1):
            qsug = val[:i]
            NGram.objects.completions(qsug, sug_size)
