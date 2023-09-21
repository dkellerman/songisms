import argparse
from django.core.management.base import BaseCommand
from songs.models import Song
from songisms import utils
from tqdm import tqdm
from wonderwords import RandomWord
import pronouncing as pron


class Command(BaseCommand):
    help = 'Process IPA'

    def add_arguments(self, parser):
        parser.add_argument('--id', '-i', type=str, default=None)
        parser.add_argument('--limit', '-l', type=int, default=10)
        parser.add_argument('--fetch', '-f', action=argparse.BooleanOptionalAction)
        parser.add_argument('--force-fetch', '-F', action=argparse.BooleanOptionalAction)
        parser.add_argument('--cache', '-c', action=argparse.BooleanOptionalAction)
        parser.add_argument('--clear-cache', '-C', action=argparse.BooleanOptionalAction)


    def handle(self, *args, **options):
        if options['clear_cache']:
            print("Clearing cache")
            utils.get_ipa_cache().clear(save=False)

        if options['cache']:
            print("Filling cache")
            self.fill_cache()

        if options['fetch'] or options['force_fetch']:
            songs = Song.objects.filter(is_new=False).exclude(lyrics=None)
            if options['id']:
                songs = songs.filter(spotify_id__in=options['id'].split(','))
            limit = options['limit']
            queue = []

            for song in songs:
                if not song.metadata or not song.metadata.get('ipa') or options['force_fetch']:
                    queue.append(song)

            print("Queue size (TOTAL):", len(queue))
            if len(queue):
                print("Fetching", min(len(queue), limit or 0))
                for song in queue[0:limit] if limit else queue:
                    fetch_wrapper(song)

    def fill_cache(self, save=True):
        rw = RandomWord()
        all = set()

        def add_tok(tok):
            all.add(utils.to_ipa(tok, save_cache=False))

        for s in tqdm(Song.objects.exclude(lyrics=None), 'songs'):
            for tok in utils.tokenize_lyric(s.lyrics):
                add_tok(tok)
            if s.rhymes_raw:
                for tok1, tok2 in utils.get_rhyme_pairs(s.rhymes_raw):
                    add_tok(tok1)
                    add_tok(tok2)

        for l in tqdm(utils.data.lines, 'lines'):
            for tok in utils.tokenize_lyric(l):
                add_tok(tok)

        for i in tqdm(range(1000), 'random'):
            add_tok(rw.word())

        common = [c for c in list(utils.data.get_common_words(10000).keys())]
        for tok in tqdm(common, 'common'):
            add_tok(tok)

        for _, tok1, tok2 in tqdm(utils.data.rhymes_train, 'train'):
            add_tok(tok1)
            add_tok(tok2)

        for _, tok1, tok2 in tqdm(utils.data.rhymes_test, 'test'):
            add_tok(tok1)
            add_tok(tok2)

        all2 = list(all)
        for tok in tqdm(all2, 'rhymes'):
            for tok in [r for r in pron.rhymes(tok) if r in common]:
                add_tok(tok)

        print('total:', len(all))
        if save:
            utils.get_ipa_cache().save()


def fetch_wrapper(song):
    try:
        print("==> Fetching IPA", song.pk, song.title)
        ipa = utils.gpt_fetch_ipa(song.lyrics)
        song.metadata = song.metadata or {}
        song.metadata['ipa'] = ipa
        song.save()
    except Exception as err:
        print("Error fetching IPA", err, song.pk, song.title)

