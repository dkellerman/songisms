from django.core.management.base import BaseCommand
import argparse
from rhymes import nn
from songisms import utils


class Command(BaseCommand):
    help = '''Rhyme-detecting neural net'''

    def add_arguments(self, parser):
        parser.add_argument('--train', '-t', action=argparse.BooleanOptionalAction, help='Train model')
        parser.add_argument('--test', '-T', action=argparse.BooleanOptionalAction, help='Test model')
        parser.add_argument('--predict', '-p', nargs=2, help='Predict rhymes, with extra info')

    def handle(self, *args, **options):
        if options['train']:
            nn.train()
        if options['test']:
            nn.test()
        if options['predict']:
            text1, text2 = options['predict']
            self.predict_with_info(text1, text2)

    def predict_with_info(self, text1, text2):
        score = nn.predict(text1, text2)

        print("IPA:", utils.data.ipa.get(text1), '|', utils.data.ipa.get(text2))
        print("Tails:", utils.data.ipa.get(text1), '|', utils.data.ipa.get(text2))
        print("Score:", score)
        print("===>", nn.score_to_label(score))
