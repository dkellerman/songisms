#!/usr/bin/env python

import runpy
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Run script'

    def add_arguments(self, parser):
        parser.add_argument('script', nargs=1)
        parser.add_argument('script_args', nargs='*')

    def handle(self, *args, **options):
        s = options['script'][0]
        runpy.run_path(s, run_name='__main__')
