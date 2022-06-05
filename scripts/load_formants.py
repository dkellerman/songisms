#!/usr/bin/env python

import os, django
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()

from api.models import NGram
from api.utils import get_formants, get_phones

for n in tqdm(NGram.objects.all()):
    n.formants = get_formants(n.text)
    n.phones = get_phones(n.text)
    n.save()
