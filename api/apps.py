import time
from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        from api.models import Rhyme, NGram
        try:
            now = str(int(time.time()))
            Rhyme.objects.top_rhymes()
            Rhyme.objects.query(q=f'startup {now}')
            NGram.objects.suggest(q=f'startup {now}')
        except:
            pass
