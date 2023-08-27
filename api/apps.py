import time
from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        from django.conf import settings
        from api.models import Rhyme, NGram
        try:
            if settings.IS_PROD:
                now = str(int(time.time()))
                Rhyme.objects.top_rhymes()
                Rhyme.objects.query(q=f'startup {now}')
                NGram.objects.completions(q=f'startup {now}')
        except:
            pass
