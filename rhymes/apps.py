import time
from django.apps import AppConfig


class RhymesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rhymes"

    def ready(self):
        from django.conf import settings
        from rhymes.models import Rhyme, NGram
        try:
            if settings.IS_PROD:
                now = str(int(time.time()))
                Rhyme.objects.top_rhymes()
                Rhyme.objects.query(q=f'startup {now}')
                NGram.objects.completions(q=f'startup {now}')
        except:
            pass
