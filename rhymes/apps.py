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
                Rhyme.objects.top_rhymes(0, 100)
                Rhyme.objects.query(f'startup {now}', 0, 100)
                NGram.objects.completions(f'startup {now}', 20)
        except:
            pass
