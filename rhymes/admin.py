from django.contrib import admin
from .models import NGram, Rhyme, Cache, Vote


@admin.register(NGram)
class NGramAdmin(admin.ModelAdmin):
    search_fields = ('text',)
    list_display = ('text', 'n', 'pct', 'adj_pct', 'title_pct', 'song_pct', 'mscore',)


@admin.register(Rhyme)
class RhymeAdmin(admin.ModelAdmin):
    search_fields = ('from_ngram__text', 'to_ngram__text',)
    list_display = ('from_ngram', 'to_ngram', 'score', 'source',)
    list_filter = ('source',)
    autocomplete_fields = ('to_ngram', 'from_ngram',)

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset \
            .select_related('from_ngram', 'to_ngram') \
            .order_by('frequency', 'source', 'from_ngram', 'to_ngram')


@admin.register(Cache)
class CacheAdmin(admin.ModelAdmin):
    list_display = ('key', 'version', 'updated')


@admin.register(Vote)
class VoteAdmin(admin.ModelAdmin):
    list_display = ('anchor', 'alt1', 'alt2', 'label', 'voter_uid', 'created',)
    readonly_fields = ('anchor', 'alt1', 'alt2', 'voter_uid', 'created',)
    search_fields = ('anchor', 'alt1', 'alt2', 'voter_uid',)
    list_filter = ('label',)
