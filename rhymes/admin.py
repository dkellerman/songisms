from django.contrib import admin
from django.db.models import Sum, Count
from .models import NGram, Rhyme, Cache


@admin.register(NGram)
class NGramAdmin(admin.ModelAdmin):
    search_fields = ('text',)
    list_display = ('text', 'n', 'song_ct', 'total_ct', 'ipa', 'mscore',)
    fields = ('text', 'n', 'song_ct', 'total_ct', 'ipa', 'phones', 'mscore',)
    readonly_fields = ('song_ct', 'total_ct',)

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.annotate(
            song_ct=Count('song_ngrams'),
            total_ct=Sum('song_ngrams__count'),
        ).order_by('-n', '-song_ct',)

    def song_ct(self, obj):
        return obj.song_ct

    def total_ct(self, obj):
        return obj.total_ct


@admin.register(Rhyme)
class RhymeAdmin(admin.ModelAdmin):
    fields = ('from_ngram', 'to_ngram', 'song_uid', 'level',)
    search_fields = ('from_ngram__text', 'to_ngram__text', 'song_uid',)
    list_display = ('from_ngram', 'to_ngram', 'level', 'song_uid',)
    autocomplete_fields = ('to_ngram', 'from_ngram',)

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset \
            .select_related('from_ngram', 'to_ngram') \
            .order_by('level', 'song_uid', 'from_ngram', 'to_ngram')


@admin.register(Cache)
class CacheAdmin(admin.ModelAdmin):
    list_display = ('key', 'version', 'updated')

