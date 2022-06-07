from django.contrib import admin
from django.db.models import JSONField, Count
from django.utils.safestring import mark_safe
from django_json_widget.widgets import JSONEditorWidget
from reversion_compare.admin import CompareVersionAdmin
from .models import *


@admin.register(Song)
class SongAdmin(CompareVersionAdmin):
    fields = ('title', 'spotify_id', 'spotify_player', 'artists',
              'writers', 'tags', 'lyrics', 'lyrics_raw', 'lyrics_ipa',
              'rhymes_raw', 'jaxsta_id', 'jaxsta_link', 'youtube_id',
              'youtube_link', 'audio_file', 'metadata',)
    search_fields = ('title', 'spotify_id',)
    list_display = ('title', 'artists_display', 'spotify_link', 'has_lyrics',
                    'has_audio', 'has_metadata', 'has_ipa', 'has_jaxsta_id',)
    readonly_fields = ('spotify_player', 'jaxsta_link', 'youtube_link', 'rhymes',)
    filter_horizontal = ('tags', 'artists', 'writers',)
    list_filter = ('tags',)
    formfield_overrides = {
        JSONField: {'widget': JSONEditorWidget},
    }

    def queryset(self, request, queryset):
        return queryset.prefetch_related('artists', 'writers', 'tags', 'rhymes')

    def artists_display(self, obj):
        return ', '.join([a.name for a in obj.artists.all()])
    artists_display.short_description = 'Artists'

    def has_audio(self, obj):
        return check(obj.audio_file)
    has_audio.short_description = 'Aud'

    def has_metadata(self, obj):
        return check(obj.metadata)
    has_metadata.short_description = 'MD'

    def has_lyrics(self, obj):
        return check(obj.lyrics)
    has_lyrics.short_description = 'Lyr'

    def has_ipa(self, obj):
        return check(obj.lyrics_ipa)
    has_ipa.short_description = 'IPA'

    def has_jaxsta_id(self, obj):
        return '-' if obj.jaxsta_id == '-' else check(obj.jaxsta_id)
    has_jaxsta_id.short_description = 'Jax'

    def tags_value(self, obj):
        return ', '.join([t.label for t in obj.tags.all()])
    tags_value.short_description = 'Tags'

    def spotify_link(self, obj):
        return mark_safe(f'<a href={obj.spotify_url} target="__blank">{obj.spotify_id}</a>')
    spotify_link.short_description = 'Spotify ID'

    def jaxsta_link(self, obj):
        return mark_safe(f'<a href={obj.jaxsta_url} target="__blank">{obj.jaxsta_id}</a>')
    jaxsta_link.short_description = 'Jaxsta Link'

    def youtube_link(self, obj):
        return mark_safe(f'<a href={obj.youtube_url} target="__blank">{obj.youtube_id}</a>')
    youtube_link.short_description = 'Youtube Link'


@admin.register(Artist)
class ArtistAdmin(CompareVersionAdmin):
    search_fields = ('name',)


@admin.register(Writer)
class WriterAdmin(CompareVersionAdmin):
    search_fields = ('name', 'alt_names',)
    list_display = ('name', 'alt_names',)


@admin.register(NGram)
class NGramAdmin(admin.ModelAdmin):
    search_fields = ('text',)
    list_display = ('text', 'n', 'song_ct', 'stresses', 'ipa', 'phones',)
    fields = ('text', 'n', 'song_ct', 'stresses', 'ipa', 'phones',)

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.prefetch_related('songs').annotate(
            song_ct=Count('songs'),
        ).order_by('-n', '-song_ct',)

    def song_ct(self, obj):
        return obj.song_ct


@admin.register(Rhyme)
class RhymeAdmin(admin.ModelAdmin):
    fields = ('from_ngram', 'to_ngram', 'song',)
    search_fields = ('from_ngram__text',)
    list_display = ('from_ngram', 'to_ngram', 'song',)


@admin.register(TaggedText)
class TaggedTextAdmin(admin.ModelAdmin):
    list_display = ('tag', 'song', 'snip',)

    def snip(self, obj):
        return obj.text[0:50] + ('...' if len(obj.text) > 50 else '')


@admin.register(Tag)
class TagAdmin(CompareVersionAdmin):
    list_display = ('value', 'label', 'category',)


def check(val):
    return '√' if bool(val) else 'X'
