from urllib.parse import quote_plus
from django.contrib import admin
from django.contrib.contenttypes.admin import GenericTabularInline
from django.utils.safestring import mark_safe
from django.forms import Textarea
from .models import Artist, Writer, Tag, TaggedText, Song, Attachment


class AttachmentInline(GenericTabularInline):
    model = Attachment


@admin.register(Song)
class SongAdmin(admin.ModelAdmin):
    fields = ('is_new', 'title', 'spotify_id', 'spotify_player', 'artists',
              'writers', 'tags', 'lyrics', 'rhymes_raw', 'jaxsta_id', 'jaxsta_link',
              'youtube_id', 'youtube_link', 'audio_file', 'metadata',)
    search_fields = ('title', 'spotify_id', 'lyrics',)
    list_display = ('title', 'artists_display', 'spotify_link', 'is_new_check', 'has_lyrics',
                    'has_audio', 'has_metadata', 'has_jaxsta_id',)
    readonly_fields = ('spotify_player', 'jaxsta_link', 'youtube_link',)
    list_filter = ('is_new', 'tags',)
    autocomplete_fields = ('artists', 'tags', 'writers',)
    inlines = [AttachmentInline]
    change_list_template = 'smuggler/change_list.html'

    def queryset(self, request, queryset):
        return queryset.prefetch_related('artists', 'writers', 'tags')

    def formfield_for_dbfield(self, db_field, **kwargs):
        formfield = super().formfield_for_dbfield(db_field, **kwargs)
        if db_field.name == 'lyrics':
            formfield.widget = Textarea(attrs=dict(rows=40, cols=100))
        return formfield

    def artists_display(self, obj):
        return ', '.join([a.name for a in obj.artists.all()])
    artists_display.short_description = 'Artists'

    def is_new_check(self, obj):
        return check(obj.is_new)
    is_new_check.short_description = 'New'

    def has_audio(self, obj):
        return check(obj.audio_file)
    has_audio.short_description = 'Aud'

    def has_metadata(self, obj):
        return check(obj.metadata)
    has_metadata.short_description = 'MD'

    def has_lyrics(self, obj):
        return check(obj.lyrics)
    has_lyrics.short_description = 'Lyr'

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
        if obj.youtube_id:
            return mark_safe(f'<a href={obj.youtube_url} target="__blank">{obj.youtube_id}</a>')
        else:
            q = quote_plus(
                f'{obj.title} {obj.artists.all()[0].name} official audio')
            return mark_safe(f'<a href="https://youtube.com/results?search_query={q}" target="_blank">Search</a>')
    youtube_link.short_description = 'Youtube Link'


@admin.register(Artist)
class ArtistAdmin(admin.ModelAdmin):
    search_fields = ('name',)
    change_list_template = 'smuggler/change_list.html'


@admin.register(Writer)
class WriterAdmin(admin.ModelAdmin):
    search_fields = ('name', 'alt_names',)
    list_display = ('name', 'alt_names',)
    change_list_template = 'smuggler/change_list.html'


@admin.register(TaggedText)
class TaggedTextAdmin(admin.ModelAdmin):
    list_display = ('tag', 'song', 'snip',)
    autocomplete_fields = ('tag', 'song',)
    change_list_template = 'smuggler/change_list.html'

    def snip(self, obj):
        return obj.text[0:50] + ('...' if len(obj.text) > 50 else '')


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('value', 'label', 'category',)
    search_fields = ('value', 'label', 'category',)
    change_list_template = 'smuggler/change_list.html'


@admin.register(Attachment)
class AttachmentAdmin(admin.ModelAdmin):
    list_display = ('content_object', 'attachment_type',)
    list_filter = ('attachment_type',)


def check(val):
    return '√' if bool(val) else 'X'
