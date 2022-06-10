import graphene
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from graphene_django import DjangoObjectType
import graphql_jwt
from graphql_jwt.decorators import login_required
from .models import Song, Artist, Writer, Tag, Rhyme, NGram, TaggedText
from .utils import get_paginator, GraphenePaginatedType

DEFAULT_PAGE_SIZE = 20


class UserType(DjangoObjectType):
    class Meta:
        model = User
        fields = ['username']


class SongType(DjangoObjectType):
    spotify_url = graphene.String(source='spotify_url')
    spotify_player = graphene.String(source='spotify_player')
    jaxsta_url = graphene.String(source='jaxsta_url')
    youtube_url = graphene.String(source='youtube_url')
    youtube_player = graphene.String(source='youtube_player')
    audio_file_url = graphene.String(source='audio_file_url')

    class Meta:
        model = Song
        fields = ['title', 'artists', 'writers', 'tags', 'spotify_id', 'lyrics',
                  'lyrics_raw', 'lyrics_ipa', 'jaxsta_id', 'rhymes',
                  'youtube_id', 'audio_file', 'metadata', 'tagged_texts',
                  'created', 'updated', 'spotify_url', 'jaxsta_url', 'youtube_url',
                  'spotify_player', 'youtube_player', 'audio_file_url',
                  'rhymes_raw']


class ArtistType(DjangoObjectType):
    class Meta:
        model = Artist
        fields = ['name', 'songs']


class WriterType(DjangoObjectType):
    song_ct = graphene.Int()
    class Meta:
        model = Writer
        fields = ['name', 'songs', 'song_ct']


class RhymeType(graphene.ObjectType):
    ngram = graphene.String()
    frequency = graphene.Int()
    type = graphene.String()

    class Meta:
        fields = ['ngram', 'frequency', 'type', ]


class NGramType(DjangoObjectType):
    class Meta:
        model = NGram
        fields = ['text', ]


class TagType(DjangoObjectType):
    class Meta:
        model = Tag
        fields = ['value', 'label', 'songs', 'texts']
        convert_choices_to_enum = False


class TaggedTextType(DjangoObjectType):
    class Meta:
        model = TaggedText
        fields = ['text', 'tag', 'song']
        convert_choices_to_enum = False


SongsPaginatedType = GraphenePaginatedType('SongsPaginatedType', SongType)
WritersPaginatedType = GraphenePaginatedType('WritersPaginatedType', WriterType)
ArtistsPaginatedType = GraphenePaginatedType('ArtistsPaginatedType', ArtistType)
NGramsPaginatedType = GraphenePaginatedType('NGramsPaginatedType', NGramType)


class Query(graphene.ObjectType):
    user = graphene.Field(UserType)
    songs = graphene.Field(SongsPaginatedType,
                           page=graphene.Int(required=False),
                           q=graphene.String(required=False),
                           tags=graphene.List(required=False, of_type=graphene.String),
                           ordering=graphene.List(required=False, of_type=graphene.String))
    song = graphene.Field(SongType, spotify_id=graphene.String(required=True))
    artists = graphene.Field(ArtistsPaginatedType,
                             page=graphene.Int(required=False),
                             q=graphene.String(required=False),
                             ordering=graphene.List(required=False, of_type=graphene.String))
    artist = graphene.Field(ArtistType, name=graphene.String(required=True))
    writers = graphene.Field(WritersPaginatedType,
                             page=graphene.Int(required=False),
                             q=graphene.String(required=False),
                             ordering=graphene.List(required=False, of_type=graphene.String))
    writer = graphene.Field(WriterType, name=graphene.String(required=True))
    rhymes = graphene.List(RhymeType,
                           q=graphene.String(required=False),
                           limit=graphene.Int(required=False),
                           offset=graphene.Int(required=False),
                           tags=graphene.List(required=False, of_type=graphene.String),
                           n_min=graphene.Int(required=False),
                           n_max=graphene.Int(required=False),
                           search_type=graphene.String(required=False))
    ngrams = graphene.Field(NGramsPaginatedType,
                            page=graphene.Int(required=False),
                            q=graphene.String(required=False),
                            tags=graphene.List(required=False, of_type=graphene.String),
                            ordering=graphene.List(required=False, of_type=graphene.String))
    tags_by_category = graphene.List(TagType, category=graphene.String(required=True))

    @staticmethod
    @login_required
    def resolve_user(root, info):
        return info.context.user

    @staticmethod
    @login_required
    def resolve_songs(root, info, q=None, tags=None, page=1, ordering=()):
        songs = Song.objects.prefetch_related('artists', 'tags').order_by(*ordering)

        if q:
            songs = songs.filter(title__icontains=q)
        if tags:
            for tag in tags:
                songs = songs.filter(tags__value=tag)

        return get_paginator(songs, DEFAULT_PAGE_SIZE, page, SongsPaginatedType, q=q)

    @staticmethod
    @login_required
    def resolve_song(root, info, spotify_id):
        qs = Song.objects.prefetch_related('artists', 'tags', 'writers')
        return get_object_or_404(qs, spotify_id=spotify_id)

    @staticmethod
    def resolve_writers(root, info, q, page=1, ordering=None):
        writers = Writer.objects.query(q, ordering)
        return get_paginator(writers, DEFAULT_PAGE_SIZE, page, WritersPaginatedType, q=q)

    @staticmethod
    @login_required
    def resolve_writer(root, info, name):
        return get_object_or_404(Writer.objects.prefetch_related('songs'), name=name)

    @staticmethod
    @login_required
    def resolve_artists(root, info, q, page=1, ordering=('name',)):
        artists = Artist.objects.prefetch_related('songs').order_by(*ordering)
        if q:
            artists = artists.filter(name__icontains=q)
        return get_paginator(artists, DEFAULT_PAGE_SIZE, page, ArtistsPaginatedType, q=q)

    @staticmethod
    @login_required
    def resolve_artist(root, info, name):
        return get_object_or_404(Artist.objects.prefetch_related('songs'), name=name)

    @staticmethod
    @login_required
    def resolve_tags_by_category(root, info, category):
        return Tag.objects.filter(category=category).order_by('label')

    @staticmethod
    def resolve_rhymes(root, info, q=None, offset=0, limit=50, search_type='rhyme', n_min=None, n_max=None):
        if search_type == 'rhyme':
            qs = Rhyme.objects.query(q, limit=limit, offset=offset, n_min=n_min, n_max=n_max)
        elif search_type == 'suggest':
            qs = Rhyme.objects.suggest(q, limit=limit, offset=offset, n_min=n_min, n_max=n_max)
        else:
            return []

        return qs

    @staticmethod
    def resolve_ngrams(root, info, q, tags, page=1, ordering=None):
        ngrams = NGram.objects.by_query(q)
        if ordering:
            ngrams = ngrams.order_by(*ordering)
        return get_paginator(ngrams, DEFAULT_PAGE_SIZE, page, NGramsPaginatedType, q=q)


class Mutation(graphene.ObjectType):
    token_auth = graphql_jwt.ObtainJSONWebToken.Field()
    verify_token = graphql_jwt.Verify.Field()
    refresh_token = graphql_jwt.Refresh.Field()


schema = graphene.Schema(query=Query, mutation=Mutation)
