import graphene
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from django.db.models.functions import Lower
from graphene_django import DjangoObjectType
from graphene.types.generic import GenericScalar
import graphql_jwt
from graphql_jwt.decorators import login_required
from .models import Song, Artist, Writer, Tag, TaggedText, Attachment
from songisms.utils import get_paginator, GraphenePaginatedType

DEFAULT_PAGE_SIZE = 20

# dump: `./manage.py graphql_schema --schema songs.schema.schema --out songs/schema.graphql`


class UserType(DjangoObjectType):
    class Meta:
        model = User
        fields = ['username']


class SongIndexType(DjangoObjectType):
    class Meta:
        model = Song
        fields = ['title', 'spotify_id']


class AttachmentType(DjangoObjectType):
    url = graphene.String()

    class Meta:
        model = Attachment
        fields = ['attachment_type', 'url']

    def resolve_url(self, info):
        return self.file.url


class SongType(DjangoObjectType):
    spotify_url = graphene.String(source='spotify_url')
    spotify_player = graphene.String(source='spotify_player')
    jaxsta_url = graphene.String(source='jaxsta_url')
    youtube_url = graphene.String(source='youtube_url')
    youtube_player = graphene.String(source='youtube_player')
    audio_file_url = graphene.String(source='audio_file_url')
    metadata = GenericScalar(source='metadata')
    attachments = graphene.List(AttachmentType)

    class Meta:
        model = Song
        fields = ['title', 'artists', 'writers', 'tags', 'spotify_id', 'lyrics',
                  'is_new', 'jaxsta_id', 'rhymes', 'youtube_id', 'audio_file',
                  'metadata', 'tagged_texts', 'created', 'updated', 'spotify_url',
                  'jaxsta_url', 'youtube_url', 'spotify_player', 'youtube_player',
                  'audio_file_url', 'rhymes_raw', 'id', 'attachments']

    def resolve_attachments(self, info):
        return self.attachments.all()


class ArtistType(DjangoObjectType):
    class Meta:
        model = Artist
        fields = ['name', 'songs']


class WriterType(DjangoObjectType):
    song_ct = graphene.Int()
    alt_names = graphene.List(graphene.String)

    class Meta:
        model = Writer
        fields = ['id', 'name', 'songs', 'song_ct', 'alt_names']


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


class Query(graphene.ObjectType):
    user = graphene.Field(UserType)
    songs = graphene.Field(SongsPaginatedType,
                           page=graphene.Int(required=False),
                           q=graphene.String(required=False),
                           ordering=graphene.List(required=False, of_type=graphene.String))
    songs_index = graphene.List(SongIndexType)
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
    writer = graphene.Field(WriterType, id=graphene.Int(required=True))
    tags_by_category = graphene.List(
        TagType, category=graphene.String(required=True))

    @staticmethod
    @login_required
    def resolve_user(root, info):
        return info.context.user

    @staticmethod
    @login_required
    def resolve_songs_index(root, info):
        return Song.objects.all().order_by(Lower('title'))

    @staticmethod
    @login_required
    def resolve_songs(root, info, q=None, page=1, ordering=(Lower('title'),)):
        songs = Song.objects.query(q).order_by(*ordering)
        return get_paginator(songs, DEFAULT_PAGE_SIZE, page, SongsPaginatedType, q=q)

    @staticmethod
    @login_required
    def resolve_song(root, info, spotify_id):
        qs = Song.objects.prefetch_related('artists', 'tags', 'writers')
        return get_object_or_404(qs, spotify_id=spotify_id)

    @staticmethod
    @login_required
    def resolve_writers(root, info, q, page=1, ordering=None):
        writers = Writer.objects.query(q, ordering)
        return get_paginator(writers, DEFAULT_PAGE_SIZE, page, WritersPaginatedType, q=q)

    @staticmethod
    @login_required
    def resolve_writer(root, info, id):
        return get_object_or_404(Writer.objects.prefetch_related('songs'), pk=id)

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


class Mutation(graphene.ObjectType):
    token_auth = graphql_jwt.ObtainJSONWebToken.Field()
    verify_token = graphql_jwt.Verify.Field()
    refresh_token = graphql_jwt.Refresh.Field()


schema = graphene.Schema(query=Query, mutation=Mutation)
