from django.urls import path
from graphene_django.views import GraphQLView
from django.views.decorators.csrf import csrf_exempt

graphql_view = csrf_exempt(GraphQLView.as_view(graphiql=True))

urlpatterns = [
    path('graphql/', graphql_view),
]
