"""
Django settings for songisms project.

Generated by 'django-admin startproject' using Django 4.0.4.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.0/ref/settings/
"""

import os
import json
import base64
import dj_database_url
from google.oauth2 import service_account
from pathlib import Path
import django_on_heroku

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ['SISM_DJANGO_SECRET_KEY']

is_prod = bool(os.environ.get('DYNO'))

DEBUG = not is_prod

ALLOWED_HOSTS = []

INTERNAL_IPS = [ '127.0.0.1', '0.0.0.0', ]

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.postgres',
    'whitenoise.runserver_nostatic',
    'django.contrib.staticfiles',
    'debug_toolbar',
    'reversion',
    'reversion_compare',
    'graphene_django',
    'corsheaders',
    'graphql_jwt.refresh_token.apps.RefreshTokenConfig',
    'api.apps.ApiConfig',
    'smuggler',
]

MIDDLEWARE = [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'songisms.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'songisms.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

# django-on-heroku replaces this with DATABASE_URL in production
SISM_DATABASE_URL = os.environ.get('SISM_DATABASE_URL', None)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'songisms',
        'USER': 'songisms',
        'PASSWORD': os.environ['SISM_DB_PASSWORD'],
    } if not SISM_DATABASE_URL else dj_database_url.parse(SISM_DATABASE_URL, conn_max_age=600),
}

# Password validation
# https://docs.djangoproject.com/en/4.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/New_York'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.0/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

GRAPHENE = {
    "SCHEMA": "api.schema.schema",
    "MIDDLEWARE": [
        "graphql_jwt.middleware.JSONWebTokenMiddleware",
    ],
}

GRAPHQL_JWT = {
    'JWT_AUTH_HEADER_PREFIX': 'Bearer',
}

AUTHENTICATION_BACKENDS = [
    "graphql_jwt.backends.JSONWebTokenBackend",
    "django.contrib.auth.backends.ModelBackend",
]

CORS_ALLOW_ALL_ORIGINS = not is_prod
CORS_ALLOWED_ORIGINS = [
    'https://rhymes.vercel.app',
    'https://songisms.vercel.app',
    'https://www.rhymium.com',
]
CORS_ALLOW_CREDENTIALS = True

DEFAULT_FILE_STORAGE = 'storages.backends.gcloud.GoogleCloudStorage'
GS_BUCKET_NAME = 'songisms.appspot.com'


key = json.loads(base64.b64decode(os.environ['SISM_GOOGLE_CREDENTIALS']))
GS_CREDENTIALS = service_account.Credentials.from_service_account_info(key)

ADD_REVERSION_ADMIN = True
REVERSION_COMPARE_FOREIGN_OBJECTS_AS_ID = False
REVERSION_COMPARE_IGNORE_NOT_REGISTERED = False

REDIS_URL = os.environ.get('SISM_REDIS_URL', os.environ.get('REDIS_URL'))

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': REDIS_URL,
        'KEY_PREFIX': 'sism',
        'TIMEOUT': None,
    }
}

SMUGGLER_FIXTURE_DIR = BASE_DIR / 'data' / 'fixtures'

USE_QUERY_CACHE = is_prod

MOISES_API_KEY = 'aabb8d1b-c4a3-4a1f-8ca3-93d395142ef5'

django_on_heroku.settings(locals())
