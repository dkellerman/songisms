# Songisms

* Song lyric database
* Rhymes search client (rhymes.vercel.app)
* Song browser/admin

## Local database setup
* Create Postgresql DB/user name: `songisms` or set `SISM_DATABASE_URL` env var
* `CREATE EXTENSION fuzzystrmatch`
* `CREATE EXTENSION cube`
* `./manage.py migrate`

## Django setup
* `poetry install`
* `poetry shell`
* `./manage.py createsuperuser`
* `./manage.py runserver`
* Browse admin: https://localhost:8000/admin/
* Browse GraphQL: https://localhost:8000/songs/graphql/

## Rhymes client app setup
* `cd rhymes/client`
* `nvm use 18`
* `yarn`
* `yarn dev` OR `vercel dev` (should have env vars setup)

## Env vars
* `NEXT_PUBLIC_SISM_GOOGLE_CREDENTIALS` (base64 encoded json)
* `NEXT_PUBLIC_SISM_API_BASE_URL` - default http://localhost:8000
* `SISM_DB_PASSWORD`
* `SISM_DJANGO_SECRET_KEY`

All apps deploy automatically when pushed to master branch.

