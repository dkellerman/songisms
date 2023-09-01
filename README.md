# Songisms

* Song lyric database
* Rhymes search client (rhymes.vercel.app)
* Song browser/admin

## App layout
* `songisms` -> main django project
* `songs` -> django app, for admin of DB
* `rhymes` -> django app, for querying rhymes
* `rhymes/client` -> nuxt app, rhymes front-end

-------
# Setup

## Setup Python env
* Requires python 3.9+
* Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
* `poetry install`
* `poetry shell`

## Setup DB
* Create Postgresql DB/user and set `SISM_DATABASE_URL` env var
* With psql:
  * `CREATE EXTENSION fuzzystrmatch`
  * `CREATE EXTENSION cube`
* `./manage.py migrate`

## Setup Django
* `./manage.py createsuperuser`
* `./manage.py runserver`
* Browse admin: https://localhost:8000/admin/
* Browse GraphQL: https://localhost:8000/songs/graphql/

## Setup rhymes client app
* `cd rhymes/client`
* `nvm use 18`
* `yarn`
* `yarn dev` OR `vercel dev` (should have env vars setup)

## Other env vars
* `NEXT_PUBLIC_SISM_GOOGLE_CREDENTIALS` (base64 encoded json)
* `NEXT_PUBLIC_SISM_API_BASE_URL` - default http://localhost:8000
* `SISM_DJANGO_SECRET_KEY`
* `SISM_RHYMES_API_BASE_URL` (use 127.0.0.1 instead of localhost for dev due to ofetch bug)

All apps deploy automatically when pushed to master branch.

# Rhymes neural net

For M1 Mac, do this first:
* `pip uninstall torch` (if it's installed)
* `pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

## Train/use:
* `./manage.py rhymesnet --train` - takes a while obv.
* Maybe: `./manage.py rhymesnet --test`
* `./manage.py rhymesnet --predict "phrase 1" "phrase 2"`

See params in `rhymes/nn.py`
