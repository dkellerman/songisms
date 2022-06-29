## Folders
* `api` - django app for db/admin/graphql, deployed to heroku @ songisms.herokuapp.com
* `clients` - client apps
  * `browser` - api browser (VueJS), deployed to vercel @ songisms.vercel.app
  * `rhymes` - rhymium app (VueJS), deployed to vercel @ rhymes.vercel.app
* `data` - misc data
* `scripts` - misc scripts
* `songisms` - django project settings
* `Procfile` - heroku deployment info
* `pyproject.toml` - poetry project configuration file

## Local database setup
* Postgresql DB/user name: `songisms`
* `CREATE EXTENSION fuzzystrmatch`
* `CREATE EXTENSION cube`
* `./manage.py migrate`
* pull data from prod? `heroku pg:pull ...`
* `./manage.py createinitialrevisions`

## Django setup
* `poetry install`
* `poetry shell`
* `./manage.py createsuperuser`
* `./manage.py runserver`
* Browse admin: https://localhost:8000/admin/
* Browse GraphQL: https://localhost:8000/graphql/

## Client app setup
* `cd client/rhymes` (or `browser`)
* `nvm use 14`
* `yarn`
* `yarn dev` OR `vercel dev` (should have env vars setup)

## Env vars
* `NEXT_PUBLIC_SISM_GOOGLE_CREDENTIALS` (base64 encoded json)
* `NEXT_PUBLIC_SISM_API_BASE_URL` - default http://localhost:8000
* `SISM_DB_PASSWORD`
* `SISM_DJANGO_SECRET_KEY`

All apps deploy automatically when pushed to master branch.

