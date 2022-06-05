## Folders
* `api`` - django app for db/admin/graphql, deployed to heroku
* `client` - api browser (NextJS), deployed to vercel
* `data` - misc data
* `rhymes` - rhymium app (NextJS), deployed to vercel
* `scripts` - misc scripts
* `songisms` - django project
* `Procfile` - heroku deployment info

## Local database setup
* Postgresql DB name: `songisms`
* `CREATE EXTENSION fuzzystrmatch`
* `./manage.py migrate`
* pull data from prod? `heroku pg:pull ...`
* `./manage.py createinitialrevisions`

## Django setup
* `poetry install`
* `./manage.py createsuperuser`
* `./manage.py runserver`
* Browse admin: https://localhost:8000/admin/
* Browse GraphQL: https://localhost:8000/graphql/

## NextJS setup
* `cd client` (or `rhymes`)
* `yarn`
* `yarn dev` OR `vercel dev` (should have env vars setup)

## Env vars
* `NEXT_PUBLIC_SISM_GOOGLE_CREDENTIALS`
* `NEXT_PUBLIC_SISM_API_BASE_URL`
* `SISM_DB_PASSWORD`
* `SISM_DJANGO_SECRET_KEY`
* `REDIS_URL`

All apps deploy automatically when pushed to master branch.

