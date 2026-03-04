# Remote RAG database

Same migrations can be run against Azure Flex postgres database. It is now in the same state as our local container.

## Variables needed to connect to remote

```
PGHOST=paavorei-pgvector.postgres.database.azure.com
PGUSER=dbadmin
PGPORT=5432
PGDATABASE=postgres
PGPASSWORD="{ask-paavo}" 
```

**NOTE:** You shouldn't set these in the same session as you are running containers or notebooks/scripts. Those need the same env variables, but values are different.

If you want  to access the remote directly with psql, you can run:

```bash
psql "host=paavorei-pgvector.postgres.database.azure.com port=5432 dbname=postgres user=dbadmin@paavorei-pgvector password=<ask-paavo> sslmode=require"
```
