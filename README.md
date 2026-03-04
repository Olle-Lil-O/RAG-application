# RAG-application

## Database quick start

This project runs PostgreSQL + pgvector in Docker Compose.

- Service name: `database`
- Container name: `pgvector_database`
- Host port: `5431` (mapped to container `5432`)
- Database: `postgres`
- User: `postgres`
- Password: `password`

Start only the database:

```bash
docker compose up -d database
```

Check logs:

```bash
docker compose logs -f database
```

Stop services:

```bash
docker compose down
```

## Connect to the database container (`docker exec -it`)

Open a `psql` session directly:

```bash
docker exec -it pgvector_database psql -U postgres -d postgres
```

Or open a shell first, then connect:

```bash
docker exec -it pgvector_database bash
psql -U postgres -d postgres
```

Inside `psql`, useful commands:

```sql
\l
\dt
\d+ knowledge_base
SELECT * FROM flyway_schema_history ORDER BY installed_rank;
```

## Migrations (Flyway)

SQL migration files are in `database-migrations/`.

Run database + migrations together (recommended):

```bash
docker compose up database database-migrations --force-recreate --abort-on-container-exit
```

Expected result: Flyway applies migrations and exits with code `0`.

If the database is already running, run migrations only:

```bash
docker compose run --rm database-migrations
```

Verify applied migrations:

```bash
docker exec -it pgvector_database psql -U postgres -d postgres -c "SELECT version, description, success, installed_on FROM flyway_schema_history ORDER BY installed_rank;"
```

## Connect from host tools

Example with local `psql` client:

```bash
PGPASSWORD=password psql -h localhost -p 5431 -U postgres -d postgres
```

## Connect from local Python

When connecting from your host machine (not from inside Docker), use:

- Host: `localhost`
- Port: `5431`
- User: `postgres`
- Password: `password`
- Database: `postgres`

Important: PostgreSQL inside the container listens on `5432`, but this project maps it to host port `5431` (`5431:5432`).
Use `5431` from local Python so you do not collide with a local PostgreSQL instance on `5432`.

### Connection string (DSN)

```text
postgresql://postgres:password@localhost:5431/postgres
```

### Environment variables

```bash
export PGHOST=localhost
export PGPORT=5431
export PGUSER=postgres
export PGPASSWORD=password
export PGDATABASE=postgres
```

### Python (`dotenv` + `os`)

Install dependency:

```bash
pip install python-dotenv
```

Create a `.env` file (for local app usage):

```dotenv
PGHOST=localhost
PGPORT=5431
PGUSER=postgres
PGPASSWORD=password
PGDATABASE=postgres
```

Read variables and build a DSN:

```python
import os
from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv("PGHOST", "localhost")
db_port = os.getenv("PGPORT", "5431")
db_user = os.getenv("PGUSER", "postgres")
db_password = os.getenv("PGPASSWORD", "password")
db_name = os.getenv("PGDATABASE", "postgres")

dsn = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
print(dsn)
```

### Python (`psycopg`)

```python
import psycopg

conn = psycopg.connect("postgresql://postgres:password@localhost:5431/postgres")
with conn.cursor() as cur:
    cur.execute("SELECT version();")
    print(cur.fetchone())
conn.close()
```

