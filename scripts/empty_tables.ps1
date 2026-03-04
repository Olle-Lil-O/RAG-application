$ErrorActionPreference = 'Stop'

$pgHost = if ($env:PGHOST) { $env:PGHOST } else { 'localhost' }
$pgPort = if ($env:PGPORT) { $env:PGPORT } else { '5431' }
$pgDatabase = if ($env:PGDATABASE) { $env:PGDATABASE } else { 'postgres' }
$pgUser = if ($env:PGUSER) { $env:PGUSER } else { 'postgres' }
$pgPassword = if ($env:PGPASSWORD) { $env:PGPASSWORD } else { 'password' }
$pgSslMode = if ($env:PGSSLMODE) { $env:PGSSLMODE } else { 'disable' }

$sql = 'TRUNCATE TABLE knowledge_base_mini, knowledge_base_sm, knowledge_base_md RESTART IDENTITY;'
$conn = "host=$pgHost port=$pgPort dbname=$pgDatabase user=$pgUser password=$pgPassword sslmode=$pgSslMode"

if ($env:NO_EXEC -eq '1') {
    Write-Host "[NO_EXEC] Would run: psql -d `"$conn`" --set ON_ERROR_STOP=1 --command `"$sql`""
    exit 0
}

psql -d $conn --set ON_ERROR_STOP=1 --command $sql
Write-Host 'Tables truncated successfully.'
