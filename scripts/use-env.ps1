param(
    [Parameter(Mandatory = $true)]
    [string]$Profile
)

$ErrorActionPreference = 'Stop'

switch ($Profile.ToLower()) {
    'local' { $envFile = '.env' }
    'remote' { $envFile = 'remote/remote.env' }
    default { $envFile = $Profile }
}

if (-not (Test-Path $envFile)) {
    throw "Env file not found: $envFile"
}

function Import-EnvFile {
    param([string]$FilePath)
    Get-Content $FilePath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) { return }
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2) {
            [Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
        }
    }
}

Import-EnvFile -FilePath $envFile
if ((Test-Path '.env') -and ($envFile -ne '.env')) {
    Import-EnvFile -FilePath '.env'
}

Write-Host "Loaded environment from: $envFile"
Write-Host "Active DB target: $($env:PGUSER)@$($env:PGHOST):$($env:PGPORT)/$($env:PGDATABASE) (sslmode=$($env:PGSSLMODE))"
