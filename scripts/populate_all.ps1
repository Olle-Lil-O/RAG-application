$ErrorActionPreference = 'Stop'

$pdfPath = if ($env:PDF_PATH) { $env:PDF_PATH } else { 'data/euaiact.pdf' }
$sourceName = if ($env:SOURCE_NAME) { $env:SOURCE_NAME } else { [System.IO.Path]::GetFileName($pdfPath) }
$spacyModel = if ($env:SPACY_MODEL) { $env:SPACY_MODEL } else { 'en_core_web_sm' }
$maxSentences = if ($env:MAX_SENTENCES) { $env:MAX_SENTENCES } else { '5' }
$chunker = if ($env:CHUNKER) { $env:CHUNKER } else { 'spacy' }
$breakpointThresholdType = if ($env:BREAKPOINT_THRESHOLD_TYPE) { $env:BREAKPOINT_THRESHOLD_TYPE } else { 'percentile' }
$chunkingProvider = if ($env:CHUNKING_PROVIDER) { $env:CHUNKING_PROVIDER } else { 'local' }
$chunkingLocalModel = if ($env:CHUNKING_LOCAL_MODEL) { $env:CHUNKING_LOCAL_MODEL } else { 'sentence-transformers/all-MiniLM-L6-v2' }
$chunkingDeployment = if ($env:CHUNKING_DEPLOYMENT) { $env:CHUNKING_DEPLOYMENT } elseif ($env:DEPLOY_MEDIUM) { $env:DEPLOY_MEDIUM } else { '' }
$maxEmbedTokens = if ($env:MAX_EMBED_TOKENS) { $env:MAX_EMBED_TOKENS } else { '2000' }
$splitOverlapTokens = if ($env:SPLIT_OVERLAP_TOKENS) { $env:SPLIT_OVERLAP_TOKENS } else { '80' }

$pgHost = if ($env:PGHOST) { $env:PGHOST } else { 'localhost' }
$pgPort = if ($env:PGPORT) { $env:PGPORT } else { '5431' }
$pgDatabase = if ($env:PGDATABASE) { $env:PGDATABASE } else { 'postgres' }
$pgUser = if ($env:PGUSER) { $env:PGUSER } else { 'postgres' }
$pgPassword = if ($env:PGPASSWORD) { $env:PGPASSWORD } else { 'password' }

$noExec = ($env:NO_EXEC -eq '1')
$dryRun = ($env:DRY_RUN -eq '1')
$skipEmpty = ($env:SKIP_EMPTY -eq '1')

$baseArgs = @(
    '--pdf-path', $pdfPath,
    '--source', $sourceName,
    '--spacy-model', $spacyModel,
    '--max-sentences', $maxSentences,
    '--chunker', $chunker,
    '--chunking-provider', $chunkingProvider,
    '--chunking-local-model', $chunkingLocalModel,
    '--chunking-deployment', $chunkingDeployment,
    '--breakpoint-threshold-type', $breakpointThresholdType,
    '--max-embed-tokens', $maxEmbedTokens,
    '--split-overlap-tokens', $splitOverlapTokens,
    '--pg-host', $pgHost,
    '--pg-port', $pgPort,
    '--pg-database', $pgDatabase,
    '--pg-user', $pgUser,
    '--pg-password', $pgPassword
)

if ($dryRun) { $baseArgs += '--dry-run' }

function Invoke-Step {
    param([string[]]$Command)
    Write-Host "> Running: $($Command[0])"
    if ($noExec) {
        Write-Host '  (NO_EXEC enabled; command not executed)'
        return
    }
    & $Command[0] $Command[1..($Command.Length - 1)]
}

if (-not $skipEmpty) {
    Invoke-Step @('pwsh', '-File', 'scripts/empty_tables.ps1')
}

Invoke-Step @('uv', 'run', 'python', 'scripts/preprocess.py') + $baseArgs + @(
    '--table', 'knowledge_base_mini',
    '--provider', 'local',
    '--local-model', 'sentence-transformers/all-MiniLM-L6-v2'
)

Invoke-Step @('uv', 'run', 'python', 'scripts/preprocess.py') + $baseArgs + @(
    '--table', 'knowledge_base_sm',
    '--provider', 'local',
    '--local-model', 'BAAI/bge-large-en-v1.5'
)

if (-not $noExec) {
    if (-not $env:AZURE_ENDPOINT) { throw 'AZURE_ENDPOINT is required for Azure embeddings' }
    if (-not $env:AZURE_API_KEY) { throw 'AZURE_API_KEY is required for Azure embeddings' }
    if (-not $env:DEPLOY_MEDIUM) { throw 'DEPLOY_MEDIUM is required for knowledge_base_md' }
}

Invoke-Step @('uv', 'run', 'python', 'scripts/preprocess.py') + $baseArgs + @(
    '--table', 'knowledge_base_md',
    '--provider', 'azure',
    '--deployment', $env:DEPLOY_MEDIUM,
    '--azure-endpoint', $env:AZURE_ENDPOINT,
    '--azure-api-key', $env:AZURE_API_KEY
)

Write-Host 'Populate-all pipeline completed.'
