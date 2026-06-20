$envLine = Get-Content .env | Where-Object { $_ -match '^GROQ_API_KEY=' } | Select-Object -First 1

if (-not $envLine) {
    Write-Error "GROQ_API_KEY not found in .env"
    exit 1
}

$env:GROQ_API_KEY = $envLine.Split('=', 2)[1]

python rag_pipeline.py batch --questions questions.json --out groq_test_results.json
