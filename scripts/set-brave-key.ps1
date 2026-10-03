# Ввести новый ключ без печати в терминал и без передачи его в чат.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$secretDirectory = Join-Path $projectRoot '.secrets'
$keyPath = Join-Path $secretDirectory 'brave_api_key'
$environmentPath = Join-Path $projectRoot '.env'
if (-not (Test-Path -LiteralPath $environmentPath)) {
    throw 'Сначала создай .env из .env.example.'
}
$secureKey = Read-Host 'Новый Brave Search API key' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $newKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer).Trim()
    if (-not $newKey -or $newKey.Length -gt 4096 -or $newKey -match '\s') {
        throw 'Некорректный ключ.'
    }
    New-Item -ItemType Directory -Path $secretDirectory -Force | Out-Null
    [IO.File]::WriteAllText($keyPath, $newKey, [Text.UTF8Encoding]::new($false))
    $environmentText = [IO.File]::ReadAllText($environmentPath)
    $environmentText = [regex]::Replace($environmentText, '(?m)^BRAVE_SEARCH_API_KEY=.*$', 'BRAVE_SEARCH_API_KEY=')
    $keySetting = 'BRAVE_SEARCH_API_KEY_FILE=.secrets/brave_api_key'
    if ($environmentText -match '(?m)^BRAVE_SEARCH_API_KEY_FILE=') {
        $environmentText = [regex]::Replace($environmentText, '(?m)^BRAVE_SEARCH_API_KEY_FILE=.*$', $keySetting)
    } else {
        $environmentText = $environmentText.TrimEnd() + "`n" + $keySetting + "`n"
    }
    [IO.File]::WriteAllText($environmentPath, $environmentText, [Text.UTF8Encoding]::new($false))
    Write-Host 'Новый ключ сохранён. Перезапусти бота, проверь Brave Search и отзови старый ключ в панели Brave.'
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $newKey = $null
    $secureKey.Dispose()
}
