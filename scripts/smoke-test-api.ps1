#Requires -Version 5.1
<#
.SYNOPSIS
    Developer smoke test for the VoiceGuard backend integration.

.DESCRIPTION
    Sends a WAV clip as multipart/form-data to the Spring Boot endpoint
    POST /api/analyze, which forwards it to the FastAPI AI service
    (/predict) and combines the result with the existing RiskEngine.

    Privacy: the clip is streamed straight from disk into the HTTP
    request. Its contents are never printed, copied, or saved by this
    script - only file metadata (name, byte size) and the server's JSON
    verdict appear in the output.

.PARAMETER WavPath
    Optional path to a .wav file. If omitted, a synthetic 2-second
    16 kHz mono tone is generated in the temp folder and deleted again
    after the run.

.PARAMETER SpringBaseUrl
    Base URL of the Spring Boot backend. Default: http://127.0.0.1:8080

.EXAMPLE
    .\scripts\smoke-test-api.ps1

.EXAMPLE
    .\scripts\smoke-test-api.ps1 -WavPath C:\audio\sample.wav

.EXAMPLE
    .\scripts\smoke-test-api.ps1 -WavPath sample.wav -SpringBaseUrl http://127.0.0.1:8080

.NOTES
    Exit codes:
      0  analysis succeeded
      1  usage error (missing WAV file, oversized upload, ...)
      2  Spring Boot backend not reachable
      3  analysis request failed (HTTP error or invalid response)
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$WavPath,

    [string]$SpringBaseUrl = "http://127.0.0.1:8080"
)

$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 does not load System.Net.Http by default.
if (-not ("System.Net.Http.HttpClient" -as [type])) {
    Add-Type -AssemblyName System.Net.Http
}

$AnalyzeUri   = "$SpringBaseUrl/api/analyze"
$MaxUploadMB  = 25          # mirrors backend multipart limit
$RequestTimeoutSec = 60     # CPU inference is slow; stay generous

function Fail {
    # Print an error and leave a precise exit code for CI usage.
    param([Parameter(Mandatory)][string]$Message, [int]$Code = 1)
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit $Code
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Test-SpringReachable {
    param([Parameter(Mandatory)][string]$BaseUrl)

    # The API is POST-only, so a 405 from GET proves the server is up.
    try {
        Invoke-WebRequest -Uri "$BaseUrl/api/risk/analyze" -Method Get `
            -TimeoutSec 5 -UseBasicParsing | Out-Null
        return $true
    }
    catch {
        # Any HTTP response (405/404/...) means the server answered.
        if ($null -ne $_.Exception.Response) { return $true }
        return $false
    }
}

function New-TemporaryTestWav {
    # Generates a small deterministic 16 kHz mono PCM WAV (decaying sine).
    param([Parameter(Mandatory)][string]$Destination,
          [int]$Seconds = 2,
          [int]$SampleRate = 16000)

    $samples = $Seconds * $SampleRate
    $pcm = New-Object System.IO.MemoryStream
    try {
        for ($i = 0; $i -lt $samples; $i++) {
            $t = $i / $SampleRate
            $amp = 0.35 * [Math]::Exp(-1.5 * $t)
            $value = [int16][Math]::Round($amp * 32767 * [Math]::Sin(2 * [Math]::PI * 220 * $t))
            $bytes = [BitConverter]::GetBytes($value)
            $pcm.Write($bytes, 0, $bytes.Length)
        }

        $wav = New-Object System.IO.MemoryStream
        $writer = New-Object System.IO.BinaryWriter($wav)
        try {
            $ascii = [System.Text.Encoding]::ASCII
            $writer.Write($ascii.GetBytes("RIFF"))
            $writer.Write([int](36 + $pcm.Length))
            $writer.Write($ascii.GetBytes("WAVE"))
            $writer.Write($ascii.GetBytes("fmt "))
            $writer.Write([int]16)                    # fmt chunk size
            $writer.Write([uint16]1)                  # PCM
            $writer.Write([uint16]1)                  # mono
            $writer.Write([int]$SampleRate)
            $writer.Write([int]($SampleRate * 2))     # byte rate
            $writer.Write([uint16]2)                  # block align
            $writer.Write([uint16]16)                 # bits per sample
            $writer.Write($ascii.GetBytes("data"))
            $writer.Write([int]$pcm.Length)
            $writer.Write($pcm.ToArray())
            $writer.Flush()
            [System.IO.File]::WriteAllBytes($Destination, $wav.ToArray())
        }
        finally {
            $writer.Dispose()
            $wav.Dispose()
        }
    }
    finally {
        $pcm.Dispose()
    }
}

function Send-AnalysisRequest {
    # Streams the WAV file into a multipart/form-data POST and returns
    # @{ StatusCode; Body }. The file is never copied or buffered to disk.
    param([Parameter(Mandatory)][string]$FilePath,
          [Parameter(Mandatory)][string]$Uri)

    $boundary = "----VoiceGuardSmoke" + [Guid]::NewGuid().ToString("N")
    $content = New-Object System.Net.Http.MultipartFormDataContent($boundary)
    $client  = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds($RequestTimeoutSec)
    $fs = $null
    try {
        $fs = [System.IO.File]::OpenRead($FilePath)
        $streamContent = New-Object System.Net.Http.StreamContent($fs)
        $streamContent.Headers.ContentType =
            [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("audio/wav")
        $content.Add($streamContent, "file", [System.IO.Path]::GetFileName($FilePath))

        $response = $client.PostAsync($Uri, $content).GetAwaiter().GetResult()
        $body     = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return @{ StatusCode = [int]$response.StatusCode; Body = $body }
    }
    finally {
        $client.Dispose()
        $content.Dispose()   # also disposes the stream content
        if ($fs) { $fs.Dispose() }
    }
}

function Write-Verdict {
    param([Parameter(Mandatory)]$Json)
    Write-Host ""
    Write-Host "============== VoiceGuard verdict =============="
    Write-Host ("Spoof probability : {0:P1}  (raw: {1})" -f [double]$Json.spoofProbability, [double]$Json.spoofProbability)
    Write-Host ("Risk score        : {0}" -f $Json.riskScore)
    Write-Host ("Risk level        : {0}" -f $Json.riskLevel)
    Write-Host ("Alert             : {0}" -f $Json.alert)
    Write-Host ("AI label          : {0}" -f $Json.label)
    Write-Host ("Inference time    : {0} ms" -f $Json.inference_time_ms)
    if ($Json.message) { Write-Host ("Message           : {0}" -f $Json.message) }
    Write-Host "================================================"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

$tempWav = $null
try {
    Write-Host "VoiceGuard API smoke test"
    Write-Host "Backend: $AnalyzeUri"

    # 1. Resolve the WAV file (generate a synthetic one if omitted).
    if ([string]::IsNullOrWhiteSpace($WavPath)) {
        $tempWav = Join-Path ([System.IO.Path]::GetTempPath()) `
            ("voiceguard-smoke-" + [Guid]::NewGuid().ToString("N") + ".wav")
        New-TemporaryTestWav -Destination $tempWav
        $WavPath = $tempWav
        Write-Host "No WAV path given - generated synthetic 2 s test clip."
    }

    # 2. Validate the WAV file.
    if (-not (Test-Path -LiteralPath $WavPath -PathType Leaf)) {
        Fail "WAV file not found: $WavPath" 1
    }
    if ([System.IO.Path]::GetExtension($WavPath) -ne ".wav") {
        Write-Warning "File does not have a .wav extension; the backend may reject it."
    }
    $sizeMB = [Math]::Round((Get-Item -LiteralPath $WavPath).Length / 1MB, 2)
    if ($sizeMB -gt $MaxUploadMB) {
        Fail "File is $sizeMB MB and exceeds the $MaxUploadMB MB backend upload limit." 1
    }
    Write-Host ("Clip   : {0} ({1} MB)" -f (Split-Path -Leaf $WavPath), $sizeMB)

    # 3. Check that Spring Boot is reachable.
    if (-not (Test-SpringReachable -BaseUrl $SpringBaseUrl)) {
        Fail "Spring Boot is not reachable at $SpringBaseUrl. Start it with:  cd backend; .\mvnw.cmd spring-boot:run" 2
    }
    Write-Host "Spring Boot is reachable. Sending clip for analysis..."

    # 4. Send the multipart request.
    try {
        $result = Send-AnalysisRequest -FilePath $WavPath -Uri $AnalyzeUri
    }
    catch {
        Fail "Analysis request failed: $($_.Exception.Message)" 3
    }

    if ($result.StatusCode -ne 200) {
        Fail "Analysis request failed with HTTP $($result.StatusCode). $(if ($result.Body) { $result.Body })" 3
    }

    # 5. Parse and display the verdict.
    try {
        $json = $result.Body | ConvertFrom-Json
    }
    catch {
        Fail "Backend returned 200 but the body is not valid JSON." 3
    }

    foreach ($field in "spoofProbability", "riskScore", "riskLevel", "alert", "label", "inference_time_ms") {
        if (-not ($json.PSObject.Properties.Name -contains $field)) {
            Fail "Backend response is missing field '$field'." 3
        }
    }

    Write-Host ""
    Write-Host "Raw response:"
    $json | ConvertTo-Json -Depth 5
    Write-Verdict -Json $json
    exit 0
}
finally {
    # 6. Remove the synthetic clip, if one was generated.
    if ($tempWav -and (Test-Path -LiteralPath $tempWav)) {
        Remove-Item -LiteralPath $tempWav -Force -ErrorAction SilentlyContinue
    }
}
