# Dot-source this at the start of any PowerShell command that builds the C++
# extension: `. .\cpp\dev_env.ps1`. PowerShell tool calls don't persist shell
# state between invocations, so this has to be re-applied every time -- but
# dot-sourcing keeps each command's own preamble to one line instead of
# repeating the vcvars-capture dance inline. Sets up: MSVC env vars (INCLUDE/
# LIB/PATH via vcvars64.bat, needed because vswhere-based auto-detection
# doesn't find this install -- see session notes) and puts venv Scripts on
# PATH (for ninja.exe).
#
# Paths are derived from this script's own location ($PSScriptRoot is
# .../<repo>/cpp), not hardcoded, so this works from any clone location.

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$vcvarsOutput = cmd.exe /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && set'
foreach ($line in $vcvarsOutput) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:PATH = "$RepoRoot\.venv\Scripts;" + $env:PATH
# We activate the VC env manually (see above) rather than letting distutils/
# torch.utils.cpp_extension do it via its own (broken, vswhere-dependent)
# detection -- this flag tells it the env is already set up, avoiding a
# "multiple activations" UserWarning that setuptools treats as fatal.
$env:DISTUTILS_USE_SDK = "1"

$PY = "$RepoRoot\.venv\Scripts\python.exe"
