param([switch]$SkipDownload)
$ErrorActionPreference = 'Stop'
$taskRoot = 'I:\isaac61'
$repoRoot = $PSScriptRoot
$simRoot = "$taskRoot\IsaacSim"
$labRoot = "$taskRoot\IsaacLab"
$archive = "$taskRoot\downloads\isaac-sim-standalone-6.1.0-windows-x86_64.zip"
$dirs = @('downloads','tmp','logs','data','documents','config','cache','cache\pip','cache\uv','cache\torch','cache\cuda','cache\warp','cache\huggingface')
foreach ($dir in $dirs) { New-Item -ItemType Directory -Force -Path "$taskRoot\$dir" | Out-Null }
$env:TEMP = "$taskRoot\tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$taskRoot\cache\pip"
$env:UV_CACHE_DIR = "$taskRoot\cache\uv"
$env:TORCH_HOME = "$taskRoot\cache\torch"
$env:HF_HOME = "$taskRoot\cache\huggingface"
$env:CUDA_CACHE_PATH = "$taskRoot\cache\cuda"
$env:XDG_CACHE_HOME = "$taskRoot\cache"
$env:WARP_CACHE_PATH = "$taskRoot\cache\warp"
$env:OMNI_KIT_ACCEPT_EULA = 'yes'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CONDA_PREFIX = $null
$env:VIRTUAL_ENV = $null
$env:CONDA_DEFAULT_ENV = $null
$env:PYTHONEXE = $null
$env:PYTHONHOME = $null
$env:PYTHONPATH = $null
if (!(Test-Path "$simRoot\python.bat")) {
    if (!$SkipDownload) {
        curl.exe -fLsS --retry 3 -C - 'https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.1.0-windows-x86_64.zip' -o $archive
        if ($LASTEXITCODE -ne 0) { throw 'NVIDIA download failed' }
    }
    if (!(Test-Path $archive)) { throw "Missing installer: $archive" }
    if ((Get-FileHash $archive -Algorithm MD5).Hash.ToLower() -ne 'a07968e980072c9ca27b2166443e2d89') { throw 'Installer checksum mismatch' }
    New-Item -ItemType Directory -Force -Path $simRoot | Out-Null
    tar.exe -xf $archive -C $simRoot
    if ($LASTEXITCODE -ne 0) {
        # Windows tar can skip the USD Unicode-reference test filename. The
        # bundled Python ZIP reader handles that filename correctly.
        if (!(Test-Path "$simRoot\python.bat")) { throw 'Extraction failed before Python was unpacked' }
        & "$simRoot\python.bat" -m zipfile -e $archive $simRoot
        if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
    }
}
if (!(Test-Path "$labRoot\isaaclab.bat")) {
    git -c core.longpaths=true clone --depth 1 --branch v3.0.0-EA https://github.com/isaac-sim/IsaacLab.git $labRoot
    if ($LASTEXITCODE -ne 0) { throw 'Isaac Lab clone failed' }
}
$revision = git -C $labRoot rev-parse HEAD
if ($revision -ne 'ae37b028ea415c91ea2bc32609efcd759ed2b974') { throw 'Isaac Lab checkout is not the pinned v3.0.0-EA revision' }
if (!(Test-Path "$labRoot\_isaac_sim")) { New-Item -ItemType Junction -Path "$labRoot\_isaac_sim" -Target $simRoot | Out-Null }
if (!(Test-Path "$repoRoot\isaac_workspace_61")) { New-Item -ItemType Junction -Path "$repoRoot\isaac_workspace_61" -Target $taskRoot | Out-Null }
$env:ISAACSIM_PATH = $simRoot
$env:ISAACSIM_PYTHON_EXE = "$simRoot\python.bat"
$torchWheel = "$taskRoot\downloads\torch-2.11.0+cu128-cp312-cp312-win_amd64.whl"
if (Test-Path $torchWheel) {
    if ((Get-FileHash $torchWheel -Algorithm SHA256).Hash.ToLower() -ne '7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c') { throw 'PyTorch wheel checksum mismatch' }
    & "$simRoot\python.bat" -m pip install $torchWheel 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Cached PyTorch install failed' }
}
# The official post-install script only creates this examples link. A junction
# provides the same directory access without requiring Windows symlink privilege.
if (!(Test-Path "$simRoot\extension_examples")) {
    New-Item -ItemType Junction -Path "$simRoot\extension_examples" -Target "$simRoot\exts\isaacsim.examples.interactive\isaacsim\examples\interactive" | Out-Null
}
Push-Location $labRoot
try {
    & "$labRoot\isaaclab.bat" --install 'rl[rsl-rl],visualizer[kit]'
    if ($LASTEXITCODE -ne 0) { throw 'Isaac Lab install failed' }
} finally { Pop-Location }
& "$simRoot\python.bat" -m pip show isaaclab isaaclab-physx isaaclab-rl rsl-rl-lib
if ($LASTEXITCODE -ne 0) { throw 'Isaac Lab dependency verification failed' }
# The Windows standalone archive omits these transitive dependencies of its
# bundled SimReady, OSQP, cloud and authentication packages.
& "$simRoot\python.bat" -m pip install joblib PyJWT 'aioboto3==15.5.0' usd-validation-nvidia --extra-index-url https://pypi.nvidia.com
if ($LASTEXITCODE -ne 0) { throw 'Bundled dependency repair failed' }
& "$simRoot\python.bat" -m pip install -e "${repoRoot}[audio]"
if ($LASTEXITCODE -ne 0) { throw 'Project install failed' }
& "$simRoot\python.bat" -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
& "$simRoot\python.bat" -m pip freeze | Set-Content -Encoding utf8 "$taskRoot\logs\installed-packages.txt"
if ($LASTEXITCODE -ne 0) { throw 'Dependency inventory failed' }
Write-Output "Installation completed: $simRoot; $labRoot"
