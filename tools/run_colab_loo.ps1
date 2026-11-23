[CmdletBinding()]
param(
    [ValidateSet('T4','CPU')][string]$Runtime = 'T4',
    [ValidateRange(2,100000)][int]$Reps = 1000,
    [ValidateRange(0,2147483647)][int]$Seed = 0,
    [string]$DriveRoot = 'G:\My Drive\rsa_colab',
    [string]$ColabCli
)
$ErrorActionPreference = 'Stop'
foreach ($species in 'D','H') {
    $package = Join-Path $DriveRoot "loo_group_${species}.zip"
    if (!(Test-Path -LiteralPath $package -PathType Leaf)) { throw "Missing package: $package" }
}
$prefix = @()
if ($ColabCli) {
    $cli = (Get-Command $ColabCli -ErrorAction Stop).Source
} else {
    $cli = (Get-Command wsl.exe -ErrorAction Stop).Source
    $linuxCli = (& $cli -e bash -lc 'command -v colab || test ! -x ~/.local/bin/colab || realpath ~/.local/bin/colab' | Select-Object -Last 1)
    if (!$linuxCli) { throw 'Install and authenticate google-colab-cli in WSL, or pass -ColabCli.' }
    $prefix = @('-e', $linuxCli.Trim())
}
$session = 'rsa-loo-' + [guid]::NewGuid().ToString('N').Substring(0,10)
$source = Join-Path $PSScriptRoot 'colab_gpu\boot_joint_loo.py'
$script = Join-Path ([IO.Path]::GetTempPath()) "$session.py"
$allocated = $false
$dirs = [Collections.Generic.List[string]]::new()
try {
    $hardware = @()
    $device = 'cpu'
    if ($Runtime -eq 'T4') { $hardware = @('--gpu','T4'); $device = 'cuda' }
    $code = "LOO_JOB = dict(device='$device', reps=$Reps, seed=$Seed)`n" + [IO.File]::ReadAllText($source)
    [IO.File]::WriteAllText($script,$code)
    & $cli @prefix new -s $session @hardware
    if ($LASTEXITCODE) { throw "Could not allocate Colab session ($LASTEXITCODE)." }
    $allocated = $true
    & $cli @prefix drivemount -s $session
    if ($LASTEXITCODE) { throw 'Google Drive mount failed.' }
    $remoteScript = $script
    if ($prefix.Count) {
        $remoteScript = (& $cli -e wslpath -a $script).Trim()
        if ($LASTEXITCODE) { throw 'Could not translate the bootstrap path for WSL.' }
    }
    & $cli @prefix exec -s $session --timeout 0 -f $remoteScript | ForEach-Object {
        Write-Host $_
        if ($_ -match 'VERIFIED_RESULT_DIR=([DH]_[0-9a-f]{20})') { $dirs.Add($Matches[1]) }
    }
    if ($LASTEXITCODE -or $dirs.Count -ne 2) { throw 'Incomplete run. Verified Drive checkpoints will be reused on restart.' }
} finally {
    if ($allocated) {
        & $cli @prefix stop -s $session
        if ($LASTEXITCODE) { Write-Warning "Could not release Colab session $session. Check colab sessions." }
    }
    if (Test-Path -LiteralPath $script) { Remove-Item -LiteralPath $script }
}
$python = 'C:\ProgramData\anaconda3\python.exe'
$data = 'P:\userdata\raulh87\data'
$toolkit = Split-Path -Parent $PSScriptRoot
$folders = ($dirs | ForEach-Object { "'" + (Join-Path $DriveRoot "loo_results\$_") + "'" }) -join ' '
$import = "& '$python' '$PSScriptRoot\unpack_results.py' $folders --dataset EmoC --datafolder '$data' --replace"
Write-Host "After Google Drive finishes syncing, verify the import and then install the results:"
Write-Host "$import --dry-run"
Write-Host $import
foreach ($species in 'D','H') {
    foreach ($cls in 'dog','hum') {
        Write-Host "& '$python' '$toolkit\searchlight.py' --toolkit_dir '$toolkit' --datafolder '$data' --dataset EmoC --model basic-block --specie $species --radius 4 --mask_type b_GreyMatter2mmB --dis_method mahalanobis --rsa_method kendall --mah_fold stim-wise --rsa_class $cls --reps_group $Reps --steps_to_run 11.6 11.7 11.8 11.9 11.10"
    }
}

