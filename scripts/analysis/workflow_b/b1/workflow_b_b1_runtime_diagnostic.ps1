$ErrorActionPreference = 'Continue'
$root = 'D:\udsc2026'
$out = Join-Path $root 'reports\task1\workflow_b_b1_runtime_diag'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'scripts\analysis\workflow_b_b1_execute_recovered.py'
$stdout = Join-Path $out 'stdout.log'; $stderr = Join-Path $out 'stderr.log'; $telemetry = Join-Path $out 'telemetry.csv'
Remove-Item -Force -ErrorAction SilentlyContinue $stdout,$stderr,$telemetry
$cmd = [ordered]@{ working_directory=$root; executable=$python; arguments=@('-u',$script); command_line=('"'+$python+'" -u "'+$script+'"'); environment=@{OMP_NUM_THREADS=$env:OMP_NUM_THREADS;OMP_THREAD_LIMIT=$env:OMP_THREAD_LIMIT;MKL_NUM_THREADS=$env:MKL_NUM_THREADS;OPENBLAS_NUM_THREADS=$env:OPENBLAS_NUM_THREADS;NUMEXPR_NUM_THREADS=$env:NUMEXPR_NUM_THREADS}; script_sha256=(Get-FileHash -Algorithm SHA256 $script).Hash; wrapper_timeout='NONE' }
$cmd | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $out 'command.json')
function SysState {
  $os=Get-CimInstance Win32_OperatingSystem; $cs=Get-CimInstance Win32_ComputerSystem; $pf=Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
  [ordered]@{timestamp=(Get-Date).ToString('o');logical_cpu=$env:NUMBER_OF_PROCESSORS;physical_cpu=($cs.NumberOfProcessors);total_ram_bytes=[int64]$cs.TotalPhysicalMemory;available_ram_bytes=[int64]$os.FreePhysicalMemory*1KB;committed_virtual_bytes=([int64]$os.TotalVirtualMemorySize*1KB-[int64]$os.FreeVirtualMemory*1KB);commit_limit_bytes=[int64]$os.TotalVirtualMemorySize*1KB;pagefile_allocated_bytes=[int64](($pf|Measure-Object -Property AllocatedBaseSize -Sum).Sum)*1MB;pagefile_current_usage_bytes=[int64](($pf|Measure-Object -Property CurrentUsage -Sum).Sum)*1MB;pagefile_peak_usage_bytes=[int64](($pf|Measure-Object -Property PeakUsage -Sum).Sum)*1MB;repository_disk_free_bytes=(Get-PSDrive -Name D).Free;temp_path=$env:TEMP;temp_disk_free_bytes=(Get-PSDrive -Name ([IO.Path]::GetPathRoot($env:TEMP).TrimEnd(':\'))).Free;lightgbm_n_jobs=1}
}
SysState | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $out 'system_start.json')
'timestamp,elapsed_seconds,child_pid,working_set_bytes,peak_working_set_bytes,private_memory_bytes,virtual_memory_bytes,thread_count,available_ram_bytes,total_ram_bytes,pagefile_current_usage_bytes,pagefile_allocated_bytes,repository_disk_free_bytes,temp_disk_free_bytes' | Set-Content -Encoding UTF8 $telemetry
$start=Get-Date
$p=Start-Process -FilePath $python -ArgumentList @('-u',$script) -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
while(-not $p.HasExited) {
  Start-Sleep -Seconds 2; $p.Refresh(); $s=SysState
  $line=@($s.timestamp,((Get-Date)-$start).TotalSeconds,$p.Id,$p.WorkingSet64,$p.PeakWorkingSet64,$p.PrivateMemorySize64,$p.VirtualMemorySize64,$p.Threads.Count,$s.available_ram_bytes,$s.total_ram_bytes,$s.pagefile_current_usage_bytes,$s.pagefile_allocated_bytes,$s.repository_disk_free_bytes,$s.temp_disk_free_bytes) -join ','
  Add-Content -Encoding UTF8 $telemetry $line
}
$end=Get-Date; $endstate=SysState; $endstate | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $out 'system_end.json')
$events=@(); foreach($log in 'Application','System') { try { $events += Get-WinEvent -FilterHashtable @{LogName=$log;StartTime=$start.AddMinutes(-1);EndTime=$end.AddMinutes(1)} -ErrorAction Stop | Where-Object {$_.Id -in 1000,1001,2004 -or $_.ProviderName -match 'Error|Resource'} | Select-Object TimeCreated,ProviderName,Id,LevelDisplayName,@{N='MessageExcerpt';E={$_.Message.Substring(0,[Math]::Min(1000,$_.Message.Length))}} } catch {} }
$events | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $out 'windows_events.json')
$diag=[ordered]@{status='PASS';child_pid=$p.Id;exit_code=$p.ExitCode;process_start=$start.ToString('o');process_end=$end.ToString('o');elapsed_seconds=(($end-$start).TotalSeconds);command=$cmd.command_line;known_orchestration_timeout_seconds='UNKNOWN';wrapper_timeout='NONE';stdout=$stdout;stderr=$stderr;telemetry=$telemetry}
$diag | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $out 'runtime_diagnostic.json')
Write-Output ($diag|ConvertTo-Json -Compress)
