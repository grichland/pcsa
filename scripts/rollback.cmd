@echo off
rem Back one live-player version, or: rollback -List, rollback -To neural-<stamp>
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rollback_version.ps1" %*
