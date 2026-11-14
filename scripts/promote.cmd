@echo off
rem Promote a finished fine-tune job: promote <job folder>
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0promote_version.ps1" -Job %*
