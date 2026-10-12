@echo off
rem ComfyUI Missing Models installer (Windows). Double-click it; it can run on its own, no clone needed.
rem It finds ComfyUI, installs or updates this extension in custom_nodes, and can save a Hugging Face token
rem and a Civitai API key.
setlocal EnableExtensions EnableDelayedExpansion
title ComfyUI Missing Models - installer

set "NAME=ComfyUI-MissingModels"
if not defined MM_REPO set "MM_REPO=https://github.com/skealeye/ComfyUI-MissingModels.git"
if not defined MM_ZIP set "MM_ZIP=https://github.com/skealeye/ComfyUI-MissingModels/archive/refs/heads/main.zip"

echo.
echo  ==============================================
echo    ComfyUI Missing Models - installer
echo  ==============================================
echo.
echo  Adds a "Missing models" button to ComfyUI that downloads a workflow's
echo  missing models into the right models folders.
echo.

rem ---- 1. find ComfyUI's custom_nodes folder -------------------------------------------------
set "CN="
rem run from inside custom_nodes\ComfyUI-MissingModels, or dropped into a ComfyUI folder
call :try "%~dp0.."
call :try "%~dp0"
rem ComfyUI Desktop app: its config names the base path
set "DESKTOP="
if exist "%APPDATA%\ComfyUI\config.json" (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "try { (Get-Content -Raw $env:APPDATA'\ComfyUI\config.json' | ConvertFrom-Json).basePath } catch {}"`) do set "DESKTOP=%%i"
)
if defined DESKTOP call :try "!DESKTOP!"
call :try "%USERPROFILE%\Documents\ComfyUI"
call :try "%USERPROFILE%\Desktop\ComfyUI_windows_portable"
call :try "%USERPROFILE%\Downloads\ComfyUI_windows_portable"
call :try "%USERPROFILE%\ComfyUI_windows_portable"
call :try "%USERPROFILE%\ComfyUI"
call :try "C:\ComfyUI_windows_portable"
call :try "C:\ComfyUI"
call :try "D:\ComfyUI_windows_portable"
call :try "D:\ComfyUI"

if not defined CN goto ask
echo  Found ComfyUI's custom_nodes folder:
echo    !CN!
echo.
set "ANS="
set /p "ANS=  Install there? [Y/n]: "
if /i "!ANS!"=="n" goto ask
if /i "!ANS!"=="no" goto ask
goto install

:ask
set "CN="
echo.
echo  Where is ComfyUI? Paste its folder, for example:
echo    C:\ComfyUI_windows_portable           (portable)
echo    C:\Users\you\Documents\ComfyUI        (Desktop app)
echo    C:\ComfyUI                            (git install)
echo  Tip: in Explorer, right-click the folder and choose "Copy as path".
echo.
set "P="
set /p "P=  ComfyUI folder (blank to quit): "
if not defined P goto quit
set "P=!P:"=!"
call :try "!P!"
if defined CN goto install
echo.
echo  No custom_nodes folder found in "!P!". Try the folder that holds main.py,
echo  or the portable folder that holds run_nvidia_gpu.bat.
goto ask

rem ---- 2. install or update ---------------------------------------------------------------------
:install
set "DEST=!CN!\%NAME%"
echo.
set "HASGIT="
where git >nul 2>&1 && set "HASGIT=1"

if exist "!DEST!\.git\" if defined HASGIT (
  echo  Already installed with git: updating...
  git -C "!DEST!" pull --ff-only || goto failed
  goto token
)
if exist "!DEST!\__init__.py" (
  echo  Already installed: updating the files...
  goto zip
)
if defined HASGIT (
  echo  Downloading with git...
  git clone --depth 1 "%MM_REPO%" "!DEST!" || goto failed
  goto token
)
echo  Git not found: downloading a zip instead...

:zip
set "TMPD=%TEMP%\%NAME%-install-%RANDOM%"
mkdir "!TMPD!" >nul 2>&1
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -UseBasicParsing '%MM_ZIP%' -OutFile $env:TMPD'\src.zip'; Expand-Archive $env:TMPD'\src.zip' $env:TMPD; exit 0 } catch { Write-Host $_; exit 1 }" || goto failed
set "SRC="
for /d %%d in ("!TMPD!\%NAME%*") do set "SRC=%%d"
if not defined SRC goto failed
rem copies over the old files; your token files and config.json are not in the zip, so they stay
robocopy "!SRC!" "!DEST!" /E /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto failed
rmdir /s /q "!TMPD!" >nul 2>&1

rem ---- 3. optional Hugging Face token -------------------------------------------------------
:token
echo.
echo  Installed to:
echo    !DEST!
echo.
if defined HF_TOKEN (
  echo  Your HF_TOKEN environment variable is set: gated models will use it.
  goto civitai
)
if exist "!DEST!\hf_token.txt" (
  echo  A Hugging Face token is already saved in hf_token.txt.
  goto civitai
)
echo  Most models download without an account. Some ^(for example FLUX.1-dev^) are
echo  "gated": they need a free Hugging Face account, a click on "Agree" on the
echo  model's page, and a token.
echo.
set "ANS="
set /p "ANS=  Add a Hugging Face token now? [y/N]: "
if /i not "!ANS!"=="y" if /i not "!ANS!"=="yes" (
  echo  Skipped. You can run this installer again later to add one.
  goto civitai
)
echo.
echo  1. Open https://huggingface.co/settings/tokens
echo  2. Create a token of type "Read" and copy it ^(it starts with hf_^)
echo.
set "TOK="
set /p "TOK=  Paste the token: "
if not defined TOK (
  echo  No token entered: skipped.
  goto civitai
)
set "TOK=!TOK: =!"
>"!DEST!\hf_token.txt" echo(!TOK!
echo  Saved to hf_token.txt ^(stays on this PC; only ever sent to huggingface.co^).

rem ---- 4. optional Civitai API key --------------------------------------------------------------
:civitai
echo.
if defined CIVITAI_API_TOKEN (
  echo  Your CIVITAI_API_TOKEN environment variable is set: Civitai downloads will use it.
  goto done
)
if exist "!DEST!\civitai_token.txt" (
  echo  A Civitai API key is already saved in civitai_token.txt.
  goto done
)
echo  The extension can also find and download models from Civitai. Many work
echo  without an account; some need a free Civitai account and an API key.
echo.
set "ANS="
set /p "ANS=  Add a Civitai API key now? [y/N]: "
if /i not "!ANS!"=="y" if /i not "!ANS!"=="yes" (
  echo  Skipped. You can run this installer again later to add one.
  goto done
)
echo.
echo  1. Open https://civitai.com/user/account
echo  2. Under "API Keys", click "Add API key", name it, and copy the key
echo.
set "TOK="
set /p "TOK=  Paste the key: "
if not defined TOK (
  echo  No key entered: skipped.
  goto done
)
set "TOK=!TOK: =!"
>"!DEST!\civitai_token.txt" echo(!TOK!
echo  Saved to civitai_token.txt ^(stays on this PC; only ever sent to civitai.com^).

:done
echo.
echo  Done. Restart ComfyUI, then look for the "Missing models" button in the
echo  top bar ^(also under Extensions in the menu^).
echo.
pause
exit /b 0

:failed
echo.
echo  Something went wrong ^(see the message above^). Check your internet connection
echo  and try again, or install by hand: see the README at
echo  https://github.com/skealeye/ComfyUI-MissingModels
echo.
pause
exit /b 1

:quit
echo  Nothing installed.
pause
exit /b 1

rem ---- :try <folder>: set CN to its custom_nodes folder, if it has one (first match wins) ----
:try
if defined CN exit /b 0
set "T=%~f1"
if "!T:~-1!"=="\" set "T=!T:~0,-1!"
for %%n in ("!T!") do if /i "%%~nxn"=="custom_nodes" if exist "!T!\" (set "CN=!T!" & exit /b 0)
if exist "!T!\custom_nodes\" if exist "!T!\main.py" (set "CN=!T!\custom_nodes" & exit /b 0)
if exist "!T!\ComfyUI\custom_nodes\" (set "CN=!T!\ComfyUI\custom_nodes" & exit /b 0)
rem Desktop app base path: custom_nodes and models, no main.py
if exist "!T!\custom_nodes\" if exist "!T!\models\" (set "CN=!T!\custom_nodes" & exit /b 0)
exit /b 0
