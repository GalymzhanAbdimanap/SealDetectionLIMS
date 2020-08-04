:: 
:: Runs script for detecting stamps.
:: Creates new log with current date YYYYMMDD-HHMMSS.
::
::	Copyright (c) 2020 IDET.kz
:: Written by Daniyar Nurseitov.
::

@ECHO OFF

::------------------------------------------------------------------------------
:: Log file name
::------------------------------------------------------------------------------
::
:: Format command syntax:
:: DATE:~6,4%
::       ^-----skip 6 characters
::         ^---keep 4 characters

:: Time (pad digits with leading zeros)
set t=%time: =0%	

set hour=%t:~0,2%

set min=%t:~3,2%

set secs=%t:~6,2%

:: Date (pad digits with leading zeros)
set d=%date: =0%

set year=%d:~-4%

set month=%d:~-7,2%

set day=%d:~-11,2%

set log_file_name=%year%%month%%day%-%hour%%min%%secs%-detect_stamp.log
echo Created new log file name: %log_file_name%

::------------------------------------------------------------------------------
:: Run script with selected verbose mode [DEBUG|INFO|WARNING|ERROR|CRITICAL]
::------------------------------------------------------------------------------
::env\Scripts\python.exe app_assync.py
C:\Users\Daniyar\.conda\envs\py36-tf15\python.exe detect_stamp_async.py -f %log_file_name% -l INFO