Option Explicit

' Windowless launcher. Run the existing Python environment with stdout/stderr
' redirected to a local log; the hidden shell exits when the application exits.
Dim shell, files, root, mode, args, logDir, logPath, stamp, command, code, failure
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
mode = "start"
If WScript.Arguments.Count > 0 Then mode = LCase(WScript.Arguments(0))
Select Case mode
    Case "start": args = "start"
    Case "collect": args = "collect --vehicle unknown --camera-view unknown"
    Case "kodiak": args = "collect --vehicle KODIAK --camera-view unknown --fps 15 --max-width 512"
    Case "check": args = "--help"
    Case Else: WScript.Quit 2
End Select

logDir = files.BuildPath(root, "logs")
If Not files.FolderExists(logDir) Then files.CreateFolder logDir
stamp = Year(Now) & Pad(Month(Now)) & Pad(Day(Now)) & "_" & Pad(Hour(Now)) & Pad(Minute(Now)) & Pad(Second(Now))
logPath = files.BuildPath(logDir, "launcher_" & mode & "_" & stamp & ".log")
shell.Environment("PROCESS")("PYTHONUTF8") = "1"
command = Quote(shell.ExpandEnvironmentStrings("%ComSpec%")) & " /d /s /c " & _
          Quote("python -m wardog_ad " & args & " >> " & Quote(logPath) & " 2>&1")
On Error Resume Next
code = shell.Run(command, 0, True)
failure = Err.Description
If Err.Number <> 0 Then code = 1
On Error GoTo 0
If code <> 0 And mode <> "check" Then
    MsgBox "WARDOGS could not exit normally (code " & code & ")." & vbCrLf & _
           "See log: " & logPath & vbCrLf & failure, vbExclamation, "WARDOGS Copilot"
End If
WScript.Quit code

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function

Function Pad(value)
    Pad = Right("0" & value, 2)
End Function
