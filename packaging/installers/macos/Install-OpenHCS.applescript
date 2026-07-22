on run
    try
        set appBundle to POSIX path of (path to me)
        set resourcesDirectory to appBundle & "Contents/Resources/"
        set bootstrapPath to resourcesDirectory & "install-openhcs.sh"
        set contractPath to resourcesDirectory & "installer_contract.json"
        set productName to do shell script "/usr/bin/plutil -extract product_name raw -o - " & quoted form of contractPath
        set installerTitle to productName & " Installer"

        display dialog "This installs or updates " & productName & " in a private environment for your macOS account. No existing Python or administrator password is required." with title installerTitle buttons {"Cancel", "Install / Update"} default button "Install / Update" cancel button "Cancel" with icon note

        set progress total steps to -1
        set progress description to "Installing " & productName
        set progress additional description to "Downloading Python and the application. This can take several minutes."

        with timeout of 3600 seconds
            do shell script quoted form of bootstrapPath & space & quoted form of contractPath
        end timeout

        set progress total steps to 1
        set progress completed steps to 1
        set progress additional description to "Installation complete."
        display dialog productName & " is ready in Applications and on your Desktop." with title installerTitle buttons {"Done"} default button "Done" with icon note
    on error errorMessage number errorNumber
        set progress total steps to 0
        if errorNumber is -128 then return
        display alert "Installation failed" message "Review the durable installer log in your Library/Logs folder.\n\n" & errorMessage as critical buttons {"OK"} default button "OK"
    end try
end run
