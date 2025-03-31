import subprocess
import json
from colorama import Fore, Style
import os
import shutil


def pull_apk_by_short_name(short_name, local_path='TempApps', _app_list=None):
    """
    Pulls an APK file using the shorter app name by finding its corresponding longer name.

    Parameters:
    - short_name (str): The shorter app name.
    - local_path (str): The local directory where the APK file will be saved.
    - app_list (list): List of strings with longer app names and their paths.

    Returns:
    - bool: True if the operation was successful, False otherwise.
    """

    try:
        _app_list = get_app_list_from_json_file()
    except FileNotFoundError as e:
        print(f"{Fore.RED}An error occurred: {str(e)}{Fore.Style.RESET_ALL}")
        exit(1)

    # Find the corresponding longer app name
    longer_name = None
    for app_entry in _app_list:
        if short_name in app_entry:
            longer_name = app_entry.split('.apk')[0] + '.apk'  # Extract until '.apk' and append '.apk'
            print(longer_name)
            break

    if not longer_name:
        print(Fore.RED, f"Error: Short name '{short_name}' not found in the app list.", Style.RESET_ALL)
        print(Fore.YELLOW, "Looking in Testing Dir ", Style.RESET_ALL)
        return False

    # Create the local directory if it does not exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    command = ['adb', 'pull', longer_name, local_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, stderr = process.communicate()

    # Check for errors
    if stderr:
        print(f"Error: {stderr}")
        return False

    print(f"APK file for {short_name} saved to: {local_path}")
    return True


def get_app_list_from_json_file() -> list:
    try:
        _app_list = os.path.join(".", 'installed_packages.json')
        with open(_app_list, 'r') as f:
            data = json.load(f)
        options = data

    except FileNotFoundError:
        print(Fore.RED, "Jason App List Read Fail", Style.RESET_ALL)
        options = ['All Apps']

    return options


def delete_saved_package(package_name_that_extracted):
    if os.path.exists(package_name_that_extracted):
        shutil.rmtree(package_name_that_extracted)
        print(f"{Fore.GREEN}Deleted saved package: {Fore.YELLOW}{package_name_that_extracted}", Style.RESET_ALL)
    else:
        print(Fore.YELLOW + f"No saved package found for:{Fore.WHITE} {package_name_that_extracted}")


# shrt_name = "com.samsung.rcs"
# shrt_name2 = "com.sec.location.nfwlocationprivacy"
# app_list = get_app_list_from_json_file()
# success = pull_apk_by_short_name(shrt_name)
# success2 = pull_apk_by_short_name(shrt_name2)
# print(app_list)


class APK_PROCESS:
    def __init__(self, where_to_save):
        self.where_to_save = where_to_save

    def get_list_of_installed(self):
        # Run adb shell command to get a list of installed packages
        command = ["adb", "shell", "pm", "list", "packages", "-f"]  # -f flag to get the path to the installed packages
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = result.communicate()

        # check for errors
        if stderr:
            print(f"{Fore.RED}Error: {stderr}{Fore.WHITE}")
            exit(1)

        package_lines = stdout.strip().split("\n")
        package_names = [line.split(":")[1] for line in package_lines]

        # If the command executed successfully (return code 0), write the output to a file.
        if result.returncode == 0:
            # Save the output to a file.
            output_file = os.path.join(self.where_to_save, "installed_packages.json")
            with open(output_file, "w") as file:
                json.dump(package_names, file, indent=2)
            print(f"{Fore.BLUE}Installed Packages List Saved To: {Fore.WHITE}{self.where_to_save}{Fore.WHITE}")
        else:
            print(f"{Fore.RED}Error extracting {Fore.YELLOW}Installed Packages List{Fore.WHITE}")
        print(f"{Fore.BLUE}List of installed apps saved to {Fore.WHITE}{self.where_to_save}{Fore.WHITE}")

    def extract_shorter_names_from_saved(self):
        # Read the list of installed packages from the JSON file and extract the package names using string that
        # follow .apk= e.g. package:/data/app/com.google.android.apps.docs.editors.docs-1/base.apk=com.google.android
        # .apps.docs.editors.docs extract com.google.android.apps.docs.editors.docs
        saved_jason = os.path.join(self.where_to_save, "installed_packages.json")
        if not os.path.exists(saved_jason):
            print(f"{Fore.RED}Apps List File not Found: {saved_jason}{Fore.WHITE}")
            exit(1)

        with open(saved_jason, "r") as file:
            full_names = json.load(file)

        package_names = [name.split(".apk=")[1] for name in full_names]
        # Save the list to a JSON file
        try:
            save_new_jason = os.path.join(self.where_to_save, "installed_apps.json")
            with open(save_new_jason, "w") as file:
                json.dump(package_names, file, indent=4)
        except FileNotFoundError as e:
            print(f"{Fore.RED}An error occurred: {str(e)}{Fore.Style.RESET_ALL}")
            exit(1)

    def run(self):
        self.get_list_of_installed()
        self.extract_shorter_names_from_saved()

    def _pull_apk(self, device_path, local_path='TempApps'):
        """
        Pulls an APK file from the specified path on the Android device to the local directory.

        Parameters:
        - device_path (str): The path to the APK file on the Android device.
        - local_path (str): The local directory where the APK file will be saved.

        Returns:
        - bool: True if the operation was successful, False otherwise.
        """

        # Run adb pull command
        # create the local directory if it does not exist
        if not os.path.exists(local_path):
            os.makedirs('TempApps')

        command = ['adb', 'pull', device_path, local_path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, stderr = process.communicate()

        # Check for errors
        if stderr:
            print(f"Error: {stderr}")
            return False

        print(f"{Fore.BLUE}APK file saved to: {Fore.YELLOW}{local_path}", Fore.Style.RESET_ALL)
        return True

# apps = APK_PROCESS(".")  # where to save the output
# apps.get_list_of_installed()
# apps.extract_shorter_names_from_saved()
