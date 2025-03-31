import subprocess
import os
import re
import shutil
from androguard.misc import AnalyzeAPK
from androguard.core.bytecodes.axml import AXMLPrinter
from time import sleep

import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from tkinter import filedialog
import glob
from colorama import Fore, Style

# from axmlparserpy import axmlprinter

from APK_PROCESS import *

"""
Utilities, like major extractors for android .apk files

Author: [Michael Nhyk Ahimbisibwe]
System Name: [Mobile Phone Firmware Hardware Hacking Detection]
Model: [BSC HON YEAH PROJECT 1.0]
"""


class AppSelector:
    def __init__(self, filename, immediate_dir):
        # Initialize the main themed window
        self.immediate_dir = immediate_dir
        self.selected_name = None
        self.filename = filename
        self.root = ThemedTk(theme="arc")
        self.root.title("App Selector")
        self.root.configure(bg='#008B8B')
        self.root.geometry('400x220')
        self.root.iconbitmap("uj.ico")

        # Create a frame to contain the widgets
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(padx=20, pady=10, fill="both", expand=True)
        frame.configure(relief='solid')

        # Read lines from the file and populate the dropdown menu
        self.lines = self.read_file(self.filename)
        if not self.lines:
            return
        self.selected_string = tk.StringVar()
        self.combo = ttk.Combobox(frame, values=self.lines, font=("Arial", 12, 'bold'))
        self.combo.pack(pady=20, fill="x")
        # self.combobox_style = ttk.Style()
        # self.combobox_style.configure("Blue.TCombobox", background="blue")

        # Button to confirm the selected app
        self.button = ttk.Button(frame, text="Extract Manifest", command=self.on_select, style="Custom.TButton")
        self.button.pack(pady=10)
        # Style for the custom buttons
        style = ttk.Style()
        style.configure("TButton", foreground='Blue', font=("Arial", 10, 'normal'))  # Set button colors

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            frame,
            orient='horizontal',
            mode='indeterminate',
            length=280
        )

        self.progress_bar.pack(pady=10)

    def read_file(self, filename):
        """Reads the file and returns a list of lines."""
        if not os.path.exists(filename):
            print(f"{Fore.YELLOW}Apps List File Was Not Found{Style.RESET_ALL}")
            self.get_installed_packages()
            print(f"{Fore.GREEN}Installed Packages Saved{Style.RESET_ALL}")
            sleep(1)
        with open(filename, 'r') as file:
            lines = file.readlines()

        # self.root.destroy()
        return [line.strip() for line in lines]

    def _read_file(self, filename):
        """Reads the file and returns a list of lines."""
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
            return lines
        except FileNotFoundError:
            print(
                f"{Fore.YELLOW}Apps List File '{filename}' Was Not Found. Fetching installed packages.{Style.RESET_ALL}")
            return self.get_installed_packages()

    @classmethod
    def permissions(cls, app_name):
        try:
            xml_file_path = os.path.join(app_name, 'AndroidManifest.xml')
            perms_file_path = os.path.join(app_name, "getPermissions.txt")
            f_1 = open(xml_file_path, "r")
            f_2 = open(perms_file_path, "w")
            for line in f_1:
                if "android.permission" in str(line):
                    permission1 = re.search("android\.permission\.[\w.]+", line).group() + '\n'
                    print(permission1)
                    f_2.write(permission1)
            f_1.close()
            f_2.close()
        except FileNotFoundError:
            print(f"The Directory '{app_dir}' was not Found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def on_select(self):
        self.selected_name = self.combo.get()
        xml_file_path = os.path.join(self.immediate_dir, self.selected_name)
        if self.selected_name:
            self.progress_bar.start(interval='idle')
            print(Fore.GREEN + f"You selected: {self.selected_name}" + Fore.WHITE)
        else:
            print("You Did Not Select App Name")
        apk_extractor = APKExtractor(self.selected_name, self.immediate_dir)
        apk_extractor.extract_apk()
        apk_extractor.extract_AndroidManifest()
        apk_extractor.delete_saved_package()
        self.permissions(xml_file_path)
        self.progress_bar.stop()
        # self.progress_bar.pack_forget()
        self.root.destroy()

    def get_installed_packages(self):
        cmd = ["adb", "shell", "pm", "list", "packages"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            packages = result.stdout.split("\n")
            # Removing the 'package:' prefix and filtering out empty strings
            packages = [pkg.replace("package:", "").strip() for pkg in packages if pkg]

            # Save to a text file
            path_to_file = os.path.join(self.immediate_dir, "installed_packages.txt")
            with open(path_to_file, "w") as file:
                for pkg in packages:
                    file.write(pkg + '\n')

            return packages
        else:
            print("Error:", result.stderr)
            return []

    def run(self):
        self.root.mainloop()
        # messagebox.showinfo("Selected App", f"{self.selected_name}\nClose This Window to Continue.")
        # print(f"You selected: {self.selected_name}")
        # self.root.destroy()


class APKExtractor:
    def __init__(self, package_name_to_extract, phone_dir):
        self.phone_dir = phone_dir
        self.package_name_to_extract = package_name_to_extract  # Package name of the app to extract

    def extract_apk(self):
        result = subprocess.run(["adb", "shell", "pm", "path", self.package_name_to_extract], capture_output=True,
                                text=True)
        if result.returncode != 0:
            print(Fore.RED + f"Error finding APK path for {self.package_name_to_extract}." + Fore.WHITE)
            return

        apk_path = result.stdout.strip().split(":")[1]
        print(apk_path)
        # Extract the directory and filename from the apk_path
        apk_dir, apk_filename = os.path.split(apk_path)
        print(apk_dir)
        print(apk_filename)

        # Create a folder named after the package_name
        if not os.path.exists(self.package_name_to_extract):
            os.makedirs(self.package_name_to_extract)

        # Combine the folder path and apk_filename to construct the local file path
        local_apk_path = os.path.join(self.package_name_to_extract, apk_filename)

        subprocess.run(["adb", "pull", apk_path, local_apk_path])
        print(Fore.CYAN + f"APK for {self.package_name_to_extract} has been saved as {local_apk_path}" + Fore.WHITE)

    def _extract_AndroidManifest(self):
        """Extracts AndroidManifest.xml from the base APK using the 'aapt' tool."""

        # Define the path to the base.apk inside the given package directory.
        # This path is located in the same dir as the program and gets immediately deleted after
        # AndroidManifest.xml file is extracted and save. To clear the space
        # base_apk_path = os.path.join(self.package_name_to_extract, "base.apk")
        # global base_apk_path
        apk_files = glob.glob(os.path.join(self.package_name_to_extract, "*.apk"))

        # Ensure at least one .apk file is found
        if apk_files:
            base_apk_path = apk_files[0]
        else:
            print(Fore.RED + "No .apk file found in the directory." + Fore.WHITE)
            return
        # Define the directory where the extracted AndroidManifest.xml will be saved.
        # manifest_output_dir = os.path.join(self.saved_manifest, self.package_name_to_extract)
        manifest_output_dir = os.path.join(self.package_name_to_extract, "AndroidManifest.xml")

        # Ensure the manifest output directory exists.
        # os.makedirs(manifest_output_dir, exist_ok=True)
        if os.path.exists(manifest_output_dir):
            print(Fore.YELLOW + f"Folder '{manifest_output_dir}' Already Exists." + Fore.WHITE)
        else:
            try:
                os.makedirs(manifest_output_dir, exist_ok=True)
                print(Fore.GREEN + f"Folder '{manifest_output_dir}' successfully created." + Fore.WHITE)
            except Exception as e:
                print(Fore.RED + f"Failed to create folder '{Fore.WHITE}{manifest_output_dir}'. Error: {e}")

        # Command to extract AndroidManifest.xml using 'aapt' tool.
        aapt_command = ["aapt", "dump", "xmltree", base_apk_path, "AndroidManifest.xml"]
        # aapt_command = ["aapt", "dump", "badging", base_apk_path]

        # Execute the command and capture its output.
        result = subprocess.run(aapt_command, capture_output=True, text=True)

        # If the command executed successfully (return code 0), write the output to a file.
        if result.returncode == 0:
            manifest_output_file = os.path.join(manifest_output_dir, "AndroidManifest.xml")
            with open(manifest_output_file, "w") as manifest_file:
                manifest_file.write(result.stdout)
            print("{0}AndroidManifest.xml extracted and saved in-->{1}".format(Fore.BLUE, Fore.WHITE) +
                  f" {manifest_output_file}" + Fore.WHITE)
            messagebox.showinfo("Android Manifest", "AndroidManifest.xml Saved")
        else:
            print("{0}Error extracting {1}AndroidManifest.xml{2}".format(Fore.RED, Fore.YELLOW, Fore.WHITE))

    def extract_AndroidManifest(self):
        """Extracts AndroidManifest.xml from the base APK using the 'aapt' tool."""

        # Define the path to the base.apk inside the given package directory.
        # This path is located in the same dir as the program and gets immediately deleted after
        # AndroidManifest.xml file is extracted and save. To clear the space
        # base_apk_path = os.path.join(self.package_name_to_extract, "base.apk")
        # global base_apk_path
        apk_files = glob.glob(os.path.join(self.package_name_to_extract, "*.apk"))

        # Ensure at least one .apk file is found
        if apk_files:
            base_apk_path = apk_files[0]
        else:
            print(Fore.RED + "No .apk file found in the directory." + Fore.WHITE)
            return
        # Define the directory where the extracted AndroidManifest.xml will be saved.
        manifest_output_dir = os.path.join(self.phone_dir, self.package_name_to_extract)

        # Ensure the manifest output directory exists.
        # os.makedirs(manifest_output_dir, exist_ok=True)
        if os.path.exists(manifest_output_dir):
            print(Fore.YELLOW + f"Folder '{manifest_output_dir}' Already Exists." + Fore.WHITE)
        else:
            try:
                os.makedirs(manifest_output_dir, exist_ok=True)
                print(Fore.GREEN + f"Folder '{manifest_output_dir}' successfully created." + Fore.WHITE)
            except Exception as e:
                print(Fore.RED + f"Failed to create folder '{Fore.WHITE}{manifest_output_dir}'. Error: {e}")

        # Command to extract AndroidManifest.xml using 'aapt' tool.
        aapt_command = ["aapt", "dump", "xmltree", base_apk_path, "AndroidManifest.xml"]

        # Execute the command and capture its output.
        result = subprocess.run(aapt_command, capture_output=True, text=True)

        # If the command executed successfully (return code 0), convert the AXML output directly using androguard.
        if result.returncode == 0:
            axml_content = result.stdout.encode('utf-8')

            # Convert the AXML to XML using androguard
            axml_parser = AXMLPrinter(axml_content)
            xml_content = axml_parser.get_xml()

            # Save the converted XML
            xml_output_file = os.path.join(manifest_output_dir, "AndroidManifest.xml")
            with open(xml_output_file, "w", encoding='utf-8') as xml_file:
                xml_file.write(xml_content)

            print("{0}AndroidManifest.xml extracted, converted, and saved in-->{1}".format(Fore.BLUE, Fore.WHITE) +
                  f" {xml_output_file}" + Fore.WHITE)
            messagebox.showinfo("Android Manifest", "AndroidManifest.xml Saved")
        else:
            print("{0}Error extracting {1}AndroidManifest.xml{2}".format(Fore.RED, Fore.YELLOW, Fore.WHITE))

    def delete_saved_package(self):
        # Define the path to the package folder to be deleted.
        package_folder = self.package_name_to_extract
        # If the package folder exists, delete it. To reclaim space
        if os.path.exists(package_folder):
            shutil.rmtree(package_folder)
            print(f"{Fore.GREEN}Deleted saved package: {Fore.YELLOW}{package_folder}{Fore.WHITE}")
        else:
            print(Fore.YELLOW + f"No saved package found for:{Fore.WHITE} {self.package_name_to_extract}")


class APKDetailsExtractor:
    def __init__(self, apk_filepath):
        self.apk_analysis, self.dex, self.dx_data = AnalyzeAPK(apk_filepath)

    def extract(self):
        # Details collected from the APK are generally the same as those collected from the AndroidManifest.xml
        return {
            "app_name": self.get_app_name(),
            "app_package": self.get_package(),
            "app_icon_data": self.get_icon(),
            "allowed_permissions": self.get_permissions(),
            "defined_activities": self.get_activities(),
            "code_for_android_version": self.get_android_version_code(),
            "name_for_android_version": self.get_android_version_name(),
            "minimum_sdk": self.get_min_sdk_version(),
            "maximum_sdk": self.get_max_sdk_version(),
            "targeted_sdk": self.get_target_sdk_version(),
            "actual_target_sdk": self.get_effective_target_sdk_version()
        }

    def get_app_name(self):
        return self.apk_analysis.get_app_name()

    def get_package(self):
        return self.apk_analysis.get_package()

    def get_icon(self):
        return self.apk_analysis.get_app_icon()

    def get_permissions(self):
        return self.apk_analysis.get_permissions()

    def get_activities(self):
        return self.apk_analysis.get_activities()

    def get_android_version_code(self):
        return self.apk_analysis.get_androidversion_code()

    def get_android_version_name(self):
        return self.apk_analysis.get_androidversion_name()

    def get_min_sdk_version(self):
        return self.apk_analysis.get_min_sdk_version()

    def get_max_sdk_version(self):
        return self.apk_analysis.get_max_sdk_version()

    def get_target_sdk_version(self):
        return self.apk_analysis.get_target_sdk_version()

    def get_effective_target_sdk_version(self):
        return self.apk_analysis.get_effective_target_sdk_version()


def save_apk_details_to_txt(details, filename="apk_details.txt"):
    with open(filename, 'w') as file:
        # App basic details
        print('Saving apk_details')
        file.write(f"**{details['app_name']} App Details**\n\n")
        file.write(f"- App Name: {details['app_name']}\n")
        file.write(f"- Package Name: {details['app_package']}\n")
        file.write(f"- App Icon Data: {details['app_icon_data']}\n")
        file.write(f"- Version Code: {details['code_for_android_version']}\n")
        file.write(f"- Version Name: {details['name_for_android_version']}\n")
        file.write(f"- Minimum SDK: {details['minimum_sdk']}\n")
        file.write(f"- Targeted SDK: {details['targeted_sdk']}\n")
        file.write(f"- Actual Target SDK: {details['actual_target_sdk']}\n")
        if details['maximum_sdk']:
            file.write(f"- Maximum SDK: {details['maximum_sdk']}\n")
        else:
            file.write("- Maximum SDK: Not Specified\n")

        # Permissions
        file.write("\nPermissions:\n\n")
        for permission in details['allowed_permissions']:
            file.write(f"- {permission}\n")

        # Defined Activities
        file.write("\nDefined Activities:\n\n")
        for activity in details['defined_activities']:
            file.write(f"- {activity}\n")
        print('{0}Saving apk_details Completes{1}'.format(Fore.CYAN, Fore.WHITE))


# details = APKDetailsExtractor('ackman.placemarks.apk')


# save_apk_details_to_txt(details.extract())

def save_allowed_permissions_and_activities(details_src: APKDetailsExtractor,
                                            allowed_per_final: str, activities: str):
    app_name = None
    try:
        app_name = details_src.get_app_name()
    except subprocess.CalledProcessError as e:
        print(f"Error Extracting App Name: {e}")
    try:

        allowed_per = details_src.get_permissions()
        with open(allowed_per_final, 'w') as file:
            # Convert the list to a string and join its elements with newline characters
            permissions_str = "\n".join(allowed_per)
            # file.write(f"[Allowed Permission for {app_name}]\n")
            file.write(permissions_str)
        print(f"Allowed Permissions Saved {allowed_per_final}")
    except subprocess.CalledProcessError as e:
        print(f"Error Extracting Permissions: {e}")

    try:
        _activities = details_src.get_activities()
        with open(activities, 'w') as file:
            # Convert the list to a string and join its elements with newline characters
            activities_str = "\n".join(_activities)
            file.write(f"[Activities for {app_name}]\n")
            file.write(activities_str)
        print(f"Activities Saved {activities}")
    except subprocess.CalledProcessError as e:
        print(f"Error Extracting Activities: {e}")

# save_allowed_permissions_and_activities(details, "allowed_permissions.txt", "app_activities.txt")

