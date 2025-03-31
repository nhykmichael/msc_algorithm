import os
import zipfile
import tarfile
import logging
from colorama import Fore, Back, Style
import tkinter.messagebox as messagebox
import json

"""
Utilities, System File Extractors ... etc

Author: [Michael Nhyk Ahimbisibwe]
System Name: [Mobile Phone Firmware Hardware Hacking Detection]
Model: [BSC HON YEAH PROJECT 1.0]
"""


def color():
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    WHITE = "\033[0m"  # "\033[97m"

    color_list = [RED, GREEN, YELLOW, BLUE, WHITE]
    return color_list


COLOR = color()


class FileExtractor:

    @staticmethod
    def unzip_my_file(my_zip_file, root_file, dir_file):
        """
        Unpacks a zip file.

        param my_zip_file: str - the zip file we want to unpack.
        param output_folder: str - the folder where we want to put the unpacked files.
        """

        try:

            # Check if the root output folder exists
            if not os.path.exists(root_file):
                # Open the zip file in read mode
                with zipfile.ZipFile(my_zip_file, 'r') as my_zip:
                    # Unpack all files to the output folder
                    my_zip.extractall(dir_file)
                print("Files extracted successfully.")
            else:
                COLOR = color()
                print(
                    f"{COLOR[1]}ABD Files Saved in: {COLOR[2]}'{root_file}'{COLOR[0]} already exists. Skipping "
                    f"extraction.{COLOR[4]}")
        except Exception as error:
            # If something goes wrong, log the error message
            logging.exception(str(error))

    @staticmethod
    def extract_my_tar_file(my_tar_file, output_folder):
        """
        Unpacks a tar file.

        param my_tar_file: str - the tar file we want to unpack.
        param output_folder: str - the folder where we want to put the unpacked files.
        """
        try:
            if os.path.isfile(my_tar_file) and os.access(my_tar_file, os.R_OK):
                my_tar = tarfile.open(my_tar_file)
                my_tar.extractall(output_folder)
                my_tar.close()
        except Exception as error:
            logging.exception(str(error))


k = FileExtractor
k.unzip_my_file("C:\\Android\\zipped\\platform-tools_r34.0.3-windows.zip", "C:\\Android\\platform-tools", "C:\\Android")


def create_folder(path):
    """Create a new folder.

    If the folder already exists, do nothing.

    Arguments:
    path -- The location where the folder should be created. This should be a full path.
    """
    if os.path.exists(path):
        print(f"Folder '{path}' Already Exists.")
    else:
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Folder '{path}' successfully created.")
        except Exception as e:
            print(f"Failed to create folder '{path}'. Error: {e}")


def create_temp(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


THEM = ["#6BD0FF", '#F4D03F', '#2ECC71']  # Red, Yellow, Green


def read_text_file(filename):
    """
        Read lines from a text file and return them as a list of options.

        Args:
            filename (str): The path to the text file to be read.

        Returns:
            list: A list containing the lines from the text file, with leading and trailing whitespace removed.
                  If the file is not found or an error occurs during reading, an empty list is returned.
        """
    options = []  # Initialize an empty list to store the lines from the text file

    try:
        with open(filename, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespace, and add the line to the options list
                options.append(line.strip())
    except FileNotFoundError:
        print(Fore.RED + f"File not Found: {filename}" + Fore.WHITE)
        # messagebox.showerror("Error", f"The file '{filename}' does not exist.")
    except Exception as e:
        print(f"{COLOR[0]}An error occurred: {str(e)}{COLOR[4]}")

    return options


class GlobalReport:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.results = []
        self.file_path = os.path.join(folder_path, 'global_report.json')

    def append_result(self, action, status, result=None):
        self.results.append({'action': action, 'status': status, 'result': result})
        try:
            self.save_to_file()
        except Exception as e:
            print(f"{COLOR[0]}An error occurred while saving JASON: {str(e)}{COLOR[4]}")

    def save_to_file(self):
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.results, file, indent=2)
        except FileNotFoundError:
            print(Fore.RED + f"File not Found: {self.folder_path}" + Fore.WHITE)
            with open(".", 'w') as file:
                json.dump(self.results, file, indent=2)

    def load_from_file(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                self.results = json.load(file)

    def show_report(self):
        for result in self.results:
            print(f"Action: {result['action']}, Status: {result['status']}, Result: {result['result']}")