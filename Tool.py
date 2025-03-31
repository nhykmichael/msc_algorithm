from enum import Enum


class WarningCode(Enum):
    SUCCESS = 0
    PASS = 1
    ERROR = -1


def read_text_file(filename):
    options = []  # Initialize an empty list to store the lines from the text file

    try:
        with open(filename, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespace, and add the line to the options list
                options.append(line.strip())
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return options
