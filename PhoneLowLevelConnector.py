import logging

# import usb.core
# import usb.util
# import bluetooth
import socket
import sys
import os
import subprocess
from usable import *

__author__ = 'MN Ahimbisibwe'

global_text_green = 'Waiting for device >>'
global_text_red = "No Action Made"


# COLOR = color()  # color_list = [RED, GREEN, YELLOW, BLUE, WHITE]

def check_adb():
    """Checks if adb is available and prints its version."""
    try:
        result = subprocess.run(['adb', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"{COLOR[3]}{result.stdout.decode()}{COLOR[4]}")
        return True  # Return True if adb is available
    except FileNotFoundError:
        print(f"{COLOR[0]}adb is not found in your system's PATH. Please install Android SDK and ensure adb is in the "
              f"PATH.{COLOR[4]}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"{COLOR[0]}Error running adb:{COLOR[4]}", e)
        return False


check_adb()


def get_adb_devices():
    try:
        res = subprocess.check_output(['adb', 'devices']).decode('utf8')
        lines = res.strip().split('\n')
        devices = [line.split('\t')[0] for line in lines[1:] if line.split('\t')[1] == 'device']
        if not devices:
            logging.error(" No devices are connected.")
        return devices
    except Exception as e:
        print(f"{COLOR[0]}An error occurred: {COLOR[4]}{str(e)}")
        return []


def stop_adb_server():
    # Execute the command to stop the adb server
    result = subprocess.run(['adb', 'kill-server'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print('ADB server stopped successfully.')
    else:
        print(f'Failed to stop ADB server. Error: {result.stderr.decode()}')


def start_adb_server():
    # Execute the command to start the adb server
    result = subprocess.run(['adb', 'start-server'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check the result
    if result.returncode == 0:
        print('ADB server started successfully.')
    else:
        print(f'Failed to start ADB server. Error: {result.stderr.decode()}')


class MobilePhoneLowLevelConnector:
    def __init__(self, adb_path: str, target_name=None):
        self.target_name = target_name
        self.adb_path = adb_path

    def detect_device(self):
        # List connected adb devices
        result = subprocess.run([self.adb_path, 'devices'], capture_output=True, text=True)
        # result = subprocess.check_output(['adb', 'devices']).decode('utf8')

        devices = result.stdout.split('\n')[1:]

        # Iterate over the connected devices and find the target device
        for device in devices:
            if device:
                # Device string format is "{device_name}\t{device_status}", we're interested in the name
                device_name = device.split('\t')[0]

                if self.target_name in device_name:
                    global global_text_green
                    global_text_green = f"ADB device {COLOR[3]}{device_name.upper()}{COLOR[4]} found."
                    print(global_text_green)
                    return True, global_text_green

        global global_text_red
        global_text_red = "ADB device not found."
        print(global_text_red)
        return False, global_text_red

    def connect_phone_adb(self):
        if not self.detect_device():
            global global_text_red
            global_text_red = "ADB connection established successfully."

            print(global_text_red)
            return False

        # Now that the device is detected, you can run adb commands as needed For example, to pull a file from the
        # device: result = subprocess.run([self.adb_path, '-s', self.target_name, 'pull', '/path/to/file/on/device',
        # '/path/to/destination/on/pc'])
        else:
            global global_text_green
            global_text_green = f"{COLOR[1]}ADB connection established successfully.{COLOR[4]}"

            print(global_text_green)
        return True
