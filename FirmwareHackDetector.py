from InterfaceHub import *
from abc import ABC, abstractmethod
import logging
import pandas as pd
import re
# from datetime import datetime, date
import datetime
from serial.tools import list_ports

from PhoneLowLevelConnector import *
from APKManifestoExtractor import *
import time

# from jnius import autoclass
import matplotlib.pyplot as plt


"""
Prototype for Firmware/Hardware Mobile Phone Hacking Detection Tool
Name: MNA Ahimbisibwe
Version: Prototype A
"""
# COLOR = color()  # color_list = [RED, GREEN, YELLOW, BLUE, WHITE]
path = "C:\\Android\\platform-tools\\adb"
MODEL = 'com.mufc.fireuvw'


class MobilePhoneID(ABC):
    def __init__(self, vendor_id, device_id, manufacturer):
        self.vendor_id = vendor_id
        self.device_id = device_id
        self.manufacturer = manufacturer

    @abstractmethod
    def return_vendor(self):
        pass

    @abstractmethod
    def return_device(self):
        pass

    @abstractmethod
    def return_manufacture(self):
        pass

    @abstractmethod
    def return_phone_model(self):
        pass


# User class that implements IUserInterface, IDeviceConnector
class User(IUserInterface, IDeviceConnector):
    def __init__(self, username):
        self.username = username


def adb_logcat_text():
    # Specify the adb command
    command = ["adb", "logcat", "-d"]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if the command was successful
    if result.returncode == 0:
        print("Command executed successfully. Output:")
        print(result.stdout.decode('utf-8'))
    else:
        print("Command execution failed. Errors:")
        print(result.stderr.decode('utf-8'))


# adb_logcat_txt()


def save_installed_packages(filename):
    command = ["adb", "shell", "dumpsys", "package", "packages"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print("Command executed successfully. Saving output to file...")
        # print(result)

        with open(filename, 'w') as file:
            file.write(result.stdout.decode('utf-8'))

        print(f"Output saved to '{filename}'.")
    else:
        print("Command execution failed. Errors:")
        print(result.stderr.decode('utf-8'))


def save_permissions(folder_path, filename):
    # Specify the adb command
    command = ["adb", "shell", "pm", "list", "permissions"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if the command was successful
    # color_list = [RED, GREEN, YELLOW, BLUE, WHITE]
    if result.returncode == 0:
        print(f"{COLOR[1]}Command executed successfully.{COLOR[3]} Saving output to file...{COLOR[4]}")

        # Construct the full file path
        full_file_path = os.path.join(folder_path, filename)

        # Save the output to the specified file
        with open(full_file_path, 'w') as file:
            file.write(result.stdout.decode('utf-8'))

        print(f"{COLOR[1]}Permissions saved to {COLOR[2]}'{full_file_path}'.{COLOR[4]}")
    else:
        print(f"{COLOR[0]}Command execution failed. Errors:{COLOR[4]}")
        print(result.stderr.decode('utf-8'))


class NotificationManager(INotificationManager):
    def send_alert(self, user, message):
        user.show_error(message)

    def send_report(self, hacking_report):
        report.display()


def enumerate_serial_phones():
    devices = []
    for device in list_ports.comports():
        devices.append({
            'Description': device.description,
            'Serial No.': device.serial_number,
            'USB Description': device.usb_description(),
            'Location': device.location,
            'Port': device.device,
            'Vendor ID': device.vid,
            'Product ID': device.pid,
            'Manufacturer': device.manufacturer})
    return devices


def show_device():
    devices = enumerate_serial_phones()
    for device in devices:
        print('Description: {}, Serial No.: {}, USB Description: {}, Location: {}, '.format(
            device['Description'], device['Serial No.'], device['USB Description'], device['Location']))
        print('Port: {}, Vendor ID: {}, Product ID: {}, Manufacturer: {}'.format(
            device['Port'], device['Vendor ID'], device['Product ID'], device['Manufacturer']))


show_device()


def get_port_name():
    for device in list_ports.comports():
        if device.vid and device.pid:  # Ensure the device has both a Vendor ID and Product ID
            return device.device  # Return the first found device's port name

    return None  # Return None if no suitable device is found


# print(get_port_name())

def get_device_info(property_name: str):
    command = ["adb", "shell", "getprop", property_name]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if the command was successful
    output_prop = ""
    if result.returncode == 0:
        print(f"{COLOR[1]}Command executed successfully. Output:{COLOR[4]}")
        # print(f"{property_name}: {result.stdout.decode('utf-8').strip()}")
        output_prop = result.stdout.decode('utf-8').strip()
    else:
        print(f"{COLOR[0]}Command execution failed. Errors:{COLOR[4]}")
        print(result.stderr.decode('utf-8'))
    return output_prop


# print(get_device_info("ro.product.model"))
# print(get_device_info("ro.product.brand"))
# print(get_device_info("ro.boot.serialno"))

def extract_device_info(directory, file_name):
    file_path = os.path.join(directory, file_name)
    device_info = {"ro.product.manufacturer": None, "ro.product.model": None, "ro.serialno": None}
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            input_text = file.read()
            lines = input_text.split('\n')

            for line in lines:
                for property_key in device_info.keys():
                    if property_key in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            device_info[property_key] = parts[1].strip()
    return device_info["ro.product.manufacturer"].strip('[]').upper(), \
           device_info["ro.product.model"].strip('[]').upper(), \
           device_info["ro.serialno"].strip('[]').upper()


class Phone(MobilePhoneID):
    def __init__(self, port_name=get_port_name()):
        super().__init__(port_name, None, None)
        self.port_name = port_name
        self.device_info = enumerate_serial_phones()
        device = self.get_device_info_by_port()
        try:
            if device is not None:
                super().__init__(device['Vendor ID'], device['Product ID'], device['Manufacturer'])
            else:  # Handle the absence of device gracefully
                print(f"{COLOR[0]}No device found on port >>> {COLOR[4]}")
        except ModuleNotFoundError:  # Handle the module not found error gracefully
            # You can raise custom exception i.e raise ModuleNotFoundException("Required module not found")
            # or self.handle_module_not_found()
            pass

    def return_vendor(self):
        try:
            if self.vendor_id is not None:
                return hex(self.vendor_id)
            else:
                return get_device_info("ro.product.vendor")
        except ModuleNotFoundError:
            return "Device Vendor Not Found"

    def return_device(self):
        try:
            if self.device_id is not None:
                return hex(self.device_id)
            else:
                return get_device_info("ro.serialno")
        except ModuleNotFoundError:
            return "Device Not Found"

    def return_manufacture(self):
        brand = get_device_info("ro.product.brand")
        return brand.upper()

    def return_phone_model(self):
        model = get_device_info("ro.product.model")
        return model

    def get_device_info_by_port(self):
        for device in self.device_info:
            if device['Port'] == self.port_name:
                return device
        return None

    def show(self):
        show_device()
        for device in self.device_info:
            print('Port: {}, Vendor ID: {}, Product ID: {}'.format(device['Port'], self.return_vendor(),
                                                                   self.return_device(), ))


# phone = Phone()
# phone.show()


def detect_firmware_on_adb_device():
    try:
        adb_output = subprocess.check_output(['adb', 'shell', 'getprop'])
        adb_output = adb_output.decode('utf-8').splitlines()
        main_version = 'Unknown'
        manufacturer = 'Unknown'

        for line in adb_output:
            if line.startswith('[ro.build.version.release]'):
                main_version = line.split(': ')[1].strip()
            elif line.startswith('[ro.product.bootimage.manufacturer]'):
                manufacturer = line.split(': ')[1].strip()

        firmware_info = f"Firmware Ver: {main_version}", f"Manufacturer: {manufacturer.upper()}"
        return firmware_info

    except subprocess.CalledProcessError:
        logging.error("Failed to execute ADB command.")
        return 'Unknown/No Device'


# print(detect_firmware_on_adb_device(COLOR[1], COLOR[4]))

def get_firmware_version(directory, file_name):
    file_path = os.path.join(directory, file_name)

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            input_text = file.read()
            lines = input_text.split('\n')

            for line in lines:
                if "ro.build.version.release" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        firmware_version = parts[1].strip()
                        return firmware_version
    return None


def firmware_ver():  # Real Time
    Build = autoclass('android.os.Build')
    version = Build.VERSION.RELEASE
    return version


# firmware_ver()

class Firmware:
    def __init__(self, version):
        self.version = version
        self.manufacturer = None

    def run(self):
        # detect_firmware_on_adb_device()
        print(detect_firmware_on_adb_device() + '\n')
        print(f"{COLOR[1]}Android Build Version: {COLOR[2]}{version}{COLOR[4]}\n")


# fm = Firmware(firmware_ver())
# fm.run()
# print(detect_firmware_on_adb_device().split(',')[0])


class Hardware:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_hardware_value(self):
        hardware = None
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith("[ro.hardware]:"):
                    hardware = str(line.strip().split(': [')[1].rstrip(']'))
                    break

        if hardware:
            print(f"{COLOR[3]}Hardware: {COLOR[4]}{hardware}")
            return f"Hardware: {hardware}"
        else:
            print(f"{COLOR[0]}Error: [ro.hardware] property not found in the provided file.{COLOR[4]}")
            return "Err: NO [ro.hardware]"

    @staticmethod
    def run_adb_command(command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        return output.decode('utf-8'), error.decode('utf-8')

    @classmethod
    def extract_cpu_percent(cls, output):
        # Use regular expressions to find lines containing CPU percentages
        # cpu_pattern = re.compile(r'(\d+\.\d+)%\s+(\d+/.+):')
        cpu_pattern = re.compile(r'(\d+\.\d+)%\s+TOTAL:')
        matches = cpu_pattern.findall(output)

        if matches:
            # Convert the percentage to a float and return
            return float(matches[0])
        return None

    @classmethod
    def collect_cpu_data(cls, device_id, duration_seconds):
        cpu_data = []

        # Record the start time
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Run 'dumpsys cpuinfo' command via ADB
            adb_command = f'adb -s {device_id} shell "dumpsys cpuinfo"'
            output, _ = cls.run_adb_command(adb_command)

            print("Raw Output:", output)

            # Try to extract CPU percentage from the command output
            cpu_percent = cls.extract_cpu_percent(output)
            cpu_data.append(cpu_percent)

            # Wait for 1 second before collecting the next data point
            time.sleep(2)
        return cpu_data

    @classmethod
    def predict_cpu_usage_anomaly(cls, cpu_data):
        # Convert the CPU data to a Pandas DataFrame
        import pandas as pd
        # import matplotlib.pyplot as plt
        import numpy as np
        df = pd.DataFrame(cpu_data, columns=['CPU Usage (%)'])

        # Calculate the z-score
        ''' 
        The z-score is a measure of how many standard deviations below or above the population mean a raw score is.
        It is calculated as: z = (x - μ) / σ where x is the raw score, μ is the population mean, 
        and σ is the population standard deviation. '''

        df['z-score'] = (df['CPU Usage (%)'] - df['CPU Usage (%)'].mean()) / df['CPU Usage (%)'].std()

        # Detect anomalies based on z-score
        # If it is 2 standard deviations away from the mean, we can consider it an anomaly
        anomalies = df[np.abs(df['z-score']) > 2]  # z-score > 2 is considered an anomaly
        if not anomalies.empty:
            print("Anomalies found!")
            messagebox.showwarning("CPU Usage Status: ", "Anomalies Detected.")
            print(anomalies)
        else:
            print("No Anomalies Detected.")
            messagebox.showinfo("CPU Usage Status: ", "No Anomalies Detected.")

        # Display the data
        print(df)
        # Print cpu usage
        # Plot the data
        # df.plot() # Plotting the data obtained from the device in real time  # TOBE investigated .

        # Display the plot
        plt.show()
        return anomalies, df

    @classmethod
    def save_cpu_data(cls, cpu_dataframe, output_file_path):
        cpu_dataframe[1].to_csv(output_file_path, index=False)

    def run(self):
        return self.get_hardware_value()


# h = Hardware(None)
# print(h.predict_cpu_usage_anomaly(h.collect_cpu_data("R5CN80EC24J", 1)))


# h.run()


class MobilePhone:
    def __init__(self, model, firmware):
        self.model = model
        self.firmware = firmware


# **************************** FIRMWARE ANALYSIS ******************************************************

def get_firmware_details(prop_ext: str):
    try:
        output_property = subprocess.check_output(['adb', 'shell', 'getprop', prop_ext])
        return output_property.strip().decode('utf-8')
    except subprocess.CalledProcessError:
        return None


def process_patch_level():
    current_date = datetime.now().strftime("%Y")
    expected_patch = f"{current_date}"  # Example: The latest security patch level you consider acceptable
    # TODO
    f"""
    The ro.build.version.security_patch property indicates the security patch level applied to the device. If the
    patch level is outdated or lower than expected, it might indicate a security vulnerability that has been
    exploited for tampering.
    """

    device_patch = get_security_patch('ro.build.version.security_patch')

    if device_patch:
        if device_patch >= expected_patch:
            print("Device is up to date.")
        else:
            print("Device security patch is outdated.")
    else:
        print("Failed to retrieve security patch information.")


class PullProp:
    def __init__(self, directory):
        self.directory = directory

    @property
    def get_device_properties(self):
        try:
            # Run adb command to get device properties
            result = subprocess.run(["adb", "shell", "getprop"], stdout=subprocess.PIPE, text=True, check=True)

            # Parse the output and save to a dictionary
            properties = {}
            for line in result.stdout.strip().split('\n'):
                parts = line.split(': ', 1)
                key = parts[0]
                value = parts[1] if len(parts) > 1 else None
                properties[key] = value.strip() if value is not None else None

            return properties
        except subprocess.CalledProcessError as e:
            print("Error:", e)
            return None

    def save_properties_to_file(self, properties, filename):
        filepath = f"{self.directory}/{filename}"
        with open(filepath, 'w') as f:
            for key, value in properties.items():
                f.write(f"{key}: {value}\n")

    def pull_and_save_properties(self, filename="device_properties.txt"):
        properties = self.get_device_properties  # was self.get_device_properties() before @property
        if properties:
            self.save_properties_to_file(properties, filename)
            print(f"Device properties saved to {filename} in {self.directory}")


class PullPackage:
    def __init__(self, package_dump_dir, packages_file='clean_installed_package_list.txt'):
        self.package_dump_dir = package_dump_dir
        self.packages_file = packages_file

    def dumpsys_package_to_file(self):
        try:
            adb_output = subprocess.check_output(['adb', 'shell', 'dumpsys', 'package', 'packages']).decode('utf-8')
            with open(self.package_dump_dir, 'w') as file:
                file.write(adb_output)
            print(f"Package information saved to {self.package_dump_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error running adb command: {e}")

    def extract_package_names_from_dumpsys_file(self):
        with open(self.package_dump_dir, 'r') as file:
            dump_text = file.read()

        package_pattern = r"Package \[(.*?)\]"
        package_names = re.findall(package_pattern, dump_text)

        return package_names

    def save_package_names_to_file(self, package_names):
        with open(self.packages_file, 'w') as file:
            for package in package_names:
                file.write(f"{package}\n")
        print(f"Extracted package names saved to {self.packages_file}")

    def get_installed_packages(self):
        cmd = ["adb", "shell", "pm", "list", "packages"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            packages = result.stdout.split("\n")
            # Removing the 'package:' prefix and filtering out empty strings
            packages = [pkg.replace("package:", "").strip() for pkg in packages if pkg]

            # Save to a text file
            path_to_file = os.path.join(self.package_dump_dir, "installed_packages.txt")
            with open(path_to_file, "w") as file:
                for pkg in packages:
                    file.write(pkg + '\n')

            return packages
        else:
            print("Error:", result.stderr)
            return []

    def run(self):
        self.dumpsys_package_to_file()
        package_names = self.extract_package_names_from_dumpsys_file()
        self.save_package_names_to_file(package_names)


class PullLogcat:
    def __init__(self, logcat_dump_dir):
        self.logcat_dump_dir = logcat_dump_dir
        self.timestamp = self.generate_timestamp

    # Utility function to generate a timestamp

    @property
    def generate_timestamp(self):
        return time.strftime('%Y%m%d_%H%M%S')

    def extract_logcat(self):

        logcat_filename = os.path.join(self.logcat_dump_dir, f"logcat_{self.timestamp}.txt")

        print(f"Saving logcat to {logcat_filename} ...")

        with open(logcat_filename, 'w') as f:
            try:
                # Extract and save logcat
                subprocess.run(['adb', 'logcat', '-d'], stdout=f, check=True)
            except subprocess.CalledProcessError as e:
                print("Error getting logcat:", e)

        print(f"Logcat saved to {logcat_filename}!")

    def reorganize_logcat(self):
        input_file = os.path.join(self.logcat_dump_dir, f"logcat_{self.timestamp}.txt")
        output_file = os.path.join(self.logcat_dump_dir, f"reorganized_logcat_{self.timestamp}.txt")

        # Mapping of log levels to colors (for better visibility)
        log_colors = {
            'V': 'grey',  # Verbose
            'D': 'blue',  # Debug
            'I': 'green',  # Info
            'W': 'yellow',  # Warning
            'E': 'red',  # Error
            'F': 'magenta',  # Fatal
            'S': 'white',  # Silent
        }

        # Format and filter the logs
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                # Typical logcat line structure:
                parts = line.split(None, 6)  # Splitting on whitespace

                if len(parts) < 6:
                    continue

                date, time, pid, tid, level, tag_message = parts[:6]  # TOBE investigated ...
                tag, message = tag_message.split(':', 1) if ':' in tag_message else (tag_message, '')

                # Example format
                formatted_log = f"[{date} {time}] [{level}] {tag.strip()}: {message.strip()}"
                f_out.write(formatted_log + '\n')

        print(f"{COLOR[1]}Reorganized logcat saved to{COLOR[1]}{COLOR[2]} {output_file}!{COLOR[4]}")


def extract_security_patch_date(folder_path: str, filename: str):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            input_text = file.read()
            lines = input_text.split('\n')

            for line in lines:
                if "security_patch" in line:
                    security_patch_date = line.split(": [")[1][:-1]
                    return security_patch_date
    print(f"{COLOR[0]}Text Containing Properties not Found: Connect Device Pull Properties{COLOR[4]}")
    messagebox.showinfo("Security Patch Status: ", "Text Containing Properties not Found: Please Connect Device and "
                                                   "Pull Properties")
    return None


# print(extract_security_patch_date("temp", "device_properties.txt"))


# Function to pull memory information from the device
def pull_memory_usage_info():
    if get_adb_devices():
        try:
            memory_info = subprocess.check_output(['adb', 'shell', 'dumpsys', 'meminfo']).decode('utf-8')
            return memory_info
        except subprocess.CalledProcessError:
            return None
    else:
        return None


def pull_dump_memory(app_name: str, output_file_path: str):
    command = ["adb", "shell", "dumpsys", "meminfo", app_name, "--debug-info-size"]
    with open(output_file_path, "wb") as f:
        subprocess.run(command, stdout=f)


def save_memory_usage_info_pulled(memory_info, output_file_path):
    try:
        with open(output_file_path, 'w') as file:
            file.write(memory_info)
        return True  # Return True to indicate successful saving
    except Exception as e:
        print(f"{COLOR[0]}Error while saving memory info: {e}{COLOR[4]}")
        return False  # Return False to indicate saving failure


class FirmwareAnalyser(IAnalyser):
    def __init__(self, temp_file_path):
        self.patch_level = None
        self.temp_file_path = temp_file_path

    # Analyzing the device's firmware, bootloader, and recovery partitions for signs of modification.
    def analyse(self, folder_path):
        # Code to analyze the firmware
        current_patch_level = extract_security_patch_date(folder_path, 'device_properties.txt')
        selinux = self.read_selinux_status(folder_path + '\\device_properties.txt')
        bootState = self.get_verified_boot_state(folder_path + '\\device_properties.txt')
        is_rooted = self.is_device_rooted(folder_path + '\\device_properties.txt')
        messagebox.showinfo("Run Firmware Analysis", "Firmware analysis completed.\nView Results in Text Box or Report")
        return current_patch_level, selinux, bootState.upper(), is_rooted

    # from adb shell getprop

    # from adb shell getprop
    @property  # CRITICAL SECURITY ISSUE
    def SELinuxStatus(self):  # ro.build.selinux
        f""" The ro.build.selinux property indicates the status of SELinux. If SELinux is set to "enforcing,
        "it means that it is active and can help prevent unauthorized changes to the system. If it is set to
        "permissive" or "disabled," it might suggest a potential security vulnerability. 
        """
        try:
            with open(self.temp_file_path, "r") as file:
                getprop_text = file.read()
        except FileNotFoundError:
            return f"{COLOR[0]}File not found.{COLOR[4]}"

        lines = getprop_text.split('\n')
        selinux_property = None
        enforce_property = None

        for line in lines:
            if "[ro.build.selinux]: [" in line:
                parts = line.split('[')
                if len(parts) >= 3:
                    selinux_property = parts[2].strip().rstrip(']')
            elif "[ro.build.selinux.enforce]: [" in line:
                parts = line.split('[')
                if len(parts) >= 3:
                    enforce_property = parts[2].strip().rstrip(']')

        if selinux_property == "1" and enforce_property == "1":
            selinux_status = "enforcing"
            enforce_status = "active"
            status_description = f"{COLOR[1]}SELinux is {selinux_status.upper()} --> " \
                                 f"{enforce_status.upper()} {COLOR[2]} ## Permission is Disabled {COLOR[4]}"
            messagebox.showinfo("SELinux Status Info", "SELinux is set ENFORCING --> ACTIVE \n"
                                                       "## Permission is Disabled.")
        else:
            messagebox.showwarning("SELinux Status Warning", 'Warning: SELinux Status is NOT Set to Enforced ! ')
            status_description = f"{COLOR[0]}Warning: SELinux Status is not Set to Enforced.{COLOR[4]}"

        return status_description

    # from adb shell getprop
    @property  # CRITICAL SECURITY ISSUE
    def boot_state(self):
        f"""
        The  ro.boot.verifiedbootstate property can provide information about the verified boot state. If the value is 
        set to "green" or "locked," it indicates that the bootloader and the kernel have not been tampered with. If 
        the value is "yellow" or "orange," it may suggest that the device is unlocked or has an unlocked bootloader, 
        which could potentially indicate tampering. : 
        """
        try:
            with open(self.temp_file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(": ")
                    if len(parts) == 2:
                        _prop, value = parts[0].strip("[]"), parts[1].strip("[]")
                        # Check if the property matches the desired property
                        if _prop == "ro.boot.verifiedbootstate":
                            if value in ["green", "locked"]:
                                messagebox.showinfo("Boot Analysis", f"Bootloader NOT Tempered \nKernel Not Tempered")
                                return f"{COLOR[1]}→The bootloader and the kernel have not been tampered with." \
                                       f"{COLOR[4]}"

                            elif value in ["yellow", "orange"]:
                                messagebox.showwarning("Boot Analysis", f"Bootloader and Kernel Tempered")
                                return (f"{COLOR[0]}The device is unlocked or has an unlocked bootloader, "
                                        f"which could potentially indicate tampering.{COLOR[4]}")
                            else:
                                return f"{COLOR[0]}Unknown verified boot state: {COLOR[2]}{value}{COLOR[4]}"

            # If the loop completes without a return, the property was not found
            return "The 'ro.boot.verifiedbootstate' property was not found in the file."

        except FileNotFoundError:
            return f"File '{self.temp_file_path}' not found."

    @classmethod  # MINIMAL SECURITY ISSUE
    def security_patch_level_analysis(cls, folder_path: str, filename: str, max_months_old: int):
        # current_date = date.today()  # datetime.datetime.now() # **** With from datetime import date
        current_date = datetime.datetime.fromtimestamp(time.time()).date()
        current_patch_level = extract_security_patch_date(folder_path, filename)

        if current_patch_level:
            patch_date = datetime.datetime.strptime(current_patch_level, "%Y-%m-%d").date()

            time_difference = current_date - patch_date
            months_old = time_difference.days // 30  # Approximate months

            if months_old <= max_months_old:
                print(f"{COLOR[3]}Device Security Patch Level {COLOR[2]}{patch_date}{COLOR[1]} "
                      f"is Within the Safe Range.{COLOR[4]}")
                messagebox.showinfo("Device Security Patch Level: ", f"Patch Leve: {patch_date}\nWarning Message: "

                                                                     f"Within the Safe Range.")
                return True
            else:
                print(f"{COLOR[0]}Device Security Patch Level {COLOR[2]}{patch_date}{COLOR[0]} "
                      f"is Older Than Expected. Check Firmware Manufacture.{COLOR[4]}")
                messagebox.showwarning("Device Security Patch Level ", f"{patch_date} is Older Than Expected. Check "
                                                                       f"Firmware Manufacture.")
                return False
        else:
            return False

    @classmethod  # CRITICAL SECURITY ISSUE
    def root_state_analysis(cls, file_path):
        if cls.is_device_rooted(file_path) == 'ROOTED':
            messagebox.showwarning("Device Rooted Status", 'WARNING: Device Maybe Rooted')
            print(f"{COLOR[0]}WARNING: Device Maybe Rooted {COLOR[4]}")
        elif cls.is_device_rooted(file_path) == 'Root State OEM':
            messagebox.showinfo("Device Rooted Status", 'Device Root State is Set to OEM')
            print(f"{COLOR[1]}Device Root State is Set to OEM {COLOR[4]}")
        else:
            messagebox.showwarning("Device Rooted Status", 'Device Root State UNKNOWN')
            print(f"{COLOR[2]}Device Root State is UNKNOWN {COLOR[4]}")

    @classmethod
    def is_device_rooted(cls, file_path):
        """
        Reads the build tags from a getprop output file and determines if the device may be rooted.

        param file_path: Path to the file containing the output of getprop
        :return: True if the device might be rooted, False otherwise
        """

        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Check for various indicators of root
        if "[test-keys]" in content:
            return 'ROOTED'
        if "[release-keys]" in content:
            return 'Root State OEM'

        # If neither test-keys nor release-keys are found, the state is unknown
        print("Root state is unknown based on the provided file.")
        return 'ROOT STATE UNKNOWN'

    @classmethod
    def read_selinux_status(cls, file_path):
        """
        Reads the selinux and selinux enforce statuses from the provided txt file.
        :return: tuple containing status of selinux and selinux enforce
        """
        selinux_status = None
        selinux_enforce_status = None

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("[ro.build.selinux]:"):
                    selinux_status = int(line.strip().split(': [')[1].rstrip(']'))
                elif line.startswith("[ro.build.selinux.enforce]:"):
                    selinux_enforce_status = int(line.strip().split(': [')[1].rstrip(']'))

        if selinux_status is not None and selinux_enforce_status is not None:
            return selinux_status, selinux_enforce_status
        else:
            raise ValueError("Could not extract required values from the file.")

    @classmethod
    def get_verified_boot_state(cls, file_path):
        """
        Reads the boot state value for "ro.boot.verifiedbootstate" from the given file.

        param file_path: Path to the file containing the output of getprop
        :return: Value of the "ro.boot.verifiedbootstate" property or None if not found
        """
        with open(file_path, 'r') as file:
            for line in file:
                if "ro.boot.verifiedbootstate" in line:
                    return line.split(': ')[1].strip().replace('[', '').replace(']', '')
        return None

    @classmethod
    def get_oem_system_security_level(cls, file_path):
        """
        Reads the value of "sys.oem_unlock_allowed" from the given file.

        param file_path: Path to the file containing the output of getprop
        :return: Value of the "sys.oem_unlock_allowed" property or None if not found
        """
        # sys.oem_unlock_allowed is a boolean property that indicates whether the bootloader is unlocked
        # or the device is unlockable, which could potentially indicate tampering.
        with open(file_path, 'r') as file:
            for line in file:
                if "sys.oem_unlock_allowed" in line:
                    return line.split(': ')[1].strip().replace('[', '').replace(']', '')
        return None

    def run(self):
        process_patch_level()


# print(FirmwareAnalyser(None).get_oem_system_security_level('temp\\device_properties.txt'))

# print(r.analyse(temp))
# r.root_state_analysis(temp + '\\device_properties.txt')


class HackingReport:
    def __init__(self, date, description, analysis_results):
        self.date = date
        self.description = description
        self.analysis_results = analysis_results

    def display(self):
        # Code to display the report
        messagebox.showinfo("Show Report", "This is a report.")


class HackingDetector:  # # TODO
    def detect_hacking(self, hardware, firmware):
        # Code to detect hacking
        pass

    # TODO
    def analyze_firmware(firmware):
        pass

    # TODO
    def analyze_hardware(hardware):
        pass

    # TODO
    def generate_hacking_report(self) -> HackingReport:
        pass

    def root_detection_mechanism(self):
        # Implementing root detection mechanisms to check if the device has been rooted or tampered with.
        pass

    def integrity_check(self):
        #  Verifying the integrity of firmware and system files to detect unauthorized modifications.
        pass

    def app_behavior_analysis(self):
        # Monitoring application behavior for any suspicious or unexpected activities.
        pass

    def trusted_boot_mechanism(self):
        # Leveraging secure boot mechanisms to ensure that the device starts with trusted firmware and software.
        pass


class PolymorphicAbility:
    def __init__(self):
        print (f"{COLOR[1]}", "Polymorphically executed")
        print (f"{COLOR[RED]}", "Error Occured and you must polymorphically rebuild")
        pass


try:
    _path = create_folder('temp')
    temp_prop_puller = PullProp('temp')
    temp_prop_puller.pull_and_save_properties()
    prop = extract_device_info('temp', 'device_properties.txt')
except Exception as e:
    print(f"{COLOR[0]}An error occurred:{COLOR[4]}", str(e))


def adb_device_connected():
    try:
        result = subprocess.check_output(['adb', 'devices']).decode('utf-8')
        return 'device' in result
    except subprocess.CalledProcessError:
        return False


def file_exists(filename):
    return os.path.exists(filename)


def temp_prop():
    if adb_device_connected():
        if file_exists('device_properties.txt'):
            prop = extract_device_info('temp', 'device_properties.txt')
            return prop
        else:
            return "Error: device_properties.txt is missing."
    else:
        return "Error: No ADB device connected."

# prop = temp_prop()
