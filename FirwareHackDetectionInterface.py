import datetime
import importlib
import random
import time
from tkinter import ttk
import string
import tkinter.messagebox as messagebox

"""
Prototype for Firmware/Hardware Mobile Phone Hacking Detection Tool Interface

Author: [Michael Nhyk Ahimbisibwe]
System Name: [Mobile Phone Firmware Hardware Hacking Detection]
Model: [BSC HON YEAH PROJECT 1.0]
"""

import customtkinter
from FirmwareHackDetector import *
from InterfaceHub import IUserInterface, IDeviceConnector
from APK_PROCESS import *

customtkinter.set_appearance_mode("dark")  # Mode can be Light or Dark
# customtkinter.set_default_color_theme("blue")  # Can be blue , green or dark blue

devices = get_adb_devices()

customtkinter.set_default_color_theme('blue')

COLOUR = ["\033[1;30;40m", "\033[1;31;40m", "\033[1;32;40m", "\033[1;33;40m", "\033[1;34;40m", "\033[1;35;40m",
          "\033[1;36;40m", "\033[1;37;40m", "\033[1;38;40m", "\033[1;39;40m", "\033[1;40;40m"]

try:
    immediate_dir = f"C:\\Android\\DetectHacking\\{prop[0]}_{prop[1]}_{prop[2]}"
    print(f"{COLOR[3]}Device directory: {COLOR[2]}{immediate_dir}{COLOR[4]}")
except (IndexError, NameError) as e:
    print(f"Error: {e}")
    immediate_dir = "."

# Color names in COLOUR list are: Black, Red, Green, Yellow, Blue, Magenta, Cyan, White, Grey, Default, Reset

# from usable import GlobalReport

report = GlobalReport(immediate_dir)
report.append_result("Type", "Report")
report.append_result("Date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result="System Date")
try:
    report.append_result("Device", prop[0])
    report.append_result("Model", prop[1])
    report.append_result("Serial", prop[2])
except (IndexError, NameError) as e:
    print(Fore.RED + f"File not Found: {e}" + Fore.WHITE)
    report.append_result("Device", "No Ddevice Connected")
    report.append_result("Model", "No Device Connected")
    report.append_result("Serial", "No Device Connected")
report.show_report()


def pull_apk(package_name, save_path):
    # Get the path of the APK on the device
    cmd = ["adb", "shell", "pm", "path", package_name]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Extract APK path from the returned string
        apk_path_on_device = result.stdout.strip().replace("package:", "")

        # Command to pull the APK
        cmd = ["adb", "pull", apk_path_on_device, save_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully pulled {package_name} to {save_path}")
        else:
            print(f"Error pulling {package_name}: {result.stderr}")
    else:
        print(f"Error finding APK path for {package_name}: {result.stderr}")


def change_theme_event(strMode: str):
    current_theme = customtkinter.get_appearance_mode()
    customtkinter.set_appearance_mode(strMode)
    print(f"Theme change from '{COLOR[2]}{current_theme}{COLOR[4]}' to '{COLOR[2]}{strMode}'{COLOR[4]}")


def change_scaling_event(strNewScaling: str):
    intNewScalingDouble = int(strNewScaling.replace("%", "")) / 100.0
    customtkinter.set_widget_scaling(intNewScalingDouble)


def frame(frame_, text: str):  # This is the text box
    text_box = customtkinter.CTkTextbox(frame_, width=140, height=140, font=("Calibre", 13, "roman"),
                                        fg_color="#4c6e81")
    text_box.insert("1.0", text.format("Calibre", 13, "italic"))
    return text_box


def shorter_frame(frame_, text: str):
    text_box = customtkinter.CTkTextbox(frame_, width=130, height=80, font=("Calibre", 13, "roman"),
                                        fg_color="#4c6e81")
    text_box.insert("1.0", text.format("Calibre", 13, "italic"))
    return text_box


def create_phone_files_folder():
    # Create a folder for the device
    _phone = Phone()
    model = _phone.return_phone_model()
    name = _phone.return_manufacture()
    serial = get_device_info("ro.boot.serialno")

    folder_path = f"C:\\Android\\DetectHacking\\{name}_{model}_{serial}"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path
    # End of create folder for this device (will this work effectively in create specified folder)


# print(create_phone_files_folder())
# try:
#     immediate_dir = f"C:\\Android\\DetectHacking\\{prop[0]}_{prop[1]}_{prop[2]}"
#     # The rest of your code that depends on immediate_dir
# except (IndexError, NameError) as e:
#     print(f"Error: {e}")
#     immediate_dir = ""


def detect_patch_level():
    patch = FirmwareAnalyser(None)
    directory = immediate_dir  # "C:\\Android\\DetectHacking\\SAMSUNG_SM-N986B_R5CN80EC24J"
    patch.security_patch_level_analysis(directory, "device_properties.txt", 6)
    # FirmwareAnalyser.security_patch_level_analysis(directory, "device_properties.txt", 6)


def boot_state_analysis():
    prop_dir = '\\device_properties.txt'
    boot_state = FirmwareAnalyser(immediate_dir + prop_dir)
    print(boot_state.boot_state)


def SELinux_analysis():
    prop_dir = '\\device_properties.txt'
    SELinux_state = FirmwareAnalyser(immediate_dir + prop_dir)
    print(SELinux_state.SELinuxStatus)


def rooted_state():
    root_state_results = FirmwareAnalyser(None)
    root_state_results.root_state_analysis(immediate_dir + "\\device_properties.txt")


# To be used in status changer...
def set_widgets_status_in_frame(_frame, status):
    for child in _frame.winfo_children():
        if status == 'enabled':
            child.configure('normal')
        elif status == 'disabled':
            child.configure('disabled')


class SystemUserInterface(IUserInterface, IDeviceConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = []  # This is the list of apps to be scanned ...
        self.scan_app = None  # This is the scan app button
        self._app_list = None
        self.side_bar_text_area = None  # Added later ....
        self.mobile_phone_graphics = None  # Added later ....
        self.hardware_analysis_tabs = []
        self.buttons = []
        self.root_ctk = customtkinter.CTk()
        self.general_tab_view = customtkinter.CTkTabview(self.root_ctk, width=360)
        self.root_ctk.geometry(f"{1150}x{588}")  # Size of the window,
        self.main_frame_color = None
        self.main_color_changer(0)  # This changes the phone graphics status (Active, Inactive
        self.live_text = None

        self.drop_down_menu = None
        self.app_drop_down_menu = None
        self.drop_down_apps = None
        self.select_model = None
        self.drop_down_model_var = None
        self.drop_down_per_based_model_var = None

        # TEXT BOXES
        self.firmware_analysis_text_box = None
        self.hardware_analysis_text_box = None
        self.general_textbox = None
        self.firmware_text_box = None
        self.hardware_text_box = None
        self.hacking_text_box = None
        self.help_menu_text = None

        self.random_forest_pre_trained = None

        #
        self.permission_based_hack_detection = None

        self.root_ctk.title(self.__class__.__name__)
        self.root_ctk.iconbitmap("uj_or.ico")

        # Configure the root_ctk grid , this will span the sidebar all the way ###### SIDE BAR
        self.root_ctk.grid_rowconfigure(0, weight=1)
        self.root_ctk.grid_columnconfigure(1, weight=1)

        self.sideBarFrame = customtkinter.CTkFrame(self.root_ctk, width=145, corner_radius=0)
        self.sideBarFrame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sideBarFrame.grid_rowconfigure(4, weight=1)

        # Sidebar objects

        self.initial_buttons_title = customtkinter.CTkLabel(self.sideBarFrame, text="Attach Phone",
                                                            font=customtkinter.CTkFont(size=16, weight="bold"),
                                                            anchor="w")
        self.initial_buttons_title.grid(row=0, column=0, padx=18, pady=(18, 10))

        # Connect Device Button
        self.connect_button = customtkinter.CTkButton(self.sideBarFrame, text="Connect Phone", fg_color="#006600",
                                                      hover_color="#3EA055", command=self.connect_device)
        self.connect_button.grid(row=1, column=0, padx=20, pady=(20, 10))  # Alternative

        # Disconnect Device Button
        self.disconnect_button = customtkinter.CTkButton(self.sideBarFrame, text="Disconnect Device")
        self.disconnect_button.grid(row=2, column=0, padx=20, pady=20)
        self.state_all('disabled')
        '''self.disconnect_button.configure(state="disabled")'''

        # Create the font style object----------------Side Bar Text Box ----------------------------------
        self.side_bar_text_box = frame(self.sideBarFrame, "Connect device")  # Text area in the sidebar

        self.side_bar_text_box.grid(row=3, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.side_bar_text_box.configure(fg_color="#024641")

        self.help_button = customtkinter.CTkButton(self.side_bar_text_box, text="Help", fg_color="#b30000",
                                                   hover_color='#3D0C02', command=self.save_and_read_help_menu_html)
        self.help_button.grid(row=1, column=0, padx=20, pady=15)
        self.help_button.configure("enabled")

        self.basic_user_button = customtkinter.CTkButton(self.sideBarFrame, text="Basic User", fg_color="#454545",
                                                         hover_color='#616161',
                                                         command=lambda: self.basic_user_interface())
        self.basic_user_button.grid(row=5, column=0, padx=20, pady=15)
        self.basic_user_button.configure(state="disabled")

        self.advanced_user_button = customtkinter.CTkButton(self.sideBarFrame, text="Advanced User", fg_color="#000000",
                                                            hover_color='#616161',
                                                            command=lambda: self.advanced_user_interface())
        self.advanced_user_button.grid(row=6, column=0, padx=20, pady=15)
        self.advanced_user_button.configure(state="disabled")

        # Theme switching

        self.theme_switching_menu = customtkinter.CTkOptionMenu(self.sideBarFrame,
                                                                values=['Change Theme', "Light", "Dark", "System"],
                                                                command=change_theme_event)
        self.theme_switching_menu.grid(row=7, column=0, padx=20, pady=(10, 10))

        self.scaling_menu = customtkinter.CTkOptionMenu(self.sideBarFrame, values=["96", "100", "110"],
                                                        command=change_scaling_event)
        self.scaling_menu.grid(row=8, column=0, padx=20, pady=(10.0, 20.0))
        self.scaling_menu.set("UI Scaling:")

        # Main Screen Area

        # ------------------------ GENERAL TAB VIEW COL 2 ROW 2 --------------------------
        self.general_textbox = frame(None, "No Action Made (Notifications Will Show Here)\n")  # Text area
        self.general_textbox.grid(row=0, column=1, padx=(20, 0), pady=(16, 0), sticky="nsew")
        self.general_textbox.insert(tk.END, global_text_green + '\n')

        self.connection_progress = customtkinter.CTkLabel(self.general_textbox, text="Phone Connecting:",
                                                          font=customtkinter.CTkFont(size=12, weight="normal",
                                                                                     slant="roman"), anchor="w")
        self.connection_progress.grid(row=1, column=0, padx=20, pady=(15.0, 0.0), sticky="nw")
        # Login Status
        self.min_tab_view = customtkinter.CTkTabview(self.root_ctk, width=250, state='enabled')

        self.login_status_button = customtkinter.CTkRadioButton(self.general_textbox, text="Login Status",
                                                                state='enabled', value=1,
                                                                text_color="#16B2B2", bg_color="#282424")
        self.login_status_button.grid(row=2, column=0, padx=20, pady=(15.0, 0.0), sticky="nw")
        self.connection_progress_bar = customtkinter.CTkProgressBar(self.general_textbox)
        self.connection_progress_bar.grid(row=1, column=0, padx=20.0, pady=(15.0, 0.0), sticky="nw")
        self.general_tab_view = customtkinter.CTkTabview(self.root_ctk, width=360)
        general_tab = self.general_tab_view.add("Pull Files")
        self.general_tab_view.grid(row=1, column=1, padx=(20, 0), pady=(20.0, 0), sticky="nsew")
        # Connection progress bar

        # Save files >>>

        # Drop down
        # Create a list of options for the drop-down menu
        options = ['Drop Down Select Pull File', 'Pull Permissions', 'Pull Apps List', 'Pull Properties',
                   'Pull Memory Usage Info', 'Pull Logcat', 'CPU Usage']
        options_app = ['Drop Pull App Memory Dump', 'All Apps']
        # Create the drop-down menu widget
        self.drop_down_var = tk.StringVar()
        self.options_app_var = tk.StringVar()
        self.drop_down_menu = customtkinter.CTkOptionMenu(general_tab, values=options, variable=self.drop_down_var,
                                                          dropdown_hover_color='#000000', dropdown_text_color='#C0C0C0',
                                                          dropdown_fg_color='#033E3E', fg_color='#006A4E', button_color=
                                                          '#004225', )
        self.pull_app_memo_dump_dropdown = customtkinter.CTkOptionMenu(general_tab, values=options_app,
                                                                       variable=self.options_app_var,
                                                                       dropdown_hover_color='#000000',
                                                                       dropdown_text_color='#C0C0C0',
                                                                       dropdown_fg_color='#033E3E',
                                                                       fg_color='#006A4E', button_color='#004225')
        # Set an initial value for the drop-down menu
        self.drop_down_menu.set(options[0])
        self.pull_app_memo_dump_dropdown.set(options_app[0])
        # Configure the appearance of the drop-down menu
        self.drop_down_menu.configure(font=customtkinter.CTkFont(size=15, weight="normal", slant="roman"), width=25)
        # Place the drop-down menu in the grid
        self.drop_down_menu.grid(row=1, column=0, padx=20, pady=(15.0, 0.0), sticky='nw')
        self.pull_app_memo_dump_dropdown.grid(row=4, column=0, padx=20, pady=(15.0, 0.0), sticky='nw')
        self.drop_down_menu.configure(state='disabled')
        self.pull_app_memo_dump_dropdown.configure(state='disabled')
        # Create the button for executing the selected option
        self.execute_button = customtkinter.CTkButton(general_tab, text="<- Pull File", width=22,
                                                      command=lambda: self.execute_selected_option())
        self.pull_app_memory_dump_button = customtkinter.CTkButton(general_tab, text="<- Pull File", width=22,
                                                                   command=lambda: None)
        # self.execute_button.configure(bg_color='#FF0000')
        self.execute_button.grid(row=1, column=1, padx=10, pady=(15.0, 0.0), sticky='nw')
        self.pull_app_memory_dump_button.grid(row=4, column=1, padx=10, pady=(15.0, 0.0), sticky='nw')
        self.execute_button.configure(state="disabled")
        self.pull_app_memory_dump_button.configure(state="disabled")
        self.pull_android_manifest_button = customtkinter.CTkButton(general_tab, text="Pull App AndroidManifest",
                                                                    command=lambda: self.pull_AndroidManifest())
        self.pull_android_manifest_button.grid(row=0, column=0, padx=10, pady=(15.0, 0.0), sticky='sn')
        self.pull_android_manifest_button.configure(state='disabled')

        # ------------------------ MALWARE DETECTION TAB VIEW COL 2 ROW 2 --------------------------
        # malware_detection_tab = self.general_tab_view.add("Malware Detection")
        self.malware_detection('disabled')
        self.report('enabled')

        # ------------------------ EO MALWARE DETECTION TAB VIEW COL 2 ROW 2 -----------------------

        # Configure the root_ctk grid , this will span the sidebar all the way RIGHT
        self.initialize_tabview()  # Initial tab view elements
        self.sideBarFrameR = customtkinter.CTkFrame(self.root_ctk, width=15, corner_radius=0, fg_color="black")
        self.sideBarFrameR.grid(row=0, column=4, rowspan=4, sticky="nsew")
        self.sideBarFrameR.grid_rowconfigure(4, weight=1)

        # Configure the root_ctk grid , this will span the sidebar all the way BOTTOM
        self.initialize_tabview()  # Initial tab view elements
        self.sideBarFrameBottom = customtkinter.CTkFrame(self.root_ctk, height=10, corner_radius=0, fg_color="black")
        self.sideBarFrameBottom.grid(row=4, column=0, columnspan=5, sticky="nsew")
        self.sideBarFrameBottom.grid_columnconfigure(4, weight=1)

        self.mobile_phone_firmware_view_frame = customtkinter.CTkTabview(self.root_ctk, width=250, state='enabled')

        # self.side_bar_text_area.insert(tk.END, f"\n{self.live_text}")

        self.mobile_phone_view_frame = customtkinter.CTkTabview(self.root_ctk, width=250, state='enabled')

        self.mobile_phone_hacking_detector_view_frame = customtkinter.CTkTabview(self.root_ctk, width=250,
                                                                                 state='enabled')

        # Function to execute the selected option based on the drop-down menu value

    def state_all(self, state: str):
        self.disconnect_button.configure(state=state)
        # TOOLS
        # 0001 ***********************
        self.phone_hardware_firmware(state)
        # 0002 ***********************
        self.firmware_hardware_analysis(state)
        # 0002 ***********************
        self.firmware_hardware_hacking_detect(state)

        # MOBILE PHONE ************************************
        self.phone(state)

    def pull_AndroidManifest(self):
        self.general_textbox.insert(tk.END, "AndroidManifest Saved: C:\\Android\\DetectHacking\\...\\manifest\n")
        path = os.path.join(immediate_dir, 'clean_installed_package_list.txt')
        path2 = os.path.join(immediate_dir, 'installed_packages.txt')
        if not os.path.exists(path):
            app_list = AppSelector(path2, immediate_dir)
        else:
            app_list = AppSelector(path, immediate_dir)
        app_list.run()

    def execute_selected_option(self):
        selected_option = self.drop_down_var.get()
        directory = create_phone_files_folder()
        if selected_option == 'Pull Permissions':
            save_permissions(directory, 'permissions.txt')  # permissions saved in the directory
            self.general_textbox.insert(tk.END, "Permission saved in: C:\\Android\\DetectHacking\n")
        elif selected_option == 'Pull Apps List':
            # apm_app_list = os.path.join(immediate_dir, 'apm_app_list.txt')
            apm_app_list = os.path.join(immediate_dir)
            get_apps = PullPackage(apm_app_list, None)

            dumpsys_ = os.path.join(immediate_dir, 'dumpsys_package_list.txt')
            dumpsys__ = os.path.join(immediate_dir, 'clean_installed_package_list.txt')
            apps = PullPackage(dumpsys_, dumpsys__)
            get_apps.get_installed_packages()
            apps.run()
            jason_app_list = APK_PROCESS(immediate_dir)  # This is the APK_PROCESS
            jason_app_list.run()
            # self.get_app_list()
            self.get_app_list_from_json_file()
            self.general_textbox.insert(tk.END, "Installed App Logs Saved in: C:\\Android\\DetectHacking\n")
        elif selected_option == 'Pull Properties':
            prop_puller = PullProp(directory)
            prop_puller.pull_and_save_properties()
            self.general_textbox.insert(tk.END, "Properties Saved in: C:\\Android\\DetectHacking\n")
        elif selected_option == 'Pull Memory Usage Info':
            memory_info = pull_memory_usage_info()
            if memory_info:
                output_file_path = f"{immediate_dir}/memory_usage_info.txt"  # Update this path
                save_memory_usage_info_pulled(memory_info, output_file_path)
                print(f"{COLOR[3]}Memory information saved to {COLOR[2]}{output_file_path}{COLOR[4]}")
            self.general_textbox.insert(tk.END, "Memory Info Saved in: C:\\Android\\DetectHacking\n")
            print(f"{COLOR[1]}Memory information pulled{COLOR[4]}")
        elif selected_option == "Pull Logcat":
            puller = PullLogcat(immediate_dir)
            puller.extract_logcat()
            puller.reorganize_logcat()
            self.general_textbox.insert(tk.END, "Logcat saved: C:\\Android\\DetectHacking\n")
        elif selected_option == 'CPU Usage':
            h = Hardware(None)
            cpu_data = h.collect_cpu_data(f"{prop[2]}", 1)
            cpu_dataframe = h.predict_cpu_usage_anomaly(cpu_data)

            h.save_cpu_data(cpu_dataframe, immediate_dir + "\\cpu_usage.csv")
            pass

    @staticmethod  # This is a static method to be used in the interface class only ...
    def get_installed_packages():
        cmd = ["adb", "shell", "pm", "list", "packages"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            packages = result.stdout.split("\n")
            # Removing the 'package:' prefix and filtering out empty strings
            packages = [pkg.replace("package:", "").strip() for pkg in packages if pkg]

            # Save to a text file
            immediate_dir_ = create_phone_files_folder()
            path_to_file = os.path.join(immediate_dir_, "installed_packages.txt")
            with open(path_to_file, "w") as file:
                for pkg in packages:
                    file.write(pkg + '\n')

            return packages
        else:
            print("Error:", result.stderr)
            return []

    # ***************************************************************************************************************
    def get_app_list(self) -> list:
        try:
            app_list = os.path.join(immediate_dir, 'clean_installed_package_list.txt')
            options = read_text_file(app_list) + ['All Apps']
        except FileNotFoundError:
            options = ['All Apps']
            # options = read_text_file('apps.txt') + ['All Apps']
        self.app_drop_down_menu['values'] = options
        return options

    def get_app_list_from_json_file(self) -> list:
        try:
            app_list = os.path.join(immediate_dir, 'installed_apps.json')
            with open(app_list, 'r') as f:
                data = json.load(f)
            options = data + ['ackman.placemarks', 'All Apps']

        except FileNotFoundError:
            print(Fore.RED, "Jason App List Read Fail", Style.RESET_ALL)
            options = ['All Apps']

        self.app_drop_down_menu['values'] = options

        return options

    def malware_detection(self, status):
        malware_detection_tab = self.general_tab_view.add("|Malware Detection|")

        self.drop_down_model_var = tk.StringVar()
        self.random_forest_pre_trained = ["apk_high.model", "apk_good.model"]

        self.select_model = \
            customtkinter.CTkOptionMenu(malware_detection_tab, variable=self.drop_down_model_var,
                                        values=self.random_forest_pre_trained, dropdown_hover_color='#000000',
                                        dropdown_text_color='#C0C0C0', dropdown_fg_color='#033E3E', fg_color='#006A4E',
                                        button_color='#004225')
        self.select_model.set('Select Model-->')
        self.select_model.grid(row=0, column=0, padx=20, pady=(15.0, 0.0), sticky="nw", )
        self.select_model.configure(font=customtkinter.CTkFont(size=15, weight="bold", slant="roman"),
                                    state=status, width=15)
        # Create the drop-down menu widget
        scan_saved_apk = customtkinter.CTkButton(malware_detection_tab, text="Analyse Saved APK ...",
                                                 state=status, command=self.scan_saved_apk)
        scan_saved_apk.grid(row=0, column=1, padx=(10, 10), pady=(10, 10), sticky="nw")

        self.drop_down_apps = tk.StringVar()
        self.app_drop_down_menu = ttk.Combobox(malware_detection_tab, textvariable=self.drop_down_apps)
        # Set an initial value for the drop-down menu
        self.app_drop_down_menu.set('Select Model -> Drop Down to Select & Scan App')
        # Configure the appearance of the drop-down menu
        self.app_drop_down_menu.config(font=customtkinter.CTkFont(size=14, weight="normal", slant="roman"), width=38)
        # Place the drop-down menu in the grid
        self.app_drop_down_menu.grid(row=1, column=0, padx=20, pady=(15.0, 0.0), sticky="nw", columnspan=2)
        # self.app_drop_down_menu['values'] = self.options
        self.app_drop_down_menu.config(state=status)
        self.scan_app = customtkinter.CTkButton(malware_detection_tab, text="Scan App ...", fg_color='#B8860B',
                                                hover_color='#633a00', state=status,
                                                command=lambda: self.scan_selected_app())
        self.scan_app.grid(row=2, column=0, padx=20, pady=(10, 10))
        scan_all_app = customtkinter.CTkButton(malware_detection_tab, text="Scan All Apps ...",
                                               state=status, command=None)
        scan_all_app.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="nw")

    def report(self, state):
        # script_directory = os.path.dirname(os.path.abspath(__file__))
        # image_path = os.path.join(script_directory, "dlogo.png")
        hacking_report_tab = self.general_tab_view.add("|Report and Stats|")
        report_button = customtkinter.CTkButton(hacking_report_tab, text="📈 Generate Report 📱", state=state,
                                                border_width=0, font=customtkinter.CTkFont(size=18, weight="normal",
                                                                                           slant="roman"),
                                                corner_radius=8, width=200, height=50, fg_color="#045D5D",
                                                hover_color="#033E3E")
        report_button.grid(row=0, column=0, padx=20, pady=(15.0, 0.0), sticky="nw")
        report_button.configure(state=state, command=lambda: self.generate_report())

    def generate_report(self):
        print("Generate Report ...")
        # Open JASON saved in the immediate directory
        try:  # This is to check if the file exists
            JASON = os.path.join(immediate_dir, 'global_report.json')
        except FileNotFoundError:
            print("No JASON file found")
            JASON = os.path.join(".", 'global_report.json')
        import webbrowser
        webbrowser.open(JASON)

    def scan_saved_apk(self):
        print("Scan Saved APK ...")

    def scan_selected_app(self):
        selected_option = self.drop_down_apps.get()
        model = self.select_model.get()
        parent_dir = os.path.join(immediate_dir, selected_option)

        swift_path = selected_option
        # os.makedirs(swift_path, exist_ok=True)
        try:
            pull_apk_by_short_name(selected_option, swift_path)
            os.makedirs(parent_dir, exist_ok=True)
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showwarning("Error", "App Name is not Selected")
            return

        JASON = os.path.join(parent_dir, 'model_stats.json')

        if model == "apk_good.model":
            # Walk in to find the apk file
            if selected_option == "ackman.placemarks":
                self.detect_malicious_app("MA\\ackman.placemarks.apk", JASON, "\\apk_hacking_model\\apk_good.model")
            for root, dirs, files in os.walk(swift_path):
                for file in files:
                    if file.endswith(".apk"):
                        apk_file = os.path.join(root, file)
                        print(f"APK File: {apk_file}")
                        self.detect_malicious_app(apk_file, JASON, "\\apk_hacking_model\\apk_good.model")
                        delete_saved_package(swift_path)

        else:
            print("No model selected")
            messagebox.showwarning("No Model Selected", "Please select a model to scan the app")
        self.get_app_list_from_json_file()
        # app_extractor.delete_saved_package()

    def pull_app_memory_dump(self):
        selected_app = self.pull_app_memo_dump_dropdown.get()
        print(f"{selected_app}")
        app_save_dir = os.join(immediate_dir, selected_app)  # This is the directory to save the app memory dump
        pull_app_memory_dump(selected_app, app_save_dir)  # This is the function to pull the app memory dump
        print("Pull App Memory Dump ...")

    def phone_hardware_firmware(self, state):
        # 0001 Mobile phone hardware and firmware flame****************************
        self.initialize_tabview()  # Initial tab view elements
        self.mobile_phone_view_frame = customtkinter.CTkTabview(self.root_ctk, width=250, state=state)
        self.mobile_phone_view_frame.grid(row=0, column=2, padx=(20, 0), pady=(20.0, 0), sticky="nsew")
        self.mobile_phone_view_frame.grid_rowconfigure(4, weight=1)
        mobile_phone_tab = self.mobile_phone_view_frame.add("Mobile Hardware")

        # status = 'normal' if self.main_frame_color in ["#6BD0FF", "#CC3300"] else 'disabled'
        status = self.object_state()
        set_widgets_status_in_frame(self.mobile_phone_view_frame, status)

        # Mobile phone flame button and other attributes
        phone_detect_button = customtkinter.CTkButton(mobile_phone_tab, text="Phone Hardware Detect", state=status,
                                                      command=self.get_hardware_details)
        phone_detect_button.grid(row=1, column=1, padx=20, pady=(10, 10))
        self.hardware_text_box = frame(mobile_phone_tab, "Phone Hardware >>>>\n ")
        self.hardware_text_box.grid(row=2, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        # self.text_box(2, 1, mobile_phone_tab, "Phone Hardware >>>> ")

        # Mobile phone firmware flame button and other attributes
        mobile_phone_firmware_tab = self.mobile_phone_view_frame.add("Phone Firmware")
        detect_firmware = customtkinter.CTkButton(mobile_phone_firmware_tab, text="Firmware Detect", state=status,
                                                  command=self.get_firmware)
        detect_firmware.grid(row=1, column=1, padx=20, pady=(10, 10))
        self.firmware_text_box = frame(mobile_phone_firmware_tab, "Phone Firmware >>>>\n ")
        self.firmware_text_box.grid(row=2, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        # self.text_box(2, 1, mobile_phone_firmware_tab, "Phone Firmware >>>> ")

    def firmware_hardware_analysis(self, state):
        self.initialize_tabview()  # Initial tab view elements
        self.mobile_phone_firmware_view_frame = customtkinter.CTkTabview(self.root_ctk, width=250, state=state)
        self.mobile_phone_firmware_view_frame.grid(row=0, column=3, padx=(20, 0), pady=(20.0, 0), sticky="nsew")
        self.mobile_phone_firmware_view_frame.grid_rowconfigure(4, weight=1)
        firmware_analysis = self.mobile_phone_firmware_view_frame.add("Firmware Analysis")

        status = 'normal' if self.main_frame_color in ["#6BD0FF", "#CC3300"] else 'disabled'
        for child in self.mobile_phone_firmware_view_frame.winfo_children():
            child.configure(status)

        firmware_analysis_button = customtkinter.CTkButton(firmware_analysis, text="Firmware Analyser", state=status,
                                                           command=self.analyseFirmware)
        firmware_analysis_button.grid(row=1, column=3, padx=20, pady=(10, 10))

        # Add text box
        self.firmware_analysis_text_box = frame(firmware_analysis, "Firmware Analysis >>\n")
        # Insert the default text at the beginning of the textbox
        # Apply the default text style to the placeholder
        self.firmware_analysis_text_box.grid(row=2, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")

        hardware_analysis = self.mobile_phone_firmware_view_frame.add("Hardware Analysis")
        hardware_analysis_button = customtkinter.CTkButton(hardware_analysis, text="CPU Usage Analyser⚙", state=status,
                                                           command=self.analyseHardware, fg_color='#0C090A',
                                                           hover_color='#040720')
        hardware_analysis_button.grid(row=1, column=3, padx=20, pady=(10, 10))
        # Add text box
        self.hardware_analysis_text_box = frame(hardware_analysis, "Phone Hardware Analysis >> ")
        self.hardware_analysis_text_box.grid(row=2, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")

        self.hardware_analysis_tabs.append(hardware_analysis)
        self.buttons.append(hardware_analysis_button)
        # hardware_analysis_button.grid_remove()

    def firmware_hardware_hacking_detect(self, state):
        # Mobile phone hack detector
        self.initialize_tabview()  # Initial tab view elements
        self.mobile_phone_hacking_detector_view_frame = customtkinter.CTkTabview(self.root_ctk, width=250, state=state)
        self.mobile_phone_hacking_detector_view_frame.grid(row=1, column=2, padx=(20, 0), pady=(20.0, 0), sticky="nsew")
        self.mobile_phone_hacking_detector_view_frame.grid_rowconfigure(4, weight=1)
        phone_hack_detecting = self.mobile_phone_hacking_detector_view_frame.add("Detect Hacking")

        status = self.object_state()
        set_widgets_status_in_frame(self.mobile_phone_hacking_detector_view_frame, status)

        firmware_hack_detect = customtkinter.CTkButton(phone_hack_detecting, text="Detect Hardware Hacking",
                                                       state=status, command=lambda: self.detect_hack_hardware())
        firmware_hack_detect.grid(row=0, column=3, padx=20, pady=(10, 10))
        options: list = ['Drop Down Get Model', 'Pre-trained: model.h5']
        self.drop_down_per_based_model_var = tk.StringVar()
        self.permission_based_hack_detection = customtkinter.CTkOptionMenu(phone_hack_detecting, values=options,
                                                                           dropdown_hover_color='#000000',
                                                                           dropdown_text_color='#C0C0C0',
                                                                           dropdown_fg_color='#033E3E',
                                                                           fg_color='#006A4E', button_color='#004225',
                                                                           variable=self.drop_down_per_based_model_var)

        self.permission_based_hack_detection.set(options[0])
        self.permission_based_hack_detection.grid(row=1, column=3, padx=20, pady=(15.0, 0.0), sticky='nw')
        self.permission_based_hack_detection.configure(state=state)

        hardware_hack_detect = customtkinter.CTkButton(phone_hack_detecting, text="Detect Firmware Hacking",
                                                       state=status, command=lambda: self.detect_hack_firmware(),
                                                       fg_color='#B8860B', hover_color='#633a00')
        hardware_hack_detect.grid(row=2, column=3, padx=20, pady=(10, 10))
        # "\U0001F512" # Lock emoji \U0001F513 # Unlocked emoji
        hardware_hack_detect = customtkinter.CTkButton(phone_hack_detecting, text="OEM Firmware Security" +
                                                                                  "\U0001F512",
                                                       state=status, command=lambda: self.OEM_firmware_security_lock())
        hardware_hack_detect.grid(row=3, column=3, padx=20, pady=(10, 10))

        # ********************************* MORE ANALYTIC HACKING DETECTION) *********************************

        # Patch Level Status
        optional_analytical_detection = self.mobile_phone_hacking_detector_view_frame.add("Analytical Detecting")
        patch_level_status = customtkinter.CTkButton(optional_analytical_detection, text="Detect Patch-level",
                                                     state=status, command=detect_patch_level)
        patch_level_status.grid(row=1, column=3, padx=20, pady=(10, 10))

        # Boot state
        boot_state_status = customtkinter.CTkButton(optional_analytical_detection, text="Boot State Analysis",
                                                    state=status, command=boot_state_analysis)
        boot_state_status.grid(row=3, column=3, padx=20, pady=(10, 10))

        # SELinux Status
        SELinux_status = customtkinter.CTkButton(optional_analytical_detection, text="SELinux Analysis",
                                                 state=status, command=SELinux_analysis)
        SELinux_status.grid(row=2, column=3, padx=20, pady=(10, 10))

        # Rooted State Analysis
        root_state = customtkinter.CTkButton(optional_analytical_detection, text="isRooted Analysis",
                                             state=status, command=rooted_state)
        root_state.grid(row=4, column=3, padx=20, pady=(10, 10))

    # Then, you can use this method in your existing code:

    def login_state(self) -> str:
        return "enabled" if not self.login_status else "disabled"  # Controlled by login status

    def disconnect_button_status(self, status, *args):
        self.disconnect_button.destroy()
        self.disconnect_button = customtkinter.CTkButton(self.sideBarFrame, text="Disconnect Device",
                                                         command=self.disconnect_device)
        self.disconnect_button.grid(row=2, column=0, padx=20, pady=20)
        self.disconnect_button.configure(state=status)

    def main_color_changer(self, int_random: int):
        #   int_random = int_val  # random.randint(0, 1)
        if int_random == 0:
            self.main_frame_color = "#00538E"  # Inactive color
        if int_random == 1:
            self.main_frame_color = "#6BD0FF"  # Active color
        if int_random == 2:
            self.main_frame_color = THEM[2]  # Green
        if int_random == 3:
            self.main_frame_color = THEM[1]  # Yellow
        if int_random == 4:
            self.main_frame_color = "#CC3300"  # Hacked color Red

    def object_state(self):
        return 'normal' if self.main_frame_color in ["#6BD0FF", "#CC3300"] else 'disabled'

    def basic_user_interface(self):
        print("Basic User Interface ...")
        self.app_drop_down_menu.configure(state='disabled')
        self.drop_down_menu.configure(state='disabled')
        self.select_model.configure(state='disabled')
        self.pull_android_manifest_button.configure(state='disabled')
        self.execute_button.configure(state="disabled")
        self.scan_app.configure(state="disabled")

    def advanced_user_interface(self):
        print("Advanced User Interface ...")
        # self.malware_detection('enabled')
        self.app_drop_down_menu.configure(state='enabled')
        self.drop_down_menu.configure(state='enabled')
        self.select_model.configure(state='enabled')
        self.pull_android_manifest_button.configure(state='enabled')
        self.execute_button.configure(state="enabled")
        self.scan_app.configure(state="enabled")

    def connect_device(self):
        # You can replace the print statement with your connection code.
        start_adb_server()
        self.connection_progress_bar.start()  # Progress bar starts
        self.live_text = "Connection >>>"  # Insert text at the end of the textbox
        self.side_bar_text_box.insert(tk.END, f"\n{self.live_text}")

        port = get_port_name()  # Imported
        phone = Phone(port)
        option_vendor = None
        option_device = None

        phone_option = enumerate_serial_phones()
        for device in phone_option:
            option_vendor = device['Vendor ID']
            option_device = device['Product ID']
        if devices:
            connector = MobilePhoneLowLevelConnector(path, target_name=devices[0])
            create_phone_files_folder()  # Create a folder for the device if not exist
            connector.connect_phone_adb()
            self.general_textbox.insert(tk.END, connector.detect_device()[1] + f"@ Port {port}" + '\n')
            self.general_textbox.insert(tk.END, f"Vendor ID: {phone.return_vendor()} | {option_vendor} "
                                                f"Device ID: {phone.return_device()} | {option_device}\n")

        else:
            self.general_textbox.insert(tk.END, "No ADB device connected\n")
            return

        self.main_frame_color = "#6BD0FF"  # Active color
        self.disconnect_button_status("enabled")
        self.mobile_phone_graphics.destroy()
        self.mobile_phone_firmware_view_frame.destroy()
        self.mobile_phone_view_frame.destroy()
        self.mobile_phone_hacking_detector_view_frame.destroy()
        self.main_color_changer(1)
        self.phone('enabled')  # boot enabled tab
        self.firmware_hardware_analysis('enabled')  # boot enabled tab
        self.phone_hardware_firmware('enabled')
        self.firmware_hardware_hacking_detect('enabled')
        # self.drop_down_menu.configure(state='enabled')
        self.connection_progress_bar.start()  # Progress bar stops
        self.basic_user_button.configure(state='normal')
        self.advanced_user_button.configure(state='normal')

    def disconnect_device(self):
        # You can replace the print statement with your disconnection code.
        stop_adb_server()
        print("Device Disconnected")
        self.live_text = "Disconnection >>>"  # Insert text at the end of the textbox
        self.side_bar_text_box.insert(tk.END, f"\n{self.live_text}")
        self.connection_progress_bar.stop()  # Progress bar stops
        self.mobile_phone_graphics.destroy()
        self.connection_progress_bar.start()  # Progress bar stops
        self.main_color_changer(0)
        self.phone("disabled")
        self.disconnect_button_status("disabled")
        self.mobile_phone_firmware_view_frame.destroy()
        self.mobile_phone_view_frame.destroy()
        self.mobile_phone_hacking_detector_view_frame.destroy()
        self.firmware_hardware_analysis('disabled')
        self.phone_hardware_firmware('disabled')
        self.firmware_hardware_hacking_detect('disabled')
        # self.drop_down_menu.configure(state='disabled')
        self.execute_button.configure(state="disabled")
        self.pull_android_manifest_button.configure(state="disabled")
        self.connection_progress_bar.stop()  # Progress bar stops
        self.app_drop_down_menu.configure(state='disabled')
        self.pull_app_memo_dump_dropdown.configure(state='disabled')
        self.basic_user_button.configure(state='disabled')
        self.advanced_user_button.configure(state='disabled')
        self.app_drop_down_menu.configure(state='disabled')
        self.drop_down_menu.configure(state='disabled')
        self.select_model.configure(state='disabled')
        self.scan_app.configure(state='disabled')

    def get_hardware_details(self):
        print("Hardware Loaded")
        host = os.path.join(immediate_dir, 'device_properties.txt')
        # h = Hardware(immediate_dir + "\\device_properties.txt")
        if not os.path.exists(host):
            prop_puller = PullProp(immediate_dir)
            prop_puller.pull_and_save_properties()
        h = Hardware(host)
        extract_info = extract_device_info(immediate_dir, 'device_properties.txt')[0]
        self.hardware_text_box.insert(tk.END, extract_info + '\n' + h.run() + '\n')
        report.append_result(f"{extract_info}", h.run())

    def get_firmware(self):
        # print(detect_firmware_on_adb_device().split(',')[0])
        # self.firmware_text_box.insert(tk.END, detect_firmware_on_adb_device('{', '}').split(',')[0] + '\n')
        firmware_ver = get_firmware_version(immediate_dir, 'device_properties.txt')
        self.firmware_text_box.insert(tk.END, f"Pulled Firmware:"
                                              f"{firmware_ver}\n")
        self.firmware_text_box.insert(tk.END, f"LIVE FIRMWARE: \n")
        self.firmware_text_box.insert(tk.END, {detect_firmware_on_adb_device()[0]})
        self.firmware_text_box.insert(tk.END, f"\n{detect_firmware_on_adb_device()[1]}")
        report.append_result(f"Pulled Firmware: {firmware_ver}", f"LIVE FIRMWARE: {detect_firmware_on_adb_device()[0]}")

    def if_isProp(self):
        if not os.path.exists(host):
            prop_puller = PullProp(immediate_dir)
            prop_puller.pull_and_save_properties()

    def analyseFirmware(self):  # TODO pass firmware: Firmware
        self.live_text = 'Analysis Begins ...'
        self.firmware_analysis_text_box.insert(tk.END, f"{self.live_text}\n")
        print(f"{COLOR[2]}This Firmware Analysis Begins{COLOR[4]}")
        host = os.path.join(immediate_dir, 'device_properties.txt')
        # h = Hardware(immediate_dir + "\\device_properties.txt")
        if not os.path.exists(host):
            prop_puller = PullProp(immediate_dir)
            prop_puller.pull_and_save_properties()
        f_analysis = FirmwareAnalyser(None)
        f_analysis_result = f_analysis.analyse(immediate_dir)
        self.firmware_analysis_text_box.insert(tk.END, f"PatchLevel: {f_analysis_result[0]}\nSelinux: "
                                                       f"{f_analysis_result[1]}\nBoot State: {f_analysis_result[2]}\n"
                                                       f"{f_analysis_result[3]}\n")
        self.general_textbox.insert(tk.END, f"***Analytical Detection***\nSecurity Patch Level {f_analysis_result[0]}\n"
                                            f"→The Bootloader, Kernel Not Tampered\nSELinux is ENFORCING --> ACTIV\n"
                                            f"Device No Rooted State Detected\n***Analytical Detection END***\n")
        self.live_text = 'Analysis ENDS'
        self.firmware_analysis_text_box.insert(tk.END, f"{self.live_text}\n")
        # Patch Level Computation Begins ....
        current_date = datetime.datetime.fromtimestamp(time.time()).date()
        patch_date = datetime.datetime.strptime(f_analysis_result[0], "%Y-%m-%d").date()
        time_difference = current_date - patch_date
        months_old = time_difference.days // 30  # Approximate months

        if f_analysis_result[1] == (1, 1) and f_analysis_result[2] == 'GREEN' and f_analysis_result[3] \
                == 'Root State OEM':
            self.main_frame_color = "#6BD0FF"  # Active color
            self.main_color_changer(2 if months_old <= 1 else 3)
            report.append_result("Security Patch Level", f_analysis_result[0], "GREEN" if months_old <= 1 else "YELLOW")
            print(f"{COLOR[1]}Patch Level Passed{COLOR[4]}\n" if months_old <= 1 else
                  f"{COLOR[0]}Patch Level Failed!!{COLOR[4]}\n")
            report.append_result("SELinux", "[1,1]", "GREEN")
            report.append_result("Boot State", f_analysis_result[2], "GREEN")
            report.append_result("Root State", f_analysis_result[3], "GREEN")
        else:
            self.main_frame_color = "#6BD0FF"  # Active color
            self.main_color_changer(4)
            report.append_result("SELinux", "Not [1,1]", "RED")
            report.append_result("Boot State", f_analysis_result[2], "RED")
            report.append_result("Root State", f_analysis_result[3], "RED")
            # TODO **********************************************************************TODO
            print(f"{COLOR[0]}Check the Report to Determine Failed Property {COLOR[0]}")

        self.mobile_phone_graphics.destroy()
        self.phone('enabled')

        print(".....\n" * 2)
        print(f"{COLOR[1]}This Firmware Analysis is Complete, See the Report{COLOR[4]}")

    def analyseHardware(self):  # TODO hardware: Hardware
        self.live_text = 'Analysis Begins ...'
        print(f"{COLOR[2]}This Hardware Analysis Begins{COLOR[4]}")
        self.hardware_analysis_text_box.insert(tk.END, f"{self.live_text}\n")
        h = Hardware(None)
        cpu_data = h.collect_cpu_data(f"{prop[2]}", 1)
        cpu_analysis = h.predict_cpu_usage_anomaly(cpu_data)
        if cpu_analysis[0].empty:
            self.mobile_phone_graphics.destroy()
            self.main_color_changer(2)
            self.phone('enabled')
            self.hardware_analysis_text_box.insert(tk.END, f"CPU Usage: {cpu_analysis[1]['CPU Usage (%)'].mean()}\n")
            self.hardware_analysis_text_box.insert(tk.END, f"No Anomalies Detected.\n")
            self.general_textbox.insert(tk.END, f"CPU Usage: {cpu_analysis[1]['CPU Usage (%)'].mean()}\n")
            # report.append_result(f"CPU Usage: {cpu_analysis[1]['CPU Usage (%)'].mean()}", "No Anomalies Detected.",
            #                      "GREEN")
        else:
            self.mobile_phone_graphics.destroy()
            self.main_color_changer(4)
            self.phone('enabled')
            self.hardware_analysis_text_box.insert(tk.END, f"CPU Usage: {cpu_analysis[1]['CPU Usage (%)'].mean()}\n")
            self.hardware_analysis_text_box.insert(tk.END, f"Anomalies Detected.\n")
            self.general_textbox.insert(tk.END, f"CPU Usage: {cpu_analysis[1]['CPU Usage (%)'].mean()}\n")
        # report.append_result(f"CPU Usage: {cpu_analysis[1]['CPU Usage (%)'].mean()}", "Anomalies Detected.", "RED")
        report.append_result("CPU Usage", cpu_analysis[1]['CPU Usage (%)'].mean(),
                             "GREEN" if cpu_analysis[0].empty else "RED")

    def detect_hack_firmware(self):
        print(Fore.GREEN, "Permission Based Hack Detection Started", Fore.RESET)
        model_token: str = self.drop_down_per_based_model_var.get()
        print(model_token)
        if model_token == 'Pre-trained: model.h5':
            print(model_token)
            self.general_textbox.insert(tk.END, "Permission Based Hack Detection Started>\n")
            print(Fore.GREEN, "Permission Based Hack Detection Started", Fore.RESET)
            from PermissionBasedHackingDetection import MalwarePredictor
            # Permission Based Hack Detection
            directory = immediate_dir + "\\permissions.txt"
            if not os.path.exists(directory):
                save_permissions(immediate_dir, 'permissions.txt')  # permissions saved in the directory
            permissions_file_path = directory
            predictor = MalwarePredictor()
            features = predictor.extract_features(permissions_file_path)
            predicted_class = predictor.predict(features)
            print(Fore.YELLOW, f"The predicted class is: {Style.RESET_ALL}{predicted_class}")
            if predicted_class == b'S':
                print(Fore.RED, "The app exhibits malicious behavior.", Style.RESET_ALL)
                self.general_textbox.insert(tk.END, "The app exhibits malicious behavior.\n")
                report.append_result("Phone Scanned", "MALICIOUS", "RED")
                report.append_result("Pretected Class", predicted_class, "RED")
                self.mobile_phone_graphics.destroy()
                self.main_color_changer(4)
                self.phone('enabled')
            else:
                print(Fore.CYAN, "The app is benign.", Style.RESET_ALL)
                self.general_textbox.insert(tk.END, "The app is benign.\n")
                self.mobile_phone_graphics.destroy()
                self.main_color_changer(2)
                self.phone('enabled')
                self.general_textbox.insert(tk.END, f"The predicted class is: {predicted_class}\n")
                report.append_result("Phone Scanned", "Not MALICIOUS Behaviors Detected", "GREEN")
                report.append_result("Pretected Class", predicted_class, "GREEN")
            self.general_textbox.insert(tk.END, "Permission Based Hack Detection Complete>\n")
            print(Fore.GREEN, "Permission Based Hack Detection Complete", Fore.RESET)
        else:
            messagebox.showwarning(message="Select Pre-Trained: model.h5")
            print(Fore.RED, "Please Select Model to Continue", Fore.RESET)
            self.general_textbox.insert(tk.END, "Please Select Pretrained Model>\n")

    def OEM_firmware_security_lock(self):
        host = os.path.join(immediate_dir, 'device_properties.txt')
        # h = Hardware(immediate_dir + "\\device_properties.txt")
        if not os.path.exists(host):
            prop_puller = PullProp(immediate_dir)
            prop_puller.pull_and_save_properties()
        value = FirmwareAnalyser(None).get_oem_system_security_level(host)
        report.append_result("OEM Security Lock", 0, "GREEN" if value == "0" else "YELLOW")
        self.main_frame_color = "#6BD0FF"  # Active color
        self.main_color_changer(2 if value == "0" else 3)
        if value == "1":
            self.general_textbox.insert(tk.END, f"OEM Firmware Security Lock: OFF\n")
            print(f"OEM Firmware Security Lock: OFF\n")
            messagebox.showwarning("EOM Security Lock", 'OEM Firmware Security Lock: OFF')
        elif value == "0":
            self.general_textbox.insert(tk.END, f"OEM Firmware Security Lock: ON\n")
            messagebox.showinfo("EOM Security Lock", 'OEM Firmware Security Lock: ON')
            print(f"OEM Firmware Security Lock: ON\n")
        else:
            self.general_textbox.insert(tk.END, f"OEM Firmware Security Lock: Unknown\n")
            messagebox.showwarning("EOM Security Lock", 'OEM Firmware Security Lock: Unknown')
            print(f"OEM Firmware Security Lock: Unknown\n")
        self.mobile_phone_graphics.destroy()
        self.phone('enabled')

        if value == "1":
            self.general_textbox.insert(tk.END, f"OEM Firmware Security Lock: OFF\n")
            report.append_result("OEM Security Lock", "1", "RED")
            print(f"OEM Firmware Security Lock: OFF\n")
        elif value == "0":
            self.general_textbox.insert(tk.END, f"OEM Firmware Security Lock: ON\n")
            report.append_result("OEM Security Lock", "0", "GREEN")
            print(f"OEM Firmware Security Lock: ON\n")
        else:
            self.general_textbox.insert(tk.END, f"OEM Firmware Security Lock: Unknown\n")
            report.append_result("OEM Security Lock", "Null", "YELLOW")
            print(f"OEM Firmware Security Lock: Unknown\n")

    def detect_hack_hardware(self):
        print("Hardware hack detection ran")
        self.general_textbox.insert(tk.END, "Hardware Detecting>\n")

    def monitor_system(self, system):
        pass

    def phone(self, status):
        # Mobile phone graphics
        self.initialize_tabview()  # Initial tab view elements
        self.mobile_phone_graphics = customtkinter.CTkTabview(self.root_ctk, width=250, state=status)
        self.mobile_phone_graphics.grid(row=1, column=3, padx=(20, 0), sticky="nsew")
        self.mobile_phone_graphics.grid_rowconfigure(4, weight=1)
        phone_demo = self.mobile_phone_graphics.add("Phone")
        flame_width = 100
        flame_height = 210
        phone_screen_graphics = customtkinter.CTkFrame(phone_demo, width=flame_width, height=flame_height,
                                                       fg_color=self.main_frame_color)

        # Pack it in the center of the tab
        # Pack it in the center top of the tab
        phone_screen_graphics.pack(expand=True, anchor="n")
        # self.mobile_phone_graphics.configure(state='enabled')

        # Create symbols for network and battery
        network_symbol = "📶"  # This can be replaced with an image for a more accurate representation
        battery_symbol = "🔋"  # This can be replaced with an image for a more accurate representation
        call_symbol = "📞"  # Telephone receiver
        mute_symbol = "🔇"  # Muted speaker
        key_up_symbol = "⬆"  # Upward-pointing arrow symbol
        return_symbol = "↩"

        '''Add network, battery, time and date at the top of the phone screen'''
        # Add network label in the top left corner of the frame
        network_label = customtkinter.CTkLabel(phone_screen_graphics, text=network_symbol,
                                               fg_color=self.main_frame_color,
                                               text_color='black', font=("Arial", 18))
        network_label.place(x=2, y=2, anchor="nw")

        # Add battery label in the top right corner of the frame
        battery_label = customtkinter.CTkLabel(phone_screen_graphics, text=battery_symbol,
                                               fg_color=self.main_frame_color,
                                               text_color='black', font=("Arial", 18))
        battery_label.place(x=flame_width - 2, y=flame_height - 208, anchor="ne")  # WAS 98

        '''Get current date and time'''
        # current_time = datetime.now().strftime("%H:%M") # *** works with from datetime import datetime
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime("%H:%M")
        # current_date = datetime.now().strftime("%Y-%m-%d") # *** works with from datetime import datetime
        current_date = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")
        current_time_label = customtkinter.CTkLabel(phone_screen_graphics, text=current_time,
                                                    fg_color=self.main_frame_color, text_color='black',
                                                    font=("Arial", 18))
        x__ = (flame_width / 2.0)  # WAS 2
        # y__ = (flame_height / 2) + 10
        y__ = (16.0 / 21.0) * flame_height
        current_time_label.place(x=flame_width - x__, y=flame_height - y__, anchor="center")
        '''Get current date and time'''
        current_date_label = customtkinter.CTkLabel(phone_screen_graphics, text=current_date,
                                                    fg_color=self.main_frame_color,
                                                    text_color='black',
                                                    font=("Arial", 12))
        _y = 40  # (flame_height / 2) - 10  # WAS 2
        _y = (16.0 / 21.0) * flame_height - 20
        current_date_label.place(x=flame_width - x__, y=flame_height - _y, anchor="center")

        # Call Button *********************
        call_button = customtkinter.CTkLabel(phone_screen_graphics, text=call_symbol, text_color='white', height=21,
                                             width=21, font=("Arial", 16), bg_color='#000033')
        call_button.place(x=flame_width - x__, y=flame_height - 115, anchor="center")  # WAS 15

        # Disable all child widgets within the frame
        button_bg_state = ''
        for child in phone_screen_graphics.winfo_children():
            if self.main_frame_color == "#6BD0FF" or self.main_frame_color == "#CC3300":
                child.configure(state="normal")
                button_bg_state = 'normal'
            else:  # self.main_frame_color == "#00538E":
                child.configure(state="disabled")
                button_bg_state = 'disabled'

        """ Place buttons in the phone graphics"""

        row = flame_width - 98.0
        col = flame_height - 90.0
        x_ = 0.0

        one_7 = []
        for i in range(1, 10):
            one_7.append(i)
        for i in range(8):  # Increase the range for placing four buttons
            btn = customtkinter.CTkButton(phone_screen_graphics, text=f"{one_7[i]}", width=2, height=1,
                                          text_color='#F9E79F', fg_color='#2d2766', state=button_bg_state,
                                          font=("Arial", 10), command=None)
            btn.place(x=row + x_, y=col)
            x_ += 14.0  # Increment x_ for horizontal spacing between buttons

        '''Generate some qwerty keyboard by abstracting a QWERTY keyboard layout'''
        lower = list(string.ascii_lowercase)
        lowercase_alphabets = [
            lower[16], lower[22], lower[4], lower[17], key_up_symbol, lower[19], lower[24], lower[20], lower[8],
            lower[14], return_symbol, lower[15], lower[0], lower[18], lower[3], lower[5], lower[6], lower[7], lower[9],
            lower[10], '#', lower[11], lower[25], lower[23], lower[2], lower[21], lower[1], lower[13], lower[12]
        ]

        eight_zero = [8, 9, 0] + lowercase_alphabets
        x_ = 0.0
        for i in range(8):  # Increase the range for placing four buttons
            btn = customtkinter.CTkButton(phone_screen_graphics, text=f"{eight_zero[i]}", width=2, height=1,
                                          text_color='#F9E79F', fg_color='#2d2766', state=button_bg_state,
                                          font=("Arial", 10), command=None)
            btn.place(x=row + x_, y=col + 22.0)
            x_ += 14.0  # Increment x_ for horizontal spacing between buttons

        x_ = 0.0
        start_index = lowercase_alphabets.index(key_up_symbol)
        lowercase_alphabets = lowercase_alphabets[start_index:] + lowercase_alphabets[:start_index]
        for i in range(7):  # Increase the range to 4 for placing four buttons
            btn = customtkinter.CTkButton(phone_screen_graphics, text=f"{lowercase_alphabets[i]}", width=2, height=1,
                                          text_color='#F9E79F', fg_color='#2d2766', state=button_bg_state,
                                          font=("Arial", 10), command=None)
            btn.place(x=(row - 1.0) + x_, y=col + 44.0)
            x_ += 14  # Increment x_ for horizontal spacing between buttons

        x_ = 0.0
        start_index = lowercase_alphabets.index('#')
        lowercase_alphabets = lowercase_alphabets[start_index:] + lowercase_alphabets[:start_index]
        for i in range(7):  # Increase the range for placing four buttons
            btn = customtkinter.CTkButton(phone_screen_graphics, text=f"{lowercase_alphabets[i]}", width=2, height=1,
                                          text_color='#F9E79F', fg_color='#2d2766', state=button_bg_state,
                                          font=("Arial", 10), command=None)
            btn.place(x=(row - 1.0) + x_, y=col + 66.0)
            x_ += 14.0  # Increment x_ by horizontal spacing between buttons

    def initialize_tabview(self):
        self.root_ctk.grid_rowconfigure(0, weight=1)
        self.root_ctk.grid_columnconfigure(1, weight=1)

    @property
    def create_help_menu_html(self):
        help_menu_text = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    .green { color: green; }
                    .red { color: red; }
                    .blue { color: blue; }
                    .grey { color: grey; }
                    .black { color: black; }
                    table {
                        border-collapse: collapse;
                        width: 60%;
                    }
                    th, td {
                        border: 1px solid black;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                </style>
            </head>
            <body>
            <h2 style="color: blue;">Detecting Mobile Phone Firmware Hardware Hacking</h2>
            <h2 style="color: red;"> -------------------User Menu--------------------</h2>
        
            <h2>User Interface (GUI)</h2>
            <table>
                <tr>
                    <th class="green">GUI Commands</th>
                    <th class="green">Description</th>
                </tr>
                <tr>
                    <td class="red">Them Switching</td>
                    <td class="red">Change Theme to Dark, Light & System Defaulted.</td>
                </tr>
                <!-- Add more rows for other commands here -->
            </table>
        
            <h2>System Commands </h2>
            <table>
                <tr>
                    <th class="green">User Commands</th>
                    <th class="green">Description</th>
                </tr>
                <tr>
                    <td class="red">Connect Phone <strong>-- BUTTON</strong></td>
                    <td class="red">>>> Click to connect Phone</td>
                </tr>
                <tr>
                    <td class="grey"> Pull File <strong>--ITEMS_STORE</strong></td>
                    <td class="grey"> Pull Selected File </td>
                </tr>
                <tr>
                    <td class="red"> Drop Down for Files <strong>COMBO_BOX</strong></td>
                    <td class="red"> Drop Down Selection Apps </td>
                </tr>
                <tr>
                    <td class="red"> Pull File <strong>-- BUTTON</strong></td>
                    <td class="red"> Pull Selected file </td>
                </tr>
                <tr>
                    <td class="red"> Pull App AndroidManifest.xml <strong>-- BUTTON</strong></td>
                    <td class="red"> Pull App Android Manifest </td>
                </tr>
                <tr>
                    <td class="red">--------------</td>
                    <td class="red">------------------</td>
                </tr>
                <tr>
                    <td class="grey">Malware Detection <strong>ITEMS_STORE</strong></td>
                    <td class="grey">Malicious Apps Detection House </td>
                </tr>
                <tr>
                    <td class="blue"><strong> Select Model COMBO_BOX<strong></td>
                    <td class="blue"><strong> !Select ML Model for Analysis/Prediction</strong></td>
                </tr>
                <tr>
                    <td class="red">Analyse Saved APK <strong>-- BUTTON</strong></td>
                    <td class="red">Scan Immidiate App to Detect If Malicious </td>
                </tr>
                <tr>
                    <td class="red"> Drop Down Select App To Scan <strong>-- COMBO_BOX<strong></td>
                    <td class="red"> Here you Chose App To Scan Using ML </strong></td>
                </tr>
                <tr>
                    <td class="red"> Scan App <strong>-- BUTTON</strong></td>
                    <td class="red">>>>Scan App Selected and Wait for Results<br>>>>.jason Stats Report is Now Saved </td>
                </tr>
                <tr>
                    <td class="red"> Scan All App <strong>-- BUTTON</strong></td>
                    <td class="red">>>>Select All Apps Scan All & Wait for Results<br>>>>All.jason Stats Report is Now Saved </td>
                </tr>
                <tr>
                    <td class="grey">Mobile Hardware <strong>ITEMS_STORE</strong></td>
                    <td class="grey">Mobile Phone Hardware Properties House </td>
                </tr>
                <tr>
                    <td class="red"> Phone Hardware Detect <strong>-- BUTTON</strong></td>
                    <td class="red">>>>Detect Phone Key Hardware Components</td>
                </tr>
                <tr>
                    <td class="grey">Mobile Firmware <strong>ITEMS_STORE</strong></td>
                    <td class="grey">Mobile Phone Firmware Properties House </td>
                </tr>
                <tr>
                    <td class="red"> Phone Firmware Detect <strong>-- BUTTON</strong></td>
                    <td class="red">>>>Detect Phone Firmware Properties</td>
                </tr>
                <tr>
                    <td class="grey">Firmware Analysis  <strong>ITEMS_STORE</strong></td>
                    <td class="grey">Mobile Phone Firmware Analysis House </td>
                </tr>
                <tr>
                    <td class="red">Firmware Analyser<strong>-- BUTTON</strong></td>
                    <td class="red">Analyse Phone Firmware Will Dectect<br> >>> Patch-level<br>>>> Boot State<br>>>> 
                    SELinux<br>>>> isRooted State</td>
                </tr>
                <tr>
                    <td class="grey">Hardware Analysis  <strong>ITEMS_STORE</strong></td>
                    <td class="grey">Mobile Phone Hardware Analysis House </td>
                </tr>
                <tr>
                    <td class="red"> Hardware Analysis <strong>-- BUTTON</strong></td>
                    <td class="red">>>>Analyse Phone Firmware Will Dectect</td>
                </tr>
                <tr>
                    <td class="grey">Detect Hacking  <strong>ITEMS_STORE</strong></td>
                    <td class="grey">Hacking Detector House </td>
                </tr>
                <tr>
                    <td class="red"> Detect Hardware Hacking<strong>-- BUTTON</strong></td>
                    <td class="red">>>>Detect if Phone Hardware Has Been Compromised</td>
                </tr>
                <tr>
                    <td class="red"> Detect Firmware Hacking <strong>-- BUTTON</strong></td>
                    <td class="red">>>>Detect if Phone Firmware Has Been Compromised</td>
                </tr>
                <tr>
                    <td class="grey">Analytical Detecting  <strong>ITEMS_STORE</strong></td>
                    <td class="grey">Analysitic Detection House </td>
                </tr>
                <tr>
                    <td class="red"> Detect Patch-level <strong>-- BUTTON</strong></td>
                    <td class="red">>>>This Detects Patch Level Installed</td>
                </tr>
                <tr>
                    <td class="red"> Boot State Analysis <strong>-- BUTTON</strong></td>
                    <td class="red">>>>This Detects if Boot State Has Changed</td>
                </tr>
                <tr>
                    <td class="red"> SELinux Analysis <strong>-- BUTTON</strong></td>
                    <td class="red">>>>This Detects if SELinux State has Been Modified </td>
                </tr>
                <tr>
                    <td class="red"> isRooted Analysis <strong>-- BUTTON</strong></td>
                    <td class="red">>>>This Detects if Phone has Been Roots !Serious Vulunable State</td>
                </tr>
                <!-- Add rows for ADB options here -->
            </table>
        
            <h2>Results</h2>
            <table>
                <tr>
                    <th class="green">Outcomes</th>
                    <th class="green">Description</th>
                </tr>
                <tr>
                    <td class="red">Phone COLOR = LIGHT_BLUE</td>
                    <td class="red">Indicates Phone Successfully Connected in Developer Mode</td>
                </tr>
        
                <!-- Add more rows for other commands here -->
            </table>
        
            </body>
            </html>
        """
        return help_menu_text

    def save_and_read_help_menu_html(self, folder_path='doc'):
        try:
            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Define the full path for the HTML file
            html_file_path = os.path.join(folder_path, 'help.html')

            # Generate the HTML content

            # Check if the HTML file already exists
            if not os.path.exists(html_file_path):
                # Generate the HTML content
                help_menu_html = self.create_help_menu_html

                # Save the HTML content to the file
                with open(html_file_path, 'x') as file:
                    file.write(help_menu_html)

                print(f"Help HTML Generated Saved in {html_file_path}")
            else:
                print(f"Help HTML Opened")

            # Open the HTML file in a web browser
            import webbrowser
            webbrowser.open(html_file_path)

        except Exception as e:
            print(f"An error occurred: {e}")

    def detect_malicious_app(self, app_apk_to_analyse: str, results_destination_JSON: str, pre_trained_model):
        from APKRandomForestTrainer import APKAnalyseWithRandomForest
        import json
        we_can_analyse_with = APKAnalyseWithRandomForest("android-malware", "normal_apks")  # If you need training again
        # pass both files
        hacking_model_path = f"{os.path.dirname(os.path.abspath(__file__))}" + pre_trained_model

        # Provide the APK file to check
        # app_apk_to_analyse = input("Enter the path to the APK file to analyze: ")

        if not app_apk_to_analyse.endswith(".apk"):
            raise Exception("Please provide an .apk file.")

        # Check if model should be re-trained
        if not os.path.isfile(hacking_model_path):
            malware_folder = "android-malware"
            normal_folder = "normal_apks"

            if os.path.isdir(malware_folder) and os.path.isdir(normal_folder):
                apk_info = we_can_analyse_with.train_model(malware_apks_folder_path=malware_folder,
                                                           normal_apks_folder_path=normal_folder)
            else:
                raise Exception("Malware and normal APK folders not found for training.")

        # Check if the model exists
        if os.path.exists(hacking_model_path):
            outcomes, apk_data = we_can_analyse_with.identify_is_hacked(app_apk_to_analyse, hacking_model_path)

            if outcomes == 1:
                print(Fore.YELLOW + "Analysed App" + Fore.RED + "{}', Status--> Malicious!".format(
                    app_apk_to_analyse) + Style.RESET_ALL)
                self.general_textbox.insert(tk.END, f"Analysed App {app_apk_to_analyse}, Status--> Malicious!\n")
                report.append_result(f"Analysed App {app_apk_to_analyse}", "Status--> MALICIOUS", "RED")
                messagebox.showwarning(message="App is Malicious")
                self.main_frame_color = "#CC3300"  # Active color
                self.main_color_changer(4)
                self.phone('enabled')

            else:
                print(Fore.YELLOW + "Analysed App" + Fore.GREEN + "'{}', Status-->Not Malicious.".format(
                    app_apk_to_analyse) + Style.RESET_ALL)
                self.general_textbox.insert(tk.END, f"Analysed App {app_apk_to_analyse}, Status--> Not Malicious.\n")
                report.append_result(f"Analysed App {app_apk_to_analyse}", "Status--> NOT MALICIOUS", "GREEN")
                messagebox.showinfo(message="App is not Malicious")
                self.main_frame_color = "#6BD0FF"  # Active color
                self.main_color_changer(2)
                self.phone('enabled')

            # Provide the destination JSON file if needed
            # results_destination_JSON = input("Enter the path to the destination JSON file (optional): ")

            if results_destination_JSON.endswith(".json"):
                outcomes = True if outcomes == 1 else False
                package_name = apk_data.get("package", "Unknown Package")  # Get the package name or use a default
                data_to_write = {package_name: outcomes}
                # data_to_write = {apk_data["package"]: outcomes}

                if os.path.isfile(results_destination_JSON) and os.stat(results_destination_JSON).st_size != 0:
                    with open(results_destination_JSON) as json_file:
                        current_json_data = json.load(json_file)
                        current_json_data.update(data_to_write)
                        data_to_write = current_json_data

                with open(results_destination_JSON, 'w') as file_p:
                    json.dump(data_to_write, file_p, indent=4)
                print(Fore.CYAN + "Data written to JSON file." + Style.RESET_ALL)

            else:
                print(Fore.RED, "Destination file provided was not a JSON file.")
                print(Style.RESET_ALL)

        else:
            raise Exception("No model found. Please train the model.")


if __name__ == "__main__":
    run = SystemUserInterface()
    run.root_ctk.mainloop()
    # run.detect_malicious_app("ackman.placemarks.apk", "model_stats.json", "\\apk_hacking_model\\apk_good.model")
    # run.detect_malicious_app("a5starapps.com.drkalamquotes.apk", "model_stats.json",
    # "\\apk_hacking_model\\apk_good.model")
    # run.detect_malicious_app("QR.apk", "model_stats.json", "\\apk_hacking_model\\apk_good.model")
    # report.append_result(f"End of Analysis", "Status--> NOT MALICIOUS", "GREEN")
