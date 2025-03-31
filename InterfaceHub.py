"""
Prototype for Firmware/Hardware Mobile Phone Hacking Detection Tool Interface Bank
Name: MNA Ahimbisibwe
Version: Prototype A
"""


# Interface
class IUserInterface:
    def display(self):
        pass

    def get_input(self):
        pass

    def show_error(self, message):
        pass

    def show_success(self, message):
        pass


# Interface
class IDeviceConnector:
    def connect_device(self, mobile_phone):
        pass

    def disconnect_device(self, mobile_phone):
        pass


# Interface
class IScanner:
    def scan(self, mobile_phone):
        pass


# Interface
class IAnalyser:
    def analyse(self, input):
        pass

    def security_patch_level(self):
        pass


# Interface
class INotificationManager:
    def send_alert(self, user, message):
        pass

    def send_report(self, user, report):
        pass


# Interface
class IReportGenerator:
    def generate_report(self, analysis_results):
        pass
