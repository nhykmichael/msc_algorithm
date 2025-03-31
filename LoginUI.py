import abc
import time
import tkinter as tk
from tkinter import ttk
from usable import color
from SimpleShiftEncryption import Encryption
import tkinter.messagebox as messagebox

# from FirwareHackDetectionInterface import SystemUserInterface

# from FirmwareHackDetector import User

"""
Sign up and Log in user interface
Name: MNA Ahimbisibwe
SN: 217005435
Model: UJ Mobile Phone Firmware Hacking Tool
"""
colors = color()  # color_list = [RED, GREEN, YELLOW, BLUE, WHITE]


class AbstractMethods:
    def __init__(self, formName, formWidth, formHeight):
        self.formName = formName
        self.formWidth = formWidth
        self.formHeight = formHeight

    @abc.abstractmethod
    def set_width(self, w: int) -> int:
        pass

    @abc.abstractmethod
    def set_height(self, h: int) -> int:
        pass

    @abc.abstractmethod
    def set_form_name(self, name: str):
        pass


def validate_integer(value):
    try:
        num = int(value)  # Try converting the input value to an integer
        if num >= 0:  # If the converted integer is greater than or equal to 0
            return num  # Return the validated integer
        else:
            # Show an error message dialog box stating that a valid integer greater than or equal to 0 is required
            messagebox.showerror("Input Error", "Please enter a valid integer larger than or equal to 0.")
            print(f"{value} is not a valid grid dimension value")
    except ValueError:  # If the conversion to an integer raises a ValueError
        try:
            num = int(float(value))  # Try converting the input value to a float first, then to an integer
            if num >= 0:  # If the converted integer is greater than or equal to 0
                return num  # Return the validated integer
            else:
                # Show an error message dialog box stating that a valid integer greater than or equal to 0 is required
                messagebox.showerror("Input Error", "Please enter a valid integer larger than or equal to 0.")
                print(f"{float(value)} is not a valid grid dimension value")
        except ValueError:  # If the conversion to a float or integer raises a ValueError
            # Show an error message dialog box
            messagebox.showerror("Input Error", "Please enter a valid integer or float.")

    return None  # Return None if the input value couldn't be validated as an integer or float


class LoginUI(AbstractMethods):
    def __init__(self, formName: str, formWidth, formHeight, strFileName, arrShifts):
        super().__init__(formName, formWidth, formHeight)
        self.arrShifts = arrShifts
        self.strFileName = strFileName
        self.root = tk.Tk()
        self.root.configure(bg="#060423")
        self.root.title(f"{formName}")
        w = validate_integer(formWidth)
        h = validate_integer(formHeight)
        self.root.geometry(f"{w}x{h}")

        # Create a Frame to hold form elements
        form_frame = ttk.Frame(self.root, style="Custom.TFrame")
        self.root.iconbitmap("uj.ico")
        form_frame.pack(expand=True, padx=50, pady=50)  # Add padding and expand the frame to fill space
        # Style for the custom frame
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#060423")  # Set the background color of the frame

        self.username_label = ttk.Label(form_frame, text="Username:", background="#060423", foreground="#E6F2A5")
        self.username_entry = ttk.Entry(form_frame)
        self.username_label.grid(row=0, column=0, sticky='e')  # Label placed on the left
        self.username_entry.grid(row=0, column=1, padx=10)  # Place entry field next to the label

        self.password_label = ttk.Label(form_frame, text="Password:", background="#060423", foreground="#E6F2A5")
        self.password_entry = ttk.Entry(form_frame, show="*")
        self.password_label.grid(row=1, column=0, sticky='e')  # Label placed on the left
        self.password_entry.grid(row=1, column=1, padx=10)  # Place entry field next to the label

        # Set default username and password
        self.username_entry.insert(0, "username")
        self.password_entry.insert(0, "password")

        self.submit_button = ttk.Button(form_frame, text="Submit", command=self.submit_form, style="Custom.TButton")
        self.submit_button.grid(row=2, column=1, pady=10)

        # Style for the custom buttons
        style = ttk.Style()
        style.configure("Custom.TButton", background="#478BE7", foreground="#0D2049")  # Set button colors

    def set_width(self, w: int) -> int:
        self.formWidth = validate_integer(w)
        self.root.geometry(f"{self.formWidth}x{self.formHeight}")
        return self.formWidth

    def set_height(self, h: int) -> int:
        self.formHeight = validate_integer(h)
        self.root.geometry(f"{self.formWidth}x{self.formHeight}")
        return self.formHeight

    def set_form_name(self, name: str):
        self.formName = name
        self.root.title(f"{self.formName}")

    def submit_form(self):
        username = self.username_entry.get()
        password = self.password_entry.get().upper()
        strShiftCryptPassword = Encryption(self.arrShifts)

        # Load login details
        stored_credentials = []
        with open(self.strFileName, 'r') as f:
            for line in f:
                if line.startswith("Username:") and "Password:" in line:
                    parts = line.split(',')  # Break line into 2 parts
                    if len(parts) == 2:
                        # line is taking the first half of our input (before the comma)
                        strSavedUserName = parts[0].split(':')[1].strip()
                        # line is taking the second half of our input (after the comma)
                        strSavedPassword = parts[1].split(':')[1].strip()
                        strCryptPassNoMore = strShiftCryptPassword.decrypt(strSavedPassword)  # Decrypt password
                        stored_credentials.append((strSavedUserName, strCryptPassNoMore))
        ''' If file is comma separated '''
        # strSavedUserName, strSavedPassword = line.strip().split(',')

        # Compare entered credentials with the saved ones
        if (username, password) in stored_credentials:
            messagebox.showinfo("Login Status", "Login Successful")
            self.run()
            print(f"{colors[1]}Login Successful >> {colors[4]}")
            # Perform desired action @TODO
            from FirwareHackDetectionInterface import SystemUserInterface
            system_interface = SystemUserInterface()
            system_interface.root_ctk.mainloop()

        else:
            messagebox.showerror("Login Status", f"Login Fail! Please Try Again")
            print(f"{colors[0]}Login details provided are incorrect{colors[4]}")
        # print(f"Username: {username}, Password: {password}")
        # Display an error message or perform any other actions for a failed login

    def run(self):
        self.root.destroy()


class SignUpUI(AbstractMethods):
    def __init__(self, formName: str, formWidth, formHeight, strFileName, arrShifts):
        super().__init__(formName, formWidth, formHeight)
        self.arrShifts = arrShifts
        self.strFileName = strFileName
        self.root = tk.Tk()
        self.root.configure(bg="#060423")
        self.root.title(f"{formName}")
        w = validate_integer(formWidth)  # Validate entry
        h = validate_integer(formHeight)  # Validate entry
        self.root.geometry(f"{w}x{h}")

        # Create a frame to hold the form elements
        form_frame = ttk.Frame(self.root, style="Custom.TFrame")  # Add style to the frame
        self.root.iconbitmap("uj.ico")
        form_frame.pack(expand=True, padx=50, pady=50)  # Add padding and expand the frame to fill the space

        # Style for the custom frame
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#060423")  # Set the background color of the frame

        self.sign_in_button = ttk.Button(form_frame, text="Login", command=self.call_sign_in_form, style="Custom"
                                                                                                         ".TButton")
        self.sign_in_button.grid(row=0, column=1, pady=10)  # Place the button below the password entry

        # Username label and entry
        self.or_label = ttk.Label(form_frame, text="or sign up", background="#060423", foreground="#E6F2A5")
        self.or_label.grid(row=2, column=1, pady=10)  # Place the label on the left side

        # Username label and entry
        self.username_label = ttk.Label(form_frame, text="Username:", background="#060423", foreground="#E6F2A5")
        self.username_entry = ttk.Entry(form_frame)
        self.username_label.grid(row=3, column=0, sticky="e")  # Place the label on the left side
        self.username_entry.grid(row=3, column=1, padx=10)  # Place the entry field next to the label

        # Password label and entry
        self.password_label = ttk.Label(form_frame, text="Password:", background="#060423", foreground="#E6F2A5")
        self.password_entry = ttk.Entry(form_frame, show="*")
        self.password_label.grid(row=4, column=0, sticky="e")  # Place the label on the left side
        self.password_entry.grid(row=4, column=1, padx=10)  # Place the entry field next to the label

        # Set default username and password
        self.username_entry.insert(0, "username")
        self.password_entry.insert(0, "password")

        # Submit button
        self.submit_button = ttk.Button(form_frame, text="Submit", command=self.submit_form, style="Custom.TButton")
        self.submit_button.grid(row=5, column=1, pady=10)  # Place the button below the password entry

        # Style for the custom buttons
        style = ttk.Style()
        style.configure("Custom.TButton", background="#478BE7", foreground="#0D2049")  # Set button colors

    def set_width(self, w: int) -> int:
        self.formWidth = w
        self.root.geometry(f"{self.formWidth}x{self.formHeight}")
        return self.formWidth

    def set_height(self, h: int) -> int:
        self.formHeight = h
        self.root.geometry(f"{self.formWidth}x{self.formHeight}")
        return self.formHeight

    def set_form_name(self, name: str):
        self.formName = name
        self.root.title(f"{self.formName}")

    def submit_form(self):
        username = self.username_entry.get()
        password = self.password_entry.get().upper()
        strShiftEncryptedPassword = Encryption(self.arrShifts)  # Encrypt password

        with open(self.strFileName, 'a') as f:
            f.write(f"Username: {username}, Password: {strShiftEncryptedPassword.encrypt(password)}\n")

        messagebox.showinfo("Sign Up Status", "Profile Successfully Created")
        print(f"{colors[3]}User credentials saved!{colors[4]}")

        self.call_sign_in_form()

    def call_sign_in_form(self):
        self.root.destroy()
        time.sleep(0.3)
        sign_in = LoginUI("Login", self.formWidth, self.formHeight, self.strFileName, self.arrShifts)
        sign_in.root.mainloop()


def validate(number) -> bool:
    if validate_integer(number):  # If validated passes return true
        return True
    else:
        return False  # Else if validated entry not pass return false


if __name__ == "__main__":
    intW = 500
    intH = 500.0
    # Define the list of shifts
    shifts = [4, 3, 1, 6, 7]  # encryption algorithm using poly alphabetic substitution
    filename = "user_credentials.txt"
    if validate(intW) and validate(intH):
        sign_up = SignUpUI("Sign Up Form", intW, intH, filename, shifts)
        sign_up.root.mainloop()

    else:
        messagebox.showerror("Error Message", f"Form size entry {intW} or {intH} is invalid")

