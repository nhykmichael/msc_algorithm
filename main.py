from LoginUI import SignUpUI

"""
Prototype for Firmware/Hardware Mobile Phone Hacking Detection Tool Entry Point
Name: MNA Ahimbisibwe
Version: Prototype A
"""


class Main:
    def __init__(self):
        intW = 500
        intH = 500
        shifts = [4, 3, 1, 6, 7]  # encryption algorithm using poly alphabetic substitution
        filename = "user_credentials.txt"

        sign_up = SignUpUI("Sign Up Form", intW, intH, filename, shifts)
        sign_up.root.mainloop()


if __name__ == "__main__":
    main_app = Main()
