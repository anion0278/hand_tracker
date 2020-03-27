import os
import sys
import webbrowser

def start_and_open():
    print("Starting tensorboard...")
    current_script_path = os.path.dirname(os.path.realpath(__file__))
    webbrowser.open_new("http://localhost:6006/#scalars")
    os.system('tensorboard --logdir='+os.path.join(current_script_path, "logs"))

if __name__ == "__main__":
    start_and_open()