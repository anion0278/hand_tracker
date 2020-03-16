import os
import sys
import subprocess
import webbrowser

current_script_path = os.path.dirname(os.path.realpath(__file__))

tb_url = "http://localhost:6006/#scalars"

webbrowser.open_new(tb_url) # if does not work for some cases, use the commented code

#if sys.platform=='win32':
#    os.startfile(tb_url)
#elif sys.platform=='darwin':
#    subprocess.Popen(['open', tb_url])
#else:
#    try:
#        subprocess.Popen(['xdg-open', tb_url])
#    except OSError:
#        print('Please open a browser on: '+tb_url)

print("Starting tensorboard...")
os.system('tensorboard --logdir='+os.path.join(current_script_path, "logs"))