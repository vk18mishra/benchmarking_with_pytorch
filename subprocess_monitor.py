# from subprocess import *
# import time
# Popen('python main.py')
# time.sleep(3)
# Popen('python realtime_monitoring.py')

import subprocess

subprocess.run("python realtime_monitoring.py & python main.py", shell=True)