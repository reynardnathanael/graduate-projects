This is a guidelines for running the Python program.
*Disclaimer: the original project using Windows 11 Home Edition

1.) To execute this program successfully, please install python first

https://www.python.org/downloads/

Please install the Python 3 version (the original project use Python 3.13.0)

2.) Install all the package required. This project using 3 imported packages: numpy, pandas, and matplotlib.

First, its recommended to install pip for easier installation (the original project use pip 24.2). 

https://pip.pypa.io/en/stable/installation/

For Windows: 
C:> py -m ensurepip --upgrade

For MacOS:
python -m ensurepip --upgrade

Pandas installation (the original project use pandas 2.2.3):
https://pypi.org/project/pandas/
pip install pandas

NumPy installation (the original project use numpy 2.1.3):
https://pypi.org/project/numpy/
pip install numpy

Matplotlib installation (the original project use matplotlib 3.9.2):
https://pypi.org/project/matplotlib/
pip install matplotlib

3.) Finally, open the folder HW3_113064710 with code editor (Visual Studio Code is recommended). Make sure that the calories.csv and exercise.csv are on the same folder with HW3.py before executing the program. Then, open the HW3.py file, and run the program using either the run button (will be available after adding python extension in VS Code) or using the command "python HW3.py" (the execution time usually takes 5-10 seconds). It's also suggested to install the Python extension in VS Code.

4.) When the program has successfully executed, then the terminal will shows the Mean Squared Error, and a figure/plot (Posterior Predictions with All Observations) will appears. If the figure/plot is closed, then another figure/plot (Posterior Predictions with Limited Observations) will appears.

5.) The program also can be accessed online with Google Colab with this link:
   https://colab.research.google.com/drive/1-9qpuDyiWUugC9o8rjEQFWymUvmcJ6ge

But first, the calories.csv and exercise.csv must be uploaded first in the Colab's cloud storage. After that, execute/run every cell from top to bottom.
