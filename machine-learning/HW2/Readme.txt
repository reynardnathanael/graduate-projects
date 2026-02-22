This is a guidelines for running the Python program.
*Disclaimer: the original project using Windows 11 Home Edition

1.) To execute this program successfully, please install python first

https://www.python.org/downloads/

Please install the Python 3 version (the original project use Python 3.13.0

2.) Install all the package required. This project using 6 imported packages: numpy, matplotlib, sklearn, seaborn, torch, and torchvision.

First, its recommended to install pip for easier installation (the original project use pip 24.2). 

https://pip.pypa.io/en/stable/installation/

For Windows: 
C:> py -m ensurepip --upgrade

For MacOS:
python -m ensurepip --upgrade

NumPy installation (the original project use numpy 2.1.3):
https://pypi.org/project/numpy/
pip install numpy

Matplotlib installation (the original project use matplotlib 3.9.2):
https://pypi.org/project/matplotlib/
pip install matplotlib

Scikit-learn installation (the original project use scikit-learn 1.6.1):
https://scikit-learn.org/stable/install.html
virtual environment is optional but strongly recommended, in order to avoid potential conflicts with other packages:

python -m venv sklearn-env
sklearn-env\Scripts\activate

pip install -U scikit-learn

Seaborn installation (the original project use seaborn 0.13.2):
https://pypi.org/project/seaborn/
pip install seaborn

Torch installation (the original project use torch 2.6.0):
https://pypi.org/project/torch/
pip install torch

Torchvision installation (the original project use torchvision 0.21.0):
https://pypi.org/project/torchvision/
pip install torchvision

3.) Finally, open the folder HW2_113064710 with code editor (Visual Studio Code is recommended). The dataset (images) must be in a folder called 'Data' in the same directory as the HW2.py file (the folder structure: Data > Data_test, Data_train). Then, open the HW2.py file, and run the program (usually takes 5-10 seconds). It's also suggested to install the Python extension in VS Code.

4.) When the program has successfully executed, then the terminal will shows the number of images loaded (1176 for training, 294 for validation, and 498 for test), and 2 figures will appears: PCA test and a sample image. 

5.) If the figures are closed, then the terminal will shows the epochs, accuracy, and loss score for the 2-layer neural network. Also, a figure (loss curves) will appears. If the figure is closed, another figure (decision regions) will appears.

6.) If the decision region's figure from the 2-layer neural network is closed, then the terminal will shows the epochs, accuracy, and loss score for the 3-layer neural network. Also, a figure (loss curves) will appears. If the figure is closed, another figure (decision regions) will appears.

7.) You can change the optimizer by searching for this code:
optimizer = 'SGD'
# optimizer = 'ADAM'

add (#) for commenting one of the optimizer that won't be used.

8.) The program also can be accessed online with Google Colab with this link:
   https://colab.research.google.com/drive/1gTkCO8XMCrIQiQvCJmX5m0kVR2yv2tEi

But first, in your Google Drive, there must be a folder called 'Dataset', with Data_test and Data_train folders inside. Then, execute all the cell, and the colab will ask for a permission first. By accepting that, the program will be running the same as the local (HW2.py) version. But, because the dataset is located in cloud, the images processing will be longer (usually about 5-10 minutes)
