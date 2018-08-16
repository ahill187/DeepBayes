# DeepBayes

Unfolding the W boson momentum using DNN's and Bayesian unfolding.

## Getting Started

### Prerequisites

You will need to have Keras and Tensorflow installed. The current Keras version will not work with CUDA 9.2, however, so you will need to install CUDA 9.0 and cuDNN 7.0.
You will need ROOT installed on your computer. 

### Installing ROOT

Instructions for installing ROOT are available at: <br>
> https://root.cern.ch/building-root <br>
I have simplified the instructions for the location-independent install below. 

1. Download ROOT *tar.gz file <br>
https://root.cern.ch/downloading-root <br>
Select the ROOT version (6.14/02) <br>
Select the operating system <br>
2. Extract the *tar.gz file:
```
$ tar -zxf root_<version>.tar.gz
```
3. Create a build directory:
```
$ mkdir <builddir>
$ cd <builddir>
```
4. Execute cmake command:
```
$ cmake <path_to_root_download/root-<version>>
```
> Example:
```
$ cmake /home/ahill/root-6.14.02
```
5. Build
```
$ cmake --build .
```
6. Run <br>
Each time you run ROOT, you must enter the following into terminal:
```
$ source <builddir>/bin/thisroot.sh
$ root
```
### Using PyROOT

If running Python in terminal, you can use the following commands to use PyROOT:
```
$ source <builddir>/bin/thisroot.sh
$ python
$ import ROOT
```
If you would like to use PyROOT in PyCharm, there are 3 options:
1. Add path to Project Structure:
```
Open PyCharm
File -> Settings -> Project: src -> Project Structure
+ Add Content Root
<builddir>/lib
```
2. Add the ROOT library to the system path
```
Open PyCharm
import sys
sys.path.extend['<builddir>/lib']
```
> Example:
```
sys.path.extend['/home/ahill/builddir/lib/]
```
3. Edit the pycharm.sh script
```
$ cd /opt/pycharm-<version>/bin
$ sudo gedit pycharm.sh
```
Add the following to the top of the script:
```
export ROOTSYS=$HOME/builddir
export PATH=$ROOTSYS/bin:$PATH
export LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$ROOTSYS/lib
export PYTHONSTARTUP=$HOME/.pythonstartup
```
### DeepML and DeepJetCore

The D

## Toy Example

For our toy example, we are generating our own testing and training data as follows:

### Training Data ("Monte Carlo Simulated"): 

z = Gaussian distributed data (default: mean = 0, sd = 0.6)

x = Smeared data: Gaussian random variable added to each point in z (default smearing: mean = 0, sd = 0.1)

### Testing Data ("Real Data"):

z = Gaussian distributed data (default: mean = 0, sd = 0.4)

x = Smeared data: Gaussian random variable added to each point in z (default smearing: mean = 0, sd = 0.1)

### Running Toy Model

```
cd DeepBayes
python model.py
```
Wait for the prompt for you to input your settings. It will ask for a Plotting Directory; if you specify a full path, it will put the plots in the specified folder. If you specify a single name, it would create that folder in the parent directory.

The next prompt will ask if you would like to use the default settings. If you choose "Y", the model will use the default settings of:

15000 training epochs for the initial DNN training<br>
1000 epochs for each Bayesian iteration<br>
30 iterations of Bayesian unfolding<br>

Depending on the speed of your computer, this should take between 10 - 60 minutes to run. 

### Plots

You can view the plots in the plotting directory specified previously. The number in the plot title indicates the iteration, i.e.

k = 0 is the initial iteration, after training the DNN on the training data<br>
k = 1 is the first Bayes iteration<br>
...<br>
k = 30 is the last Bayes iteration in the default data

#### Combined Plots

These plots show the unfolded training data, unfolded testing data, and the prediction from the model.

#### Training Plots

These plots show the smeared training data, the unfolded training data, and the prediction from the model.

#### Testing Plots

These plots show the smeared testing data, the unfolded testing data, and the prediction from the model.

#### Score

The Score.png plot shows the loss for each iteration. Lower loss means the algorithm is improving.
