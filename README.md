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
### Installing DeepJetCore

The DeepJetCore master fork can be found here: https://github.com/DL4Jets/DeepJetCore. To use this package with DeepBayes, I have edited some of the files, so please use the forked version on my repository:
> https://github.com/ahill187/DeepJetCore

To install DeepJetCore:
```
$ cd <deep_learning_directory>
$ git clone https://github.com/ahill187/DeepJetCore.git
$ cd DeepJetCore/compiled
$ make -j 4
```
### Installing DeepML

The DeepML package was developed by Pedro da Silva, and can be found on GitLab. I have adapted the package to use with DeepBayes, so please use the version from my repository for DeepBayes. You can find the repository along with a README file at:
> https://github.com/ahill187/DeepML

To install DeepML:
```
$ cd <deep_learning_directory>
$ git clone https://github.com/ahill187/DeepML.git
```
Please note that DeepML and DeepJetCore must be in the same parent directory. The instructions for installing DeepML and DeepJetCore are modified from the original instructions at
> https://twiki.cern.ch/twiki/bin/view/Main/VpTNotes#Training_the_recoil_regression

## Installing DeepBayes

Though not necessary, I have simplified the install by putting it into the same directory as DeepML and DeepJetCore. Again, not mandatory, you can install it wherever you would like. <br>

To install DeepBayes:
```
$ cd <deep_learning_directory>
$ git clone https://github.com/ahill187/DeepBayes.git
```

## Tests - Toy Model

The DeepBayes model uses W recoil variables to reconstruct the W momentum. To test the model, however, we used Gaussian data. Please see <insert link here> for more information.

### Running Toy Model

```
cd DeepBayes/toy_model
python model.py
```
Wait for the prompt for you to input your settings. It will ask for a Plotting Directory; if you specify a full path, it will put the plots in the specified folder. If you specify a single name, it would create that folder in the parent directory.

The next prompt will ask if you would like to use the default settings. If you choose "Y", the model will use the default settings of:

15000 training epochs for the initial DNN training<br>
1000 epochs for each Bayesian iteration<br>
30 iterations of Bayesian unfolding<br>

Depending on the speed of your computer, this should take between 10 - 60 minutes to run. 

### Plots

You can view the plots in the plotting directory specified previously. <br>

Combined Plots 
> These plots show the unfolded training data, unfolded testing data, and the prediction. 

Training Plots
> These plots are to verify how well the initial model performed on the training data, and to monitor how the predictions change with the reweighting. They show the smeared training data, the unfolded training data, and the prediction for the training data.

Testing Plots
> These plots show how well the model is performing on new data. They show the smeared testing data, the unfolded testing data, and the prediction for the testing data.

Score
> The Score.png plot shows the loss for each iteration.

## Running the Model

1. The first time you run the model, you will need to edit the file name in the file DeepBayes/runRecoilRegression.sh. At the top of the file, there is a variable called TRAINPATH. 
```
TRAINPATH = <deep_learning_directory>/DeepML
```
2. The first time you train the model, you will need to convert the ROOT trees to Python:
```
$ cd <deep_learning_dir>/DeepML
$ sh <deep_learning_dir>/DeepBayes/runRecoil
Regression_AH.sh -r convert -m <num> -i <deep_
learning_dir>/DeepML/data/recoil_file_list.t
xt -w <output_directory>
```
Here "convert" specifies that we want to convert the trees. The variable <num> should be an integer, and specifies the model number to be used for Keras. The model numbers are defined in the file \textbf{DeepBayes/deep\_bayes/settings.py}, and the models are described in \textbf{DeepBayes/deep\_bayes/dnn\_models.py}. The "recoil\_file\_list.txt" is a text file containing the names of the ROOT files to convert, to be accessed via the CERN network. The <output\_directory> is the directory where the results will be.

3. Train the neural network.
```
$ cd <deep_learning_directory>/DeepBayes
$ python deep_bayes/model.py
```


