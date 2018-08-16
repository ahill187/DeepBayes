# DeepBayes

Unfolding the W boson momentum using DNN's and Bayesian unfolding.

## Getting Started

### Prerequisites

You will need to have Keras and Tensorflow installed. The current Keras version will not work with CUDA 9.2, however, so you will need to install CUDA 9.0 and cuDNN 7.0.
You will need ROOT installed on your computer. 

### Installing ROOT

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
+ Example:
```
$ cmake /home/ahill/root-6.14.02
```
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
