# DeepBayes

Unfolding the W boson momentum using DNN's and Bayesian unfolding.

## Getting Started

This project was developed as part of the CERN Summer Student 2018 program. It is recommended that you read through the introductory paper before using the package: <br>
> https://github.com/ahill187/DeepBayes/blob/optiplex/documentation/unfolding-w-boson(2).pdf

For further information, please contact:
> Ainsleigh Hill ainsleigh.hill@alumni.ubc.ca <br>
> Josh Bendavid Josh.Bendavid@cern.ch <br>
> Pedro da Silva Pedro.Silva@cern.ch

This project to run in the CMSSW environment at CERN. If you would like to do a local install, please see the file Local_Install.md.

### Prerequisites

### CMSSW

To set up the CMSSW environment for the first time:

```bash
$ cd
$ cmsrel CMSSW_10_2_0_pre5
$ cd CMSSW_10_2_0_pre5/src
$ cmsenv
```
For subsequent times:

```bash
$ cd CMSSW_10_2_0_pre5/src
$ cmsenv
```

### Installing DeepJetCore

The DeepJetCore master fork can be found here: https://github.com/DL4Jets/DeepJetCore. To use this package with DeepBayes, I have edited some of the files, so please use the forked version on my repository:
> https://github.com/ahill187/DeepJetCore

To install DeepJetCore:
```bash
$ cd CMSSW_10_2_0_pre5/src
$ cmsenv
$ mkdir <deep_learning_directory>
$ cd <deep_learning_directory>
$ git clone https://github.com/ahill187/DeepJetCore.git
$ cd DeepJetCore/compiled
$ make -j 4
```
### Installing DeepML

The DeepML package was developed by Pedro da Silva, and can be found on GitLab. I have adapted the package to use with DeepBayes, so please use the version from my repository for DeepBayes. You can find the repository along with a README file at:
> https://github.com/ahill187/DeepML

To install DeepML:
```bash
$ cd <deep_learning_directory>
$ git clone https://github.com/ahill187/DeepML.git
```
Please note that DeepML and DeepJetCore must be in the same parent directory. The instructions for installing DeepML and DeepJetCore are modified from the original instructions at
> https://twiki.cern.ch/twiki/bin/view/Main/VpTNotes#Training_the_recoil_regression

## Installing DeepBayes

Though not necessary, I have simplified the install by putting it into the same directory as DeepML and DeepJetCore. Again, not mandatory, you can install it wherever you would like. <br>

To install DeepBayes:
```bash
$ cd <deep_learning_directory>
$ git clone https://github.com/ahill187/DeepBayes.git
```

## Tests - Toy Model

The DeepBayes model uses W recoil variables to reconstruct the W momentum. To test the model, however, we used Gaussian data. Please see <insert link here> for more information.

### Running Toy Model

There are two files in the toy_model folder: model.py and model_bins.py. The first uses Gaussian distributions with equal binwidths, while the second uses variant binwidths with equal events (quantiles).
```bash
cd <deep_learning_dir>/DeepBayes
python toy_model/model.py
```
or
```bash
cd <deep_learning_dir>/DeepBayes
python toy_model/model_bins.py
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

1. The first time you run the model, you will need to edit the directories in the file DeepBayes/deep_bayes/runRecoilRegression_AH.sh. At the top of the file, there are variables called TRAINPATH and DEEPBAYES.
```
TRAINPATH = <deep_learning_directory>/DeepML
DEEPBAYES = <deep_learning_directory>/DeepBayes
```
2. The first time you train the model, you will need to convert the ROOT trees to Python:
```bash
$ cd <deep_learning_dir>/DeepML
$ sh <deep_learning_dir>/DeepBayes/deep_bayes/runRecoilRegression_AH.sh -r convert -m <num> -i <deep_learning_dir>/DeepML/data/recoil_file_list.txt -w <output_directory>
```
Here "convert" specifies that we want to convert the trees. The variable <num> should be an integer, and specifies the model number to be used for Keras. The model numbers are defined in the file DeepBayes/deep_bayes/settings.py, and the models are described in DeepBayes/deep_bayes/dnn_models.py. The "recoil_file_list.txt" is a text file containing the names of the ROOT files to convert, to be accessed via the CERN network. The <output_directory> is the directory where the results will be.

3. Train the neural network.
```bash
$ cd <deep_learning_directory>/DeepML
$ sh <deep_learning_dir>/DeepBayes/runRecoilRegression_AH.sh -r train -m <num> -i <deep_learning_dir>/DeepML/data/recoil_file_list.txt -w <output_directory>
```
This is the same as converting the ROOT trees, except that you need to set "-r train" instead of "-r convert".
