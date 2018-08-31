## Using DeepBayes on Local Computer

DeepBayes was designed using the CERN CMSSW environment. However, a local install could be useful if you do not have access to the CERN network, or if you would like to use an IDE such as PyCharm for development (this can be difficult with cmsenv). Unfortunately, these instructions are incomplete as of August 31, 2018. Hopefully we will have this working in the future!

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
```bash
$ mkdir <builddir>
$ cd <builddir>
```
4. Execute cmake command:
```bash
$ cmake <path_to_root_download/root-<version>>
```
> Example:
```bash
$ cmake /home/ahill/root-6.14.02
```
5. Build
```bash
$ cmake --build .
```
6. Run <br>
Each time you run ROOT, you must enter the following into terminal:
```bash
$ source <builddir>/bin/thisroot.sh
$ root
```
### Using PyROOT

If running Python in terminal, you can use the following commands to use PyROOT:
```bash
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
```bash
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
