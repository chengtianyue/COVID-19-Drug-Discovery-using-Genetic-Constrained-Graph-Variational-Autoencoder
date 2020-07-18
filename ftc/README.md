# 2019-nCov

## Description
Efforts towards proposing a potentially highly active molecule against a target protein of the 2019 Novel Coronavirus. 

Submission for the Sage-health coronavirus deep learning competition. 

Structure of this repository:

The Jupyter Notebook: Report_Tom_MacDougall.ipynb serves as my submission, including all phases of dataset acquiring, prediction and molecular generation, docking of the final candidates and , including an autodock vina tutorial.

The requirements for this project are unfortunately pretty large. And also unfortunately, the used approaches make use of both Pytorch and Tensorflow so they have to each be installed if you want to work through the report notebook from start to finish. The notebook however does make a lot of use of pickling and saving intermeadiate answers, so you may even be able to do a kernel switch part way through. 

I have made an install.sh script that can install all of the dependencies in a fresh conda environment (called nCov_env). So, for the install script to work, you must have conda or miniconda installed. If you already have a environment with most or all of the requirements, go ahead and try the various parts. It might be however advisable to stick to tensorflow 1.XX however as I don't think backwards compatibility is guaranteed between major releases.

## Contents of this repo:
The report notebook is the main bread and butter of this repository. Unfortunately, it is poorly formated on github but might look a lot better cloned and then viewed locally! The photos that aren't embedding properly are found in the Docking/ directory, if you want to see them

Submission overview video (covers the main report notebook)
https://youtu.be/hNdh8EzalCU

A video I made about various different kinds of activity, and a bit on Enzyme Kinetics is found here: https://youtu.be/AbizsxxWcDk

Data is stored in /Data (for both dataset generation and the two methods)

Intermeadiate notebooks and rough work are stored in /notebooks

Input and output for docking studies is found in /Docking. There are a lot of files here since every docked
compound needs about 3 or 4 files.

Files for the predictive deep learning method, The Edge Memory Neural Network, are found in /EMNN, reference is mentioned in the report

Files for the generative deep learning method, The Constrained Graph Variational Autoencoder, are found in /CGVAE, reference is mentioned in the report

Various papers mentioned and referenced (including for the methods used) are found in /Literature

Various Pymol session are found in /PymolSessions

A nice Vina tutorial by the original author is available in vina-tutorial/

Other files, notes and rough work can be found in /misc, including a notes.txt file that was like my scratchpad for ideas for the duration of the competition. These are not strictly important to the narrative of the study and their contents have likely been added to the report notebook anyway.
