# InnovationDiffusion

The code as a part of "Network localization strength regulates innovation diffusion with macro-level social influence".
Please see the link https://arxiv.org/abs/2301.00151 if you are interested in the paper. 
Here, we provide some codes that are used to perform the numerical simulation and plot the figures in the manuscript.

## Describtion 
The project considers the conformity effect in innovation diffusion and study how the network structure affect the dynamical behavior of the model, i.e. outbreak threshold, and tricritical point of determining the type of phase transition.
For more detailed information, please read the paper.

## Content 
The contents contained under each folder (figure1,...,sfigure4) in the directory are consistent with figure in the paper. The networks, codes and results required to plot the figure are listed in corresponding folder, respectively. All common used functions are integrated in coupling_diffusion.py, included in the utils.

## Install and Run

You can install or download the InnovationDiffusion to local.

* Clone the repository  
$ git clone https://github.com/LeyangXue/InnovationDiffusion.git

* Prerequisites:  
    * networkx  
    * numpy  
    * matplotlib  
    * seaborn 
    * multiprocessing 
    * pandas
    * pickle 
    * scipy
    * sklearn
    * random 
    * collections 
    * os
    * sys
    * math  

    **Note**: please ensure the dependences before runing the script

* Change the root_path variable in xxx.py file, set the value of root_path as your current local path 

    **Example**:  if you run the code, e.g. /figure1/code/plot.py, correct it as  
'F:/work/work4_dynamic' --->  your local path

## Citation

If you use this code or paper for your research, please cite the following:

    @misc{https://doi.org/10.48550/arxiv.2301.00151,
    doi = {10.48550/ARXIV.2301.00151},
    url = {https://arxiv.org/abs/2301.00151},
    author = {Xue, Leyang and Yang, Kai-Cheng and Cui, Peng-Bi and Di, Zengru},
    keywords = {Physics and Society (physics.soc-ph), FOS: Physical sciences, FOS: Physical sciences},
    title = {Network localization strength regulates innovation diffusion with macro-level social influence},
    publisher = {arXiv},
    year = {2023},
    copyright = {arXiv.org perpetual, non-exclusive license}
    }

## Email
Any suggestion are welcome and please send your suggestion to hsuehleyang@gmail.com


