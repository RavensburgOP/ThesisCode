# ThesisCode

## Creating the dataset
To create a dataset for the MPI implementation a copy of the Homstrad database is needed. This can be acquired [here](ftp://mizuguchilab.org/homstrad/). Download the newest set beginning with "homstrad_with_PDB" and set the path to the root of the unpacked directory in the file InitialParseOfDataset.py.

A copy of FastTree must also be available. FastTree can be acquired from [here](http://meta.microbesonline.org/fasttree/), where compilation information is also available. The path to the FastTree executable must also be set in InitialParseOfDataset.py.

Then run the files in the following order:
1. InitialParseOfDataset.py
2. DatasetForThesis.py
3. SampleDatasetForMPI.py

This should create the full dataset used in the MPI implementation

## NLopt
For a working version of NLopt, where maximum and minimum number of interpolations is not considered a fatal error go to the NLopt folder and follow the instructions available [here](http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation).

## MPI implementation
To run the MPI implementation go to the MPI_model folder and type:

1. cmake .
2. make

This should compile the program. Compilation has only been tested on OS X and CentOS 7, so please send an email to christan dot ravnen at gmail dot com or raise an issue here at GirHub, if you experience any problems getting compilation to work.

It is then possible to run a grid search with:

mpirun -n \<number of processes\> ./parameter_search \<start of grid\> \<end of grid\> \<step size\>

All variables must be provided for the program to work.
