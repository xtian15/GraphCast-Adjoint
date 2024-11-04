# GraphCast Adjoint Model

This repository provides the tangent linear and adjoint models developed for the GraphCast weather model, enabling users to calculate sensitivities and linear perturbation responses. The code, located in gc_subs.py, builds on DeepMind’s original GraphCast implementation and extends it with tangent linear (TL) and adjoint (AD) functionalities.

Requirements

To run this script, you need to set up an environment with the original GraphCast repository, the necessary datasets, and dependencies.

Prerequisites

	1.	GraphCast Repository
Clone the original GraphCast repository by DeepMind. This contains the core GraphCast model:

git clone https://github.com/google-deepmind/graphcast


	2.	Background Data
Download the background dataset, including the model weights, normalization statistics, and example inputs are available on [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast)
Save this dataset in a location that your script can access.

	3.	Python Environment
Set up a Python environment with the required dependencies. The primary requirements are those specified in the original GraphCast repository, following the steps implemented at `graphcast_demo.ipynb` in [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb).


Repository Setup

	1.	Clone This Repository
Clone this repository containing the tangent linear and adjoint model code:

git clone https://github.com/xtian15/GraphCast-Adjoint.git


	2.	Set Up Paths
Ensure that paths in gc_subs.py correctly point to:
	•	The location of the original GraphCast code.
	•	The downloaded background dataset.

Running the Script

To run the tangent linear and adjoint models for GraphCast, execute the following command from within this repository:

python gc_subs.py

Ensure that the paths to the dataset and the GraphCast code are correctly set within the script or your environment. Some examples of running as well as checking the tangent linear and adjoint of GraphCast are already included in the gc_subs.py script as in `check_tlm` and `check_adj` functions.

License

Please refer to the licenses of the original GraphCast repository and this repository for usage terms and conditions.

This README provides the necessary information for your colleagues to set up the environment and run gc_subs.py. Adjust any details specific to your dependencies or path settings as needed.
