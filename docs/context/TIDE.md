JupyterHub

TIDE offers a dedicated JupyterHub instance that offers simplified access to TIDE resources in a browser-based development environment. JupyterHub is the latest web-based interactive development environment for notebooks, code, and data. Its flexible interface allows users to configure and arrange workflows in data science, scientific computing, computational journalism, and machine learning. Languages such as Python, R, and Julia are available along with Pandas, PyTorch, and TensorFlow libraries.

JupyterHub Info
Batch Jobs

In addition to the JupyterHub instance, batch jobs can be created and run as software containers directly on the TIDE cluster. Batch jobs allow for longer running jobs as well as parallelization of jobs.

Compute Power

With 27 Dell PowerEdge servers, TIDE is equipped to run your AI and ML workloads. TIDE offers 17 GPU nodes each equipped with 4 NVIDIA L40 48 GB RAM GPUs for a total of 68 L40 GPUs. In addtion to the GPU nodes, TIDE has one GPU-Advanced node equipped with 4 NVIDIA A100 80 GB RAM GPUs.

For workloads that do not require access to GPUs, we have 6 CPU nodes with 64 CPUs each totalling 384 CPUs. The CPU nodes are also equipped with higher amounts of RAM to support CPU-only workloads.
**Note** Higher End GPUs are less available compared to lower end ones. Default System is 8 core CPU, 32 GB RAM and L40 GPU. 

TIDE has 3 storage nodes that are configured as a Ceph cluster with triple replication and offers a combined total of 240 TB usable storage.
Node Type 	Quantity 	Specifications for each Node
GPU 	17 	PowerEdge R760
(2x) Intel Xeon SIlver 4410Y 2G, 12C/24T
(4x) NVIDIA L40, 48 GB RAM
512 GB System RAM
GPU-Advanced 	1 	PowerEdge R750XA
(2x) Intel Xeon Gold 6338 2G CPU, 32C/64T
(4x) NVIDIA A100 GPU, 80 GB RAM
512 GB System RAM
CPU 	6 	PowerEdge R760
(2x) Intel Xeon Gold 6430 2.1G, 32C/64T
768 GB System RAM
Storage 	3 	PowerEdge R760
(3x) Intel Xeon Gold 6442Y 2.6G, 24C/48T
240 TB Storage
256 GB System RAM

General Container Images


The following general purpose containers are available for use.
Name 	Image 	Software 	JupyterHub Instance
Jupyter Stack Minimal Notebook 	minimal-notebook 	Ubuntu 22.04.4
JupyterLab 4.2.4
Python 3.11.9
Git 2.34.1
vi
nano
wget
unzip
tzdata 	CSU TIDE, SDSU

Jupyter Stack R Notebook 	r-notebook 	(Everything in jupyter/minimal-notebook)
The R interpreter and base environment
IRKernel to support R code in Jupyter notebooks
tidyverse packages from conda-forge
caret, crayon, devtools, forecast, hexbin, htmltools, htmlwidgets, nycflights13, randomforest, rcurl, rmarkdown, rodbc, rsqlite, shiny, tidymodels, unixodbc packages from conda-forge 	CSU TIDE, SDSU

Jupyter Stack SciPy Notebook 	scipy-notebook 	(Everything in jupyter/minimal-notebook)
altair, beautifulsoup4, bokeh, bottleneck, cloudpickle, conda-forge::blas=*=openblas, cython, dask, dill, h5py, matplotlib-base, numba, numexpr, openpyxl, pandas, patsy, protobuf, pytables, scikit-image, scikit-learn, scipy, seaborn, sqlalchemy, statsmodel, sympy, widgetsnbextension, xlrd packages
ipympl and ipywidgets for interactive visualizations and plots in Python notebooks
Facets for visualizing machine learning datasets 	CSU TIDE, SDSU

Jupyter Stack Tensorflow Notebook 	tensorflow-notebook 	(Everything in jupyter/scipy-notebook and its ancestor images)
tensorflow machine learning library Jupyter Server Proxy to support TensorBoard 	CSU TIDE, SDSU

Jupyter Stack PyTorch Notebook 	pytorch-notebook 	(Everything in jupyter/scipy-notebook and its ancestor images)
pytorch machine learning library (torch, torchaudio, torchvision) 	CSU TIDE, SDSU

Jupyter Stack Julia Notebook 	julia-notebook 	(Everything in jupyter/minimal-notebook)
The Julia compiler and base environment
IJulia to support Julia code in Jupyter notebook
Pluto.jl reactive Julia notebook interface, made accessible with jupyter-pluto-proxy
HDF5 package 	TIDE, SDSU

Jupyter Stack Datascience Notebook 	datascience-notebook 	(Everything in the jupyter/scipy-notebook, jupyter/r-notebook, and jupyter/julia-notebook images and their ancestor images)
rpy2 package 	CSU TIDE, SDSU

