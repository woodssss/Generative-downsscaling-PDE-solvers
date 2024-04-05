# Generative downscaling PDE solver
This project provides codes for paper [Generative downscaling of PDE solvers with physics-guided diffusion models]. In this project, we 
# Usage of code
The usage of code is basically the same for each example, here we only present the details of implementing the 2D Nonlinear Poisson. 
## Nonlinear reaction diffusion equation
Consider 
```
-0.0005 \Delta u + u^3 = a,
```
with zero BC.
### Data generation
```
python gen_P_2do.py -ns 200 -nex 40 -nx 16 -m 8 -d0 -0.0003 -d1 1 -d2 1 -alp 1 -tau 7 -flg 1 -seed 9 
````
k,d are the parameters within nrd equation and l is the parameter within covariance kernel. This code generates 10000 training functions and 30 test functions. One may run it on colab as well
```
%run nrd_sample_1d_evo.py k d l
````
### Model training and prediction 
#### Discretized time setting
```
python nrd_1d_dt_deeponet.py N Nte p q st d k l ads
```
