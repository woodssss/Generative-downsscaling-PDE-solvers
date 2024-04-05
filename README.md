# Generative downscaling PDE solver
This project provides codes for paper [Generative downscaling of PDE solvers with physics-guided diffusion models]. In this project, we 
# Usage of code
The usage of code is basically the same for each example, here we only present the details of implementing the 2D Nonlinear Poisson. 
## Nonlinear reaction diffusion equation
Consider 
```
f_t = d f_xx + k f^2,
```
with zero BC
```
f(t,0)=f(t,1)=0.
```
### Data generation
```
python nrd_sample_1d_evo.py k d l
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
