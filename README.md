# Generative downscaling PDE solver
This project provides codes for paper [Generative downscaling of PDE solvers with physics-guided diffusion models]. In this project, we 
# Usage of code
The usage of code is basically the same for each example, here we only present the details of implementing the 2D Nonlinear Poisson. 
## Nonlinear reaction diffusion equation
Consider 
```
-d1 \Delta u + d2 u^3 = a,
```
with zero BC.
### Data generation example
```
python gen_P_2d.py -ns 200 -nex 40 -nx 16 -m 8 -d1 -0.0005 -d2 1 -alp 1.6 -tau 7 -flg 1 -seed 9 
````
The above code generates 200 data tuples {a_k, u_k^c, u_k^f}. The coarse mesh grid size is 16 by 16 and the fine mesh grid size is 128 by 128 (8 times superresolution: 16 times 8=128). The diffusion parameter d1 is set to be -0.0005 and the reaction coefficient d2 is set to be 1. The source term a is sampled from Gaussian random field N(0, (-\Delta + \tau^2 I)^{-alpha}), where alpha and tau are set to be 1.6 and 7 respectively.
### Model training and prediction 
The configuration of the diffusion model and the FNO are defined in the config_2D.py 
#### Train DDPM
```
python main_2D.py
```
#### Train FNO
```
python FNO_2D.py
```
#### Compare all solvers (CSI, FNO, DDPM, DDIM and theirs fine tune)
```
python Solver_2D.py
```

