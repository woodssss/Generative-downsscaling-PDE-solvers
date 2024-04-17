# Generative downscaling PDE solver
This project provides codes for paper [Generative downscaling of PDE solvers with physics-guided diffusion models](https://arxiv.org/pdf/2404.05009.pdf). In this project, we propose physics-guided diffusion models (pgdm), a novel, data-driven surrogate approach for solving PDEs. Our approach initially train a diffusion model that upscales the low fidelity solution to high fidelity solution, which are subsequently refined by minimizing the physical discrepancies as defined by the discretized PDEs at the finer scale.
# Usage of code
This section outlines the standard procedure for utilizing the code, with a focus on the implementation of the 2D Nonlinear Poisson solver. The Jupyter notebook step by step demonstration is in the 'pgdm_demo.ipynb'.  
## Nonlinear reaction diffusion equation
Consider 
```
-d0 \Delta u + d1 u^3 = d2 a,
```
with zero BC.
### Data generation example
```
python gen_P_2d.py -ns 200 -nex 40 -nx 16 -m 8 -d0 -0.0005 -d1 1 -d2 1 -alp 1.6 -tau 7 -flg 1 -seed 9 
````
- The above code generates 200 data tuples {a_k, u_k^c, u_k^f}_{k=1}^{200}. 
  - The coarse mesh grid size is 16x16 and the fine mesh grid size is 128x128 (8 times superresolution).
  - The diffusion parameter d0 is set to be -0.0005 and the reaction coefficient d1 and the source coefficient is set to be 1.
  - The source term a is sampled from Gaussian random field N(0, (-\Delta + \tau^2 I)^{-alpha}), where alpha and tau are set to be 1.6 and 7 respectively.
### Model training and prediction 
The configuration (network architecture, training strategies, etc) of the diffusion model and the FNO are defined in the config_2D.py 
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

