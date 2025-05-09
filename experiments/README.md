# Experiments

This folder contains example Jupyter notebooks demonstrating how to run and analyze experiments for the Poisson inverse and forward problems.

## Notebooks

- **PInv.ipynb**  
  End-to-end walkthrough of the Poisson inverse (PINv) workflow:
  1. Data generation (noisy / clean mix)  
  2. Pre-training a PINN model  
  3. Data partition and Selective neuron pruning  
  4. Fine-tuning the pruned model  
  5. Evaluation of u-field  

- **Poisson.ipynb**  
  End-to-end walkthrough of the forward Poisson data-assimilation problem:
  1. Data generation (noisy / clean mix) 
  2. Pre-training a PINN model  
  3. Data partition and Selective neuron pruning  
  4. Fine-tuning the pruned model  
  5. Evaluation of the reconstructed solution field

## Prerequisites

1. Python 3.8+ environment with all dependencies installed:  
```bash
   pip install -r ../requirements.txt
```

2.  quick start
```bash
cd experiments
jupyter lab PInv.ipynb
jupyter lab Poisson.ipynb
```
