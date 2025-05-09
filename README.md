

```markdown
# My PINN Prune–FineTune

A unified codebase for Physics-Informed Neural Networks (PINNs) enhanced with selective neuron pruning and fine-tuning, supporting both data-assimilation (forward) and inverse PDE problems, plus a library of pruning strategies.

## Repository Layout


my-pinn-prune-finetune/
├── configs/              # YAML config files for all problems
│   ├── Heat.yaml
│   ├── Poisson.yaml
│   ├── Stokes.yaml
│   ├── Wave.yaml
│   ├── EInv.yaml
│   ├── HInv.yaml
│   ├── NSInv.yaml
│   ├── PINv.yaml
│   └── WInv.yaml
├── common/               # shared utilities and PINN definition
│   ├── **init**.py
│   ├── pinn_model.py     # generic PINN MLP + learnable PDE params
│   ├── partition_data.py # data partitioning
│   ├── pruning.py        # iterative selective-prune routines
│   └── evaluate.py       # unified evaluation metrics
├── assimilation/         # forward (data-assimilation) problems
│   ├── heat/
│   ├── poisson/
│   ├── stokes/
│   └── wave/
├── inverse/              # inverse PDE problems
│   ├── beam/             # EInv
│   ├── heat_inv/         # HInv
│   ├── navier_stokes_inv/# NSInv
│   ├── poisson_inv/      # PINv
│   └── wave_inv/         # WInv
├── scripts/              # alternative, single-step & criterion-based pruning
│   ├── single_step.py
│   └── pruning_criteria.py
├── experiments/          # example Jupyter notebooks
│   ├── PInv.ipynb
│   └── Poisson.ipynb
├── README.md             # this file
└── requirements.txt      # Python dependencies



## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
````

2. **Run a data-assimilation (forward) problem**

   ```bash
   # Heat equation
   cd assimilation/heat
   python -m assimilation.heat.train --config  configs/Heat.yaml
   python -m assimilation.heat.finetune --config  configs/Heat.yaml
   ```

3. **Run an inverse problem**

   ```bash
   # Poisson inverse
   cd inverse/poisson_inv
   python -m inverse.PInv.train --config configs/PINv.yaml
   python -m inverse.PInv.finetune --config configs/PINv.yaml
   ```

4. **Select a pruning strategy**

   In your `configs/*.yaml`, set under `pruning`:

   ```yaml
   pruning:
     strategy: rms          # one of: iterative, single_step, rms, freq, std
     layers: [0,2,4,6]
     alpha: 0.05
     num_iter: 10            # only for multi-iter methods
   ```

   The `finetune.py` scripts will dispatch to:

   * `common/pruning.py` → `iterative` + `bias`
   * `scripts/single_step.py` → `single_step` + `bias`
   * `scripts/pruning_criteria.py` → `iterative`+ `abs`, `rms`, `freq`, `std`

5. **Launch experiment notebooks**

   ```bash
   cd experiments
   jupyter lab PInv.ipynb
   jupyter lab Poisson.ipynb
   ```

## Pruning Strategies

* **iterative** (default)
  Multi-step heuristic in `common/pruning.py`.

* **single\_step**
  One-shot prune based on mean-difference scores.

* **rms**
  RMS-based score:  score = RMS(forget) / (RMS(retain)+ε).

* **freq**
  Frequency-based score: proportion of positive activations.

* **std**
  Standard deviation score: std(forget) / (std(retain)+ε).

## Contributing

* **Add a new PDE**:

  1. Create a `configs/*.yaml`
  2. Add code under `assimilation/` or `inverse/`
  3. (Optional) add analysis notebook under `experiments/`

* **Pruning research**:
  Drop new pruning routines into `scripts/` and add a `strategy` name in your configs.

We welcome all improvements, issues and pull-requests!
```
