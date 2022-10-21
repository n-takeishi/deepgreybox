# Codes for "Deep Grey-Box Models With Adaptive Data-Driven Models Toward Trustworthy Estimation of Theory-Driven Models"

## Toy dataset experiment (in Section 3.4 of the paper)

First, run the data-making scripts, the training script, and the prediction script by:

```
bash expt.sh toy1 demo adaptive 0 1.00e-02 cpu
```

The fourth argument of `expt.sh` is the random seed value, and the fifth argument is the value of $\lambda$.

Then, use `toy1/notebooks/inspect_model.ipynb` to inspect the results.

## Controlled pendulum

```
bash expt.sh pendulum demo adaptive 0 1.00e-02 cpu
```

Then, use `pendulum/notebooks/inspect_model.ipynb`.

## Reaction-diffusion system

```
bash expt.sh reaction-diffusion demo adaptive 0 1.00e-06 cuda:0
```

Then, use `reaction-diffusion/notebooks/inspect_model.ipynb`.

## Predator-prey system

```
bash expt.sh predator-prey demo adaptive 0 1.00e-03 cpu
```

Then, use `predator-prey/notebooks/inspect_model.ipynb`.
