# Final Results Plot

We use saved results by `.npz` type (Our algorithm, and CRM) and `.json` type (QRM). Some experiments include saved files as `RLAlgorithm.zip`, and saved results (success rate, mean reward, episode length, partial success) as `.npz` files.

- **normal**: This folder includes experimental results for a deterministic environment without noise, and infeasible conditions.
  
- **noise01**: This folder contains experimental results for a noisy environment without infeasible conditions.
  
- **missing**: This folder contains experimental results for an infeasible environment without noise.

- **normal_csv**: Contains `.csv` files to plot results in environments without noise and infeasible conditions.

- **noise_csv**: Contains `.csv` files to plot results in noisy environments without infeasible conditions.

- **missing_csv**: Contains `.csv` files to plot results in infeasible environments without noise.

### Plotting Results:

1. Convert `.npz` and `.json` file results into `.csv` file results for each environment (deterministic, noisy, missing).
  
2. Use `draw_with_csv.py` to plot results for each environment.
  
3. **Optional**: To plot results on the HalfCheetah environment for the A2C and PPO algorithm, use `draw_cheetah.py`.

## compute_mean_std.py

This file aims to compute the mean reward and standard deviation saved by `npz` files for the logic-based reward shaping algorithm and CRM, and the `json` file for QRM. Computed mean rewards and standard deviations will be saved as a `csv` file.

Arguments:
- First argument relates to the `missing` condition, which is boolean.
- Second argument relates to the `noise` condition and is a float. We've tested with a 0.1 (10%) chance of noisy control.

Example:
```bash
python compute_mean_std.py False 0
```

The example above converts saved results (either `.npz` or `.json` type) in a deterministic environment with a 0% chance of noisy control for various algorithms across different environments.
```bash
python compute_mean_std.py False 0
```
This example converts saved results of the A2C algorithm on the HalfCheetah domain.

## draw_with_csv.py
This will read the converted `csv` file by `compute_mean_std.py`, then plot and save the results.

Arguments:
- First argument relates to type of plots (normal, noise, missing)

To see the result on deterministic environments without noise and infeaisble condition, 
Example:
```bash
python draw_with_csv.py normal
```

To see the result on environments with noise and without infeaisble condition, 
Example:
```bash
python draw_with_csv.py noise
```

To see the result on environments without noise and with infeaisble condition, 
Example:
```bash
python draw_with_csv.py missing
```

All plots will be saved `./saved_plots` folder.

## draw_cheetah.py
This is exclusively for the DDPG, A2C, PPO algorithm's results on the HalfCheetah Environment.

Once you have converted all saved results (.npz files) into CSV format with `compute_mea_std.py`,
use
```bash
python draw_cheetah.py
```

All plots will be saved `./saved_plots` folder.
## plotter.py
This includes functions to aggregate all different types of results (QRM result is saved with `.json`, and the others are saved with `.npz`).
These functions will not be used directly. Instead, these functions will be used when we convert .npz and .json format to CSV format with `compute_mea_std.py`.