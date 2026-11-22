# Lunar Lander: Reward Shaping

**Can changing the reward help a PPO agent land—or teach it the wrong goal?**

This project trains PPO agents on Gymnasium's LunarLander-v3 and compares the built-in reward with two shaped rewards. Evaluation uses the original game score, so a higher training reward cannot be confused with better landings.

**Author:** Ramazan · [GitHub profile](https://github.com/romagoat10)

I used AI assistance while drafting and debugging parts of the code. I ran the training and evaluation on my computer.

## Watch the agent

[Watch five baseline evaluation episodes](baseline_seed_0.mp4)

## Results

Each model was trained for 300,000 steps. Six training seeds (0–5) were used for each reward design. Each of the 18 final models was evaluated on the same 20 held-out environment reset seeds. The table reports the mean of the six model-level averages and sample standard deviation (SD) across training seeds. Solve-rate SD is shown in percentage points (pp).

| Reward design | Original task return (mean ± SD) | Solved episodes (mean ± SD) |
| --- | ---: | ---: |
| Built-in reward | **45.4 ± 94.5** | **20.0% ± 29.4 pp** |
| +1 per frame (flawed hover bonus) | **−68.1 ± 49.2** | **0.0% ± 0.0 pp** |
| Potential difference | **14.4 ± 85.9** | **10.0% ± 13.0 pp** |

The built-in reward had the highest mean in this run. The hover bonus performed poorly on the original landing objective. Scores still varied widely between training seeds, so these results do not establish a reliable general ranking. Potential shaping did not consistently outperform the baseline in this experiment.

- [Episode-level evaluation data](results/multiseed_evaluation.csv)
- [Aggregated results](results/summary.csv)
- [Reward comparison plot](results/reward_comparison.png)

## Experiment design

All agents use PPO with the same MLP policy in LunarLander-v3.

1. **Built-in reward:** the game's original landing score.
2. **Hover bonus:** adds +1 each frame; this flawed rule can reward staying aloft without landing safely.
3. **Potential difference:** adds gamma × Phi(next_state) − Phi(state), based on distance, speed, tilt, and spin. Terminal potential is zero.

The evaluator records the original task return separately from the shaped training return to help identify reward hacking.

## Run on Windows

Python 3.13 was used. From PowerShell in the repository folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Quick setup check

```powershell
python .\src\random_rollout.py
```

A random controller is expected to crash; this checks that the Box2D environment opens.

### Train and reproduce the comparison

```powershell
python .\src\train.py --experiment all --timesteps 300000 --seeds 0 1 2 3 4 5
python .\src\evaluate.py --experiment all --seeds 0 1 2 3 4 5 --episodes 20 --eval-seed 100000 --checkpoint final
Copy-Item .\artifacts\results\episodes.csv .\results\multiseed_evaluation.csv
python .\src\summarize_results.py
python .\src\plot_results.py --results results\multiseed_evaluation.csv --output results\reward_comparison.png
```

Models and logs are saved under artifacts/ and excluded from Git by default. Add --record-video to evaluation to save an MP4 per model.

## Project files

```text
src/                  Training, evaluation, environment, and analysis scripts
results/              Tracked evaluation data, summary, and plot
requirements.txt      Python dependencies
REPORT_TEMPLATE.md    Experiment report template
baseline_seed_0.mp4   Five baseline evaluation episodes
```

## Limitations

- Six training seeds provide a better view of run-to-run variation, but the estimate is still uncertain.
- All models share the same 20 held-out environment seeds; more test seeds would better assess generalization.
- Longer training may change the results.
- The video illustrates one model and does not prove reliable performance.

This is a small controlled experiment, not proof that one reward design is universally best. A useful next step is to train more seeds and evaluate on a separate, larger set of environment seeds.

## References

- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
