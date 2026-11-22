# Lunar Lander: Reward Shaping

**Can changing the reward help a PPO agent land a spacecraft—or teach it the wrong goal?**

This project trains PPO agents to play Gymnasium's LunarLander-v3 and compares the built-in reward, a flawed +1-per-frame hover bonus, and potential-difference reward shaping. Agents are evaluated on the original game score so that reward hacking is visible.

**Author:** Ramazan · [GitHub profile](https://github.com/romagoat10)

I used AI assistance while drafting and debugging parts of the code. I ran the training and evaluation on my computer.

## Watch the agent

[Watch the five-episode baseline evaluation](baseline_seed_0.mp4)

## Results

Each design was trained for 300,000 steps with training seeds 0, 1, and 2. Each of the nine final checkpoints was evaluated on the same 20 held-out environment reset seeds. Values show the mean of three model-level averages and sample standard deviation (SD) across training seeds. Solve-rate variation is in percentage points (pp).

| Reward design | Original task return (mean ± SD) | Solved episodes (mean ± SD) |
| --- | ---: | ---: |
| Built-in reward | **50.9 ± 86.0** | **23.3% ± 36.2 pp** |
| +1 per frame (flawed hover bonus) | **−78.2 ± 66.7** | **0.0% ± 0.0 pp** |
| Potential difference | **24.0 ± 85.2** | **13.3% ± 18.9 pp** |

The built-in reward had the highest mean in this run. The hover bonus performed poorly on the original landing objective. Results varied widely by training seed, and three seeds are too few to establish a reliable ranking. Potential shaping did not consistently outperform the baseline in this experiment.

- [Episode-level evaluation data](results/multiseed_evaluation.csv)
- [Aggregated results](results/summary.csv)
- [Reward comparison plot](results/reward_comparison.png)

## Experiment design

All agents use PPO (Proximal Policy Optimization) with the same small multilayer perceptron policy in LunarLander-v3.

1. **Built-in reward:** Gymnasium's original landing score.
2. **Hover bonus:** adds +1 each frame. This deliberately flawed rule can reward staying in the air without landing safely.
3. **Potential difference:** adds gamma * Phi(next_state) - Phi(state), using a potential based on distance, speed, tilt, and spin. Terminal potential is zero.

The evaluator records original task return separately from shaped training return. This helps show when an agent improves its training score without improving the intended landing task.

## Setup on Windows

Python 3.13 was used for this setup. From the repository folder, open PowerShell:

~~~powershell
py -3.13 -m venv .venv
..venvScriptsActivate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### Check the environment

~~~powershell
python ./src/random_rollout.py
~~~

The random controller is expected to crash; this checks that the Box2D game opens.

### Train a quick baseline

~~~powershell
python ./src/train.py --experiment baseline --timesteps 20000 --seeds 0 --eval-freq 10000 --eval-episodes 2
~~~

This checks setup only; it is not enough to draw conclusions about learning performance.

### Train the full comparison

~~~powershell
python ./src/train.py --experiment all --timesteps 300000 --seeds 0 1 2
~~~

Models, configurations, and logs are saved under artifacts/ and excluded from Git by default.

### Evaluate and reproduce the held-out comparison

~~~powershell
python ./src/evaluate.py --experiment all --seeds 0 1 2 --episodes 20 --eval-seed 100000 --checkpoint final
Copy-Item ./artifacts/results/episodes.csv ./results/multiseed_evaluation.csv
python ./src/summarize_results.py
python ./src/plot_results.py --results results/multiseed_evaluation.csv --output results/reward_comparison.png
~~~

The final checkpoint is used for each model to keep checkpoint selection consistent. Evaluation writes fresh metrics to artifacts/results/episodes.csv; the copy command saves a tracked snapshot in results/. Add --record-video to save MP4 videos (excluded from Git by default).

### Plot results

~~~powershell
python ./src/plot_results.py
~~~

## Project structure

~~~text
src/
  rocket_lander.py       Reward wrappers and environment setup
  random_rollout.py      Random-policy environment walkthrough
  train.py               PPO training
  evaluate.py            Held-out evaluation and video recording
  summarize_results.py   Aggregate metrics across training seeds
  plot_results.py        Plot task score and solve rate
requirements.txt         Python dependencies
REPORT_TEMPLATE.md       Experiment report template
baseline_seed_0.mp4      Five baseline evaluation episodes
results/                 Tracked evaluation data, summary, and plot
~~~

## Limitations

- Three training seeds provide an uncertain estimate of variation.
- All agents share 20 held-out environment seeds; more test seeds would better assess generalization.
- Longer training may change the results.
- The video illustrates one model and does not prove reliable performance.

This is a small controlled experiment, not proof that one reward design is universally best. A useful next step is to train more seeds and evaluate on a larger, separate set of environment seeds.

## References

- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
