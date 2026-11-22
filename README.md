# Lunar Lander: Reward Shaping

**Can changing the reward help a PPO agent land—or teach it the wrong goal?**

This project trains PPO agents on Gymnasium's LunarLander-v3 and compares three reward designs: the built-in reward, a deliberately flawed +1-per-frame hover bonus, and potential-difference shaping. Every policy is evaluated using the original game score, separately from its shaped training reward.

**Author:** Ramazan · [GitHub profile](https://github.com/romagoat10)

I used AI assistance while drafting and debugging parts of the code. I ran the training and evaluation on my computer.

## Watch the agent

[Watch five baseline evaluation flights](baseline_seed_0.mp4)

## Results

Each design was trained for 300,000 steps with training seeds 0, 1, and 2. Each of the nine final checkpoints was evaluated on the same 20 held-out environment reset seeds. The table reports the average of the three model-level means; ± is sample standard deviation across training seeds. Success-rate variation is in percentage points (pp).

| Reward design | Original task return (mean ± SD) | Solved episodes (mean ± SD) |
| --- | ---: | ---: |
| Built-in reward | **50.9 ± 86.0** | **23.3% ± 36.2 pp** |
| +1 per frame (flawed hover bonus) | **−78.2 ± 66.7** | **0.0% ± 0.0 pp** |
| Potential difference | **24.0 ± 85.2** | **13.3% ± 18.9 pp** |

The built-in reward had the highest mean in this run, while the hover bonus performed poorly on the original landing objective. Results varied widely between training seeds; with only three seeds, they do not establish a reliable general ranking. Potential-based shaping did not consistently outperform the baseline here.

- [All 180 episode results](results/multiseed_evaluation.csv)
- [Aggregated summary](results/summary.csv)
- [Comparison plot](results/reward_comparison.png)

## The experiment

All agents use PPO with the same small multilayer perceptron policy in LunarLander-v3.

1. **Built-in reward:** Gymnasium's original landing score.
2. **Hover bonus:** adds +1 each frame, even when the lander is not moving closer to a safe landing.
3. **Potential-difference shaping:** adds gamma * Phi(next_state) - Phi(state), with a potential based on distance, speed, tilt, and spin. Terminal potential is zero.

Using final checkpoints keeps training duration equal and avoids selecting checkpoints according to each design's differently shaped reward.

## Set up on Windows

Python 3.13 was used. From PowerShell in the repository folder:

~~~powershell
py -3.13 -m venv .venv
..venvScriptsActivate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

## Run the project

A random flight checks that the Box2D environment opens (a random controller is expected to crash):

~~~powershell
python .src
andom_rollout.py
~~~

Quick training check:

~~~powershell
python .src	rain.py --experiment baseline --timesteps 20000 --seeds 0 --eval-freq 10000 --eval-episodes 2
~~~

Full training run:

~~~powershell
python .src	rain.py --experiment all --timesteps 300000 --seeds 0 1 2
~~~

Held-out comparison and reproducible summary:

~~~powershell
python .srcevaluate.py --experiment all --seeds 0 1 2 --episodes 20 --eval-seed 100000 --checkpoint final
Copy-Item .artifacts
esultsepisodes.csv .
esultsmultiseed_evaluation.csv
python .srcsummarize_results.py
python .srcplot_results.py --results resultsmultiseed_evaluation.csv --output results
eward_comparison.png
~~~

The evaluator writes episode rows to artifacts/results/episodes.csv. Copying the file preserves a tracked snapshot in results/. Add --record-video to record flights. Models and run logs under artifacts/ are excluded from Git by default.

## Project files

~~~text
src/
  rocket_lander.py      Reward wrappers and environment setup
  random_rollout.py     Random-policy environment walkthrough
  train.py              PPO training
  evaluate.py           Held-out evaluation and video recording
  summarize_results.py  Aggregate results and variation by training seed
  plot_results.py       Return and solve-rate plots
requirements.txt        Python dependencies
REPORT_TEMPLATE.md      Experiment report template
baseline_seed_0.mp4     Five baseline flights
results/
  multiseed_evaluation.csv  Episode results for all nine models
  summary.csv               Aggregate results
  reward_comparison.png     Comparison plot
~~~

## Limitations

- Only three training seeds were used, so variation estimates are uncertain.
- All policies were tested on the same 20 environment seeds; a larger independent test set would help assess generalization.
- Each model was trained for 300,000 steps; longer training may change the outcome.
- The video shows one model and does not prove reliable landings.

These results describe this experiment only. More training seeds and test environments are needed before making a strong claim about which reward design works best.

## References

- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
