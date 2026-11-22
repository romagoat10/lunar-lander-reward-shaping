# Lunar Lander: Reward Shaping

**Can changing the reward help a PPO agent land a spacecraft—or teach it the wrong goal?**

This project trains agents to play Gymnasium's LunarLander-v3 and explores how reward design affects both training reward and actual landing performance. It compares the environment's original reward with two custom reward rules.

**Author:** Ramazan · [GitHub profile](https://github.com/romagoat10)

I used AI assistance while drafting and debugging parts of the code. I ran the training and evaluation on my computer. The result below is an initial baseline measurement; the held-out comparison across all reward designs is still in progress.

## Watch the agent

[Watch five baseline evaluation flights](baseline_seed_0.mp4)

## Preliminary result

| Reward design | Training seed | Evaluation episodes | Mean original task score | Solved episodes |
| --- | ---: | ---: | ---: | ---: |
| Built-in reward baseline | 0 | 5 | **202.7** | **3 / 5 (60%)** |

Gymnasium counts an episode scoring 200 or more as solved. This is a small sample from one training seed, so it does not establish that one reward design is better than another. A fair comparison needs the same held-out episode count across all reward designs and multiple training seeds.

## The experiment

All agents use PPO (Proximal Policy Optimization) with the same small neural-network policy in LunarLander-v3.

1. **Built-in reward:** Gymnasium's original landing score.
2. **Hover bonus:** adds +1 for every environment step. This intentionally flawed rule may reward staying aloft even when the lander is not moving toward a safe landing.
3. **Potential-difference shaping:** adds gamma * Phi(next_state) - Phi(state), where the potential reflects distance, speed, tilt, and spin. The terminal potential is set to zero.

The evaluator reports the original task return separately from the reward used during training. This helps identify reward hacking: an agent may earn more of its shaped reward without doing better at the actual task.

## Set up on Windows

Use Python 3.13. From the repository folder, open PowerShell and run:

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

## Run the project

### Try a random flight

~~~powershell
python .\src\random_rollout.py
~~~

A random controller is expected to crash; this checks that the Box2D environment opens.

### Train a quick baseline

~~~powershell
python .\src\train.py --experiment baseline --timesteps 20000 --seeds 0 --eval-freq 10000 --eval-episodes 2
~~~

This short run checks the training setup. It is not enough to draw a strong conclusion about learning performance.

### Train all reward designs

~~~powershell
python .\src\train.py --experiment all --timesteps 300000 --seeds 0 1 2
~~~

Each reward design starts from a fresh PPO policy for each training seed. Models and training logs are written under artifacts/ and are excluded from Git by default.

### Evaluate and record flights

~~~powershell
python .\src\evaluate.py --experiment baseline --seeds 0 --episodes 5 --record-video
~~~

This evaluates five episodes and saves one MP4 containing those flights. Episode metrics are written to artifacts/results/episodes.csv. By default, a new evaluation replaces that CSV; add --append to keep existing rows.

For a held-out comparison across all designs and seeds:

~~~powershell
python .\src\evaluate.py --experiment all --seeds 0 1 2 --episodes 10 --record-video
~~~

### Plot the results

~~~powershell
python .\src\plot_results.py
~~~

This creates a plot of original task return and solve rate under artifacts/plots/.

## Project structure

~~~text
src/
  rocket_lander.py   Reward wrappers and environment setup
  random_rollout.py  Random-policy walkthrough
  train.py           PPO training for the three reward designs
  evaluate.py        Held-out evaluation, metrics, and video recording
  plot_results.py    Task-score and solve-rate plots
requirements.txt     Python dependencies
REPORT_TEMPLATE.md  Template for methods, results, and limitations
baseline_seed_0.mp4 Five baseline evaluation flights
~~~

## Limitations and next steps

- The reported result covers five evaluation episodes from one baseline training seed.
- Held-out results for the hover-bonus and potential-shaped agents are not reported yet.
- PPO results vary by seed and training time; more runs are needed before comparing reward designs.
- The video shows one model's behavior and does not prove that it lands reliably.

The next step is to evaluate all three reward designs across the same held-out episode count and compare original task return, solve rate, and flight behavior.

## References

- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
