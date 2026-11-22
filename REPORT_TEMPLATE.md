# Experiment Report Template

> Fill this in after running the project. Do not invent results or conclusions.

## Question

How did the three reward designs affect PPO's landing performance and behavior?

## Setup

- Environment:
- PPO timesteps per run:
- Training seeds:
- Evaluation episodes per seed:
- Python / Gymnasium / Stable-Baselines3 versions:
- Hardware:

## Reward designs

- Built-in reward:
- Hover bonus:
- Potential-difference shaping:

## Results

Copy the task-return mean and solve rate from the generated plot or calculate them from `artifacts/results/episodes.csv`.

| Reward design | Mean original task return | Solve rate (score ≥ 200) | Variation across training seeds | Notes from clips |
| --- | ---: | ---: | ---: | --- |
| Built-in reward | | | | |
| +1 per frame | | | | |
| Potential difference | | | | |

## What I observed

Describe what the agent did in the clips. Distinguish observed behavior from guesses about why it happened.

## Interpretation

Did the agent optimize the goal you intended? Compare its shaped training return with its original task return. Mention whether the pattern repeated across seeds.

## Limitations

Discuss training duration, number of seeds and evaluation episodes, environment version, and anything else that limits the conclusion.

## Next experiment

State one change you would test next, why you chose it, and what result would support or weaken your explanation.
