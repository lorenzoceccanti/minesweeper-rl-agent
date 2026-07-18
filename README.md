# minesweeper-rl-agent
The aim of this project consist in training an agent to play Minesweeper game, in a customized MDP environment and evaluate the performances of various DRL algorithms.

## Usage

```bash
# train
python main.py train --alg=dqn --h=6 --w=6 --m=5 --n-episodes=500
python main.py train --alg=ppo --h=9 --w=9 --m=10 --n-episodes=500

# test (held-out evaluation of a saved checkpoint)
python main.py test --alg=ppo --ckpt=checkpoints/*-best.pt --h=6 --w=6 --m=5

# play interactively
python main.py game --h=9 --w=9 --m=10
```

`--config` overrides `config.yaml`; any `<algorithm>.train`/`<algorithm>.test` key also
has a matching `--key value` CLI flag (see `main.py`'s `ALIASES` for the short forms).

## Hyperparameter sweeps (`sweeps/`)

```bash
# validate a campaign YAML before registering it
python -m sweeps.cli validate --campaign sweeps/campaigns/<campaign>.yaml

# register the sweep(s) with W&B (creates the sweep_ids, run once)
python -m sweeps.cli register --campaign sweeps/campaigns/<campaign>.yaml

# run a worker against the registered sweeps (repeat on any machine)
python -m sweeps.cli worker --campaign sweeps/campaigns/<campaign>.yaml --profile <profile>

# promote each sweep's finalist(s): multi-seed confirmation + held-out test
python -m sweeps.cli promote --campaign sweeps/campaigns/<campaign>.yaml

# render the campaign's markdown report from promotion results
python -m sweeps.cli report --campaign sweeps/campaigns/<campaign>.yaml
```

For an unattended multi-day run, use `sweeps/run.sh` instead of calling
`worker` directly — it round-robins a few trials at a time across all of a campaign's
sweeps instead of letting one sweep monopolize the machine, and auto-restarts on failure:

```bash
./sweeps/run.sh sweeps/campaigns/<campaign>.yaml <profile>
```

# Contributors
- Cristiano Corsi
- Lorenzo Ceccanti
