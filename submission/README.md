# HVAC follow-up policy

This folder contains a policy that decides what the follow-up bot should do next with a customer
after an HVAC visit. The function `act(observation, history)` in `policy.py` returns one of the
eight allowed actions.

## How it decides

1. **One fixed safety rule comes first.** If the customer is unhappy, or the service-risk score
   is above 0.75, the case is handed to a person (`ESCALATE_TO_HUMAN`).
2. **Otherwise a trained model picks the action.** The model reads the conversation so far and
   estimates how well each action is likely to turn out. It only considers actions the old system
   would plausibly have taken in the same situation, since the logs tell us nothing reliable about
   the rest, and it picks the best of those.
3. **If the model can't run, a simple hand-written rule (`product_rule`) takes over.** This covers
   a missing model file, a problem loading PyTorch, or an error while the model is running. A
   warning is printed once when this happens. The policy always returns a valid action.

Nothing needs a network connection or a paid service.

## What is in this folder

| File | What it is |
| --- | --- |
| `policy.py` | The complete policy |
| `models/transformer_fqi_acq.pt` | The trained model |
| `train.py` | The script that trains the model |
| `requirements.txt` | The Python packages needed |
| `report.md` | The report answering the seven questions |

## Setup and check

Python 3.12.8. Install the packages (PyTorch and NumPy, at the versions used for training):

```bash
pip install -r requirements.txt
```

To check that the policy runs, from the folder that contains `starter/` and `data/`:

```bash
python starter/check_policy.py --policy submission/policy.py
```

In `policy.py`, the variable `MODEL_LOADED` tells you whether the trained model is active or the
backup rule is being used.

## How the model was trained

The model is already included, so you do not need to retrain. To recreate it, run this from inside
this folder:

```bash
cd submission
python train.py --epochs 25 --action-conditioned-q --bc-threshold 0.01 --tau 0.05 --seed 7
```

- It trains on `../data/train_logs.jsonl` and checks itself on `../data/valid_logs.jsonl`, the files
  provided with the task. They are not copied into this folder.
- After each training round, the script estimates how well the current model would have done, using
  the first 500 validation conversations, and keeps the best round. The included model is from round
  9 of 25. This estimate is noisy, which the report discusses.
- All other settings are the defaults in `train.py`.
- By default the script saves to `models/transformer_fqi_acq.pt`, which overwrites the included
  model. To keep the included file, add `--out-dir ../retrained`.
- Retraining can give slightly different results on different computers, even with the same seed.
- Checksum of the included model (sha256):
  `c51f905181ed9f1072da88728495cf308f16c4a95758686605704a9d84164666`

## Results and limitations

The report (`report.md`) covers the results, the comparisons and the limits. The main caveat is
that there was no simulator, only logs from an older system, so all performance figures are
estimates from those logs, not measured outcomes.

AI coding tools were used during development.