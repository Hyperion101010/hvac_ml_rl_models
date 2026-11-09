# HVAC follow-up policy

This folder contains a policy that decides what the follow-up bot should do next with a customer
after an HVAC visit. The function `act(observation, history)` in `policy.py` returns one of the
eight allowed actions.

## How it decides

1. **One hard safety rule comes first.** If the customer is unhappy, or the service-risk score
   is above 0.75, hand off to a person (`ESCALATE_TO_HUMAN`).
2. **Otherwise a trained model picks the action.** A causal transformer reads the episode so far
   and estimates a Q value per action. Actions the old logged policy would almost never take
   (BC probability below 0.01) are excluded; among the rest, the highest Q wins.
3. **If the model can't run, a hand-written rule (`product_rule`) takes over.** This covers a
   missing model file, a problem loading PyTorch, or an inference error. The policy always
   returns a valid action.

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

Python 3.12.8. Install the packages (PyTorch and NumPy, with the versions used for training):

```bash
pip install -r requirements.txt
```

To check that the policy runs, from the folder that contains `starter/` and `data/`:

```bash
python starter/check_policy.py --policy submission/policy.py
```

## How the model was trained

The model is already included, so you do not need to retrain. To recreate it:

```bash
cd submission
python train.py --epochs 25 --action-conditioned-q --bc-threshold 0.01 --tau 0.05 --seed 7
```

- It trains on `../data/train_logs.jsonl` and checks itself on `../data/valid_logs.jsonl`, the
  files provided with the task. They are not copied into this folder.
- After each training round, the script estimates how well the current model would have done
  using part of the validation data, and keeps the best round. The included model is from round
  9 of 25. This estimate is noisy, which is discussed in the report.
- Other settings are the defaults in `train.py`.
- Retraining can give slightly different results on different computers, even with the same seed.
- Model file checksum (sha256): `c51f905181ed9f1072da88728495cf308f16c4a95758686605704a9d84164666`

## Results and limitations

The report (`report.md`) covers results, comparisons and limits. The main caveat is that we had
no simulator, only logs from an older system, so all performance figures are estimates, not
measured outcomes.

AI coding tools were used during development.