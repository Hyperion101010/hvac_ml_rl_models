# HVAC follow-up policy: submission

This folder contains a next-action policy for post-visit HVAC follow-ups.
`policy.py` exposes `act(observation, history)` and returns one of the eight allowed actions.

## What the policy does

1. Hard safety rule: if `customer_signal` is `unhappy` or `service_risk_score` is above 0.75,
   return `ESCALATE_TO_HUMAN`.
2. Otherwise a learned policy chooses the action. A causal transformer reads the episode so far
   and predicts, for each action, the expected discounted return (Q value). A second head
   predicts how likely the old logged policy was to take each action; actions below a
   probability threshold (0.01, stored in the checkpoint) are excluded so the policy does not
   rely on values the logs cannot support. The action with the highest Q among the remaining
   ones is returned.
3. If the model cannot be used for any reason (missing file, torch import failure, shape error),
   `policy.py` falls back to a hand-written rule policy (`product_rule`) and emits one warning.
   `MODEL_LOADED` in `policy.py` tells you which path is active.

No network access or paid API is needed at run time.

## Files

| File | Purpose |
| --- | --- |
| `policy.py` | Self-contained policy: features, model class, safety rule, fallback rule, `act()` |
| `models/transformer_fqi_acq.pt` | Trained checkpoint used by `policy.py` |
| `train.py` | Training script that produces the checkpoint |
| `requirements.txt` | Pinned Python dependencies |
| `report.md` | Report answering the seven required questions |

## Setup

- Python: 3.12.8
- Install: `pip install -r requirements.txt` (torch and numpy only; versions are pinned to the
  ones used to train and test the checkpoint)

## Check the policy

From the repository root (the folder that contains `starter/` and `data/`):

```bash
python starter/check_policy.py --policy submission/policy.py
```

## How the model file was produced

The checkpoint was produced by `train.py` with these settings:

```bash
cd submission
python train.py --epochs 25 --action-conditioned-q --bc-threshold 0.01 --tau 0.05 --seed 7
```

- Data: `../data/train_logs.jsonl` for training, `../data/valid_logs.jsonl` for validation
  (both are the files provided with the task; they are not copied into this folder).
- Method: fitted Q-iteration with a causal transformer encoder, an action-conditioned Q head,
  a behavior-cloning head used for the support filter, and a target network updated with
  tau 0.05. Other hyperparameters are the defaults in `train.py` (discount 0.95, reward scale
  100, learning rate 3e-4, batch size 256, model width 128, 4 heads, 2 layers).
- Checkpoint selection: after each epoch, compute trajectory weighted importance sampling (WIS)
  on up to 500 validation episodes (clip 20). Keep the checkpoint from the epoch with the
  highest validation WIS. The shipped file is from epoch 9 of 25 under that rule. If OPE
  selection is disabled, the script falls back to lowest validation loss instead.
- Model file sha256: `c51f905181ed9f1072da88728495cf308f16c4a95758686605704a9d84164666`
- Training is not required to use the policy; the checkpoint is included. Re-running training
  can give slightly different weights on different hardware even with the same seed.

## Evaluation and limitations

Results, baselines, slices and the limits of the offline evaluation are in `report.md`.
The main limitation is that the value estimates come from logs of an older policy, with no
simulator available, so all reported performance numbers are off-policy estimates with
uncertainty, not measured outcomes.

## Notes

AI coding tools were used during development.
