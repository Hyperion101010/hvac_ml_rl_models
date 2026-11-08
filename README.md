# HVAC Follow-up Policy Optimization Take-home

## Background

We build AI tools for HVAC contractors. One product helps technicians follow up with homeowners after a completed visit. The system may need to:

- Check that the system is working after a tune-up or repair.
- Follow up on an open estimate for a furnace, central AC, heat pump, mini split, thermostat, or air-quality add-on.
- Ask what is blocking a decision when a homeowner says the quote is expensive, wants to compare competitors, needs to ask a spouse, or wants to wait.
- Nudge a membership renewal or upcoming maintenance visit.
- Stop nudging politely when the homeowner is not engaging.
- Escalate to a human when the customer is unhappy or the situation is risky.

In a typical home-service workflow, a homeowner books a visit, the contractor dispatches a technician, the technician diagnoses the equipment on site, completes the immediate work or leaves an estimate, and the business may follow up afterward about open quotes, maintenance plans, warranties, or unresolved concerns.

Message text is generated elsewhere. In this assignment, you are only deciding the next action.

The goal is not to maximize sends. The goal is to balance conversion, service quality, customer trust, cost, and opt-out risk.

You may use AI coding tools, open-source references, and external learning materials. In the follow-up interview, we will ask you to explain the strategy, data assumptions, failure modes, and evaluation choices in your own words.

## Actions

Your policy must return one of these actions:

| Action | Ends episode? | Meaning |
|---|---:|---|
| `WAIT` | No | Do nothing on this decision step. Useful when the thread is cold or another touch would be too much. |
| `CHECK_IN` | No | Relationship-oriented service check-in. Example intent: "is the AC cooling normally after yesterday's visit?" |
| `ESTIMATE_NUDGE` | No | Sales-oriented but lightweight follow-up on an open estimate. |
| `ASK_OBJECTION` | No | Ask what is blocking the decision: price, spouse, competitor, timing, uncertainty. |
| `OFFER_SCHEDULING` | Sometimes | Move toward booking, CSR help, or final decision. |
| `MEMBERSHIP_TOUCH` | Sometimes | Membership renewal, lapsed-member win-back, or plan-visit reminder. |
| `PARK_THREAD` | Yes | Politely stop proactive nudging for now. |
| `ESCALATE_TO_HUMAN` | Yes | Hand off to a technician, CSR, or safety-sensitive human review. |

## Observation Fields

At each step, your policy receives an `observation` dictionary and the current episode `history`.

Important: use only information visible at decision time. Do not use hidden simulator fields, future rewards, next observations, or final outcomes as policy inputs.

Example fields:

| Field | Type | Meaning |
|---|---|---|
| `case_id` | str | Case identifier. |
| `turn` | int | Decision step, starting at 0. Each episode has at most 4 steps. |
| `day` | int | Simulated day since the original job window. |
| `topic` | str | `open_estimate`, `maintenance_checkin`, `membership_renewal`, `service_issue`, etc. |
| `equipment_type` | str | `central_ac`, `furnace`, `heat_pump`, `mini_split`, etc. |
| `season` | str | `spring`, `summer`, `fall`, or `winter`. |
| `weather_extreme` | bool | Whether there is a heat/cold weather trigger. |
| `track_hint` | str | Noisy hint: `relationship`, `nurture`, or `deal`. This can be wrong. |
| `heat` | str | `cold`, `warm`, or `hot`. |
| `estimate_value_bucket` | str | `none`, `low`, `medium`, or `high`. |
| `membership_status` | str | `none`, `active`, `expiring`, or `lapsed`. |
| `customer_signal` | str | `none`, `positive`, `price_objection`, `competitor`, `ask_spouse`, `timing_delay`, or `unhappy`. |
| `ignored_count` | int | Consecutive proactive touches without a meaningful reply. |
| `touch_count` | int | Total proactive touches in this episode. |
| `user_patience` | int | Remaining tolerance for more bot interaction. |
| `urgency_score` | float | Noisy urgency estimate. |
| `price_sensitivity_score` | float | Noisy estimate of price sensitivity. |
| `service_risk_score` | float | Noisy estimate that this is a service recovery or safety-sensitive situation. |
| `relationship_score` | float | Noisy estimate that this is a relationship/service-care thread rather than a sales thread. |
| `previous_actions` | list[str] | Actions already taken in this episode. |

## Data

You receive historical logs from an older policy:

```text
data/train_logs.jsonl
data/valid_logs.jsonl
```

Each training row is one logged step:

```json
{
  "episode_id": "train_ep_000123",
  "step": 1,
  "observation": {"turn": 1, "track_hint": "nurture", "...": "..."},
  "action": "ASK_OBJECTION",
  "action_prob": 0.21,
  "reward": -1.0,
  "done": false,
  "next_observation": {"turn": 2, "...": "..."},
  "info": {"outcome": "continue", "cost_cents": 0.08, "...": "..."}
}
```

`action_prob` is the old policy's probability of taking the action it actually took.

The logs only contain outcomes for actions the old policy actually chose. You usually do not know what would have happened if another action had been chosen from the same observation. Please discuss this limitation in your report.

We do not provide the scoring simulator. Your local work should rely on the logged train/valid data, your own offline analysis, and any simple simulator or validation method you choose to build. After you submit, we will run your policy on a held-out evaluator and discuss the results with you.

## Policy Interface

Implement:

```python
def act(observation, history):
    """
    Args:
        observation: dict visible at the current decision step.
        history: list of prior step records in this episode.

    Returns:
        One of:
        WAIT, CHECK_IN, ESTIMATE_NUDGE, ASK_OBJECTION,
        OFFER_SCHEDULING, MEMBERSHIP_TOUCH, PARK_THREAD,
        ESCALATE_TO_HUMAN.
    """
```

Start from `starter/policy.py`.

## Local Interface Check

From the package root:

```bash
python starter/check_policy.py --policy starter/policy.py
```

This only checks that your policy imports correctly and returns valid actions for logged observations. It is not a score.

## Required Submission Format

Create a `submission/` directory with this shape:

```text
submission/
  README.md
  policy.py
  train.py                 # optional, if training is required
  models/                  # optional, if you produce model files
  report.md or report.pdf
  requirements.txt
```

`submission/policy.py` must expose:

```python
def act(observation, history):
    ...
```

If your policy needs a trained model file, include the model under `submission/models/` and document exactly how it was produced. If you include `train.py`, it should be runnable from inside `submission/` or your `README.md` should give the exact command.

Please do not depend on paid APIs or network access. The evaluation environment may be offline.

Your report should answer:

1. In product language, when do you check in, nudge an estimate, ask an objection question, offer scheduling, touch membership, park, or escalate?
2. What baselines did you compare against?
3. How does your policy perform overall and on at least 3 slices?
4. What are the main limitations of learning from old logs?
5. Which fields are visible at decision time, and which fields would be leakage?
6. What are the three most likely failure modes?
7. If this went to 5% production traffic, what would you monitor and when would you roll back?

## Baselines To Compare Offline

Compare against at least two meaningful baselines. You may use:

- `always_check_in`
- `always_estimate_nudge`
- `always_escalate`
- `simple_rule`
- a behavior-cloning or logged-policy mimic
- your own rule-based or learned policy

Because you do not have the true simulator, your baseline comparison may use logged validation data, approximate off-policy analysis, a learned reward model, a small simulator you build, or another method you can defend. Be explicit about what your evaluation can and cannot prove.

## Hints

- A relationship customer should not be treated like a hot sales deal.
- Strong sales actions too early can raise opt-outs.
- Asking about the objection can have short-term cost but long-term value.
- Escalation is valuable for unhappy or risky cases, but bad as a universal fallback.
- A policy that only imitates the old policy may learn old mistakes.
- A policy that optimizes only immediate reward may avoid useful multi-step actions.
- Production exploration on real customers would be risky; discuss safer alternatives.
# hvac_ml_rl_models
