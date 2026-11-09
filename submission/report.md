# Report: HVAC follow-up policy

## Summary

The submitted policy applies a hard escalation rule first, then a learned value policy
(`transformer_fqi_acq`) restricted to actions the logged policy plausibly took, with a rule
policy (`product_rule`) as a fallback if the model cannot run.

On 5,000 validation episodes, using off-policy estimates, the learned policies score higher
than the best hand-written rule policy (`product_rule`) in two estimators: weighted importance
sampling (WIS, +45.3 for the submitted policy, confidence interval clear of the rule policy's)
and doubly robust estimation with an independently trained value model (DR, +10.8). The size
of the gain is uncertain; it lies somewhere between about +10 and +45 points of discounted
return. Imitating the old policy (behavior cloning, advantage-weighted cloning) does no better
than the rule policies, and often worse.

Important limits, detailed in sections 3 and 4: no separate test set was held out, the checkpoint
was chosen using the validation episodes, and a one-step policy performs about as well as the
multi-step one, so multi-step value is not demonstrated.

## 1. When each action is used

The escalation rule is fixed: an unhappy customer, or a service-risk score above 0.75, always
goes to a human. For the other seven actions the policy is learned, so the table gives the product
meaning of each action, what the logs say about it, and how often the submitted policy uses it.

| Action | Product meaning | What the logs show (old policy, training data) | Share used by submitted policy on validation |
| --- | --- | --- | --- |
| ESCALATE_TO_HUMAN | Hand off to a person for an unhappy or risky case | Escalating unhappy customers averaged +21.1 (135 cases); escalating anyone else averaged -18.6 (1,693 cases). The old policy escalated only 194 of 3,000 unhappy or high-risk steps | 4.3% |
| PARK_THREAD | Stop proactive messages for now | After two or more ignored touches, parking returned -1.0 on average versus -16.1 for an estimate nudge (not controlled for who was parked) | 16.8% |
| CHECK_IN | Service-care message | Opt-out 6.3%, positive review 8.1% of uses | 0.0% |
| ESTIMATE_NUDGE | Light sales reminder on an open estimate | Opt-out 9.5%, sale 0.9% of uses. For relationship customers with no estimate it averaged -26.5 | 1.3% |
| ASK_OBJECTION | Ask what blocks the decision | Opt-out 9.5% of uses, no direct sales. With a price objection on an open estimate it averaged -5.3 versus +33.5 for offering scheduling | 12.8% |
| OFFER_SCHEDULING | Move to booking or a final decision | Booking 16.5%, sale 10.9%, opt-out 10.2% of uses | 15.0% |
| MEMBERSHIP_TOUCH | Renewal, win-back or visit reminder | Conversion 14.8%, booking 7.3%, opt-out 6.6% of uses | 2.5% |
| WAIT | Do nothing this step | Never ends an episode; after two or more ignored touches it averaged -4.5 versus -16.1 for a nudge | 47.3% |

Plain-language reading of the learned part (shares on validation steps matching each slice):
- Customers raising a price, competitor, spouse or timing objection on an open estimate: mostly
  `OFFER_SCHEDULING` (45.3%) or `WAIT` (42.5%); `ASK_OBJECTION` is only 10.4%. The policy often
  moves toward booking instead of asking another objection question.
- Relationship-type customers (high relationship score, no estimate): mostly `WAIT` (71.4%) or
  `PARK_THREAD` (18.4%); sales actions and `CHECK_IN` are essentially unused.
- Customers who have ignored two or more touches: almost always `PARK_THREAD` (98.8%); the
  remainder are escalations under the hard risk/unhappy rule.
- Expiring or lapsed memberships (excluding unhappy/risky and already-ignored cases): mostly
  `WAIT` (69.7%); `MEMBERSHIP_TOUCH` is only 8.2%, so the learned policy does not push renewal
  as aggressively as the hand-written rule would.

The log comparisons in the table are descriptive. The old policy chose actions for reasons the
logs only partly capture, so a gap between two actions does not prove one causes better outcomes.

## 2. Baselines

All baselines were evaluated with the same estimator and the same validation episodes.

| Baseline | What it represents |
| --- | --- |
| `always_check_in` | Relationship-only follow-up |
| `always_estimate_nudge` | Sales-only follow-up |
| `always_escalate` | Escalation as a universal fallback |
| `simple_rule` | The provided hand-coded rule |
| `product_rule` | A hand-coded rule written from the task description (escalate unhappy or risky, park after repeated ignores, ask objection questions, offer scheduling on hot deals, membership touches, check-ins for relationship customers) |
| `transformer_bc` | Behavior cloning: imitates the logged policy |
| `transformer_awbc` | Advantage-weighted behavior cloning: imitation weighted by return |
| `one_step_greedy` | Hard safety, then argmax of a one-step (gamma=0 FQE) Q over BC-supported actions; loads `models/transformer_one_step.pt` |
| `transformer_fqi` | Multi-step value learning with a single Q head, support filter |
| `transformer_fqi_acq` | Same with an action-conditioned Q head (submitted policy) |

## 3. Performance

### How it was measured

No simulator was available, so values are off-policy estimates computed from the validation
logs and the logged action probabilities.
- WIS: each episode is weighted by the product over steps of (probability the evaluated policy
  takes the logged action) / (logged probability). Weights are clipped per step (5 is the
  primary setting, 20 is a sensitivity check). ESS is the effective sample size of the weights;
  a small ESS means a few episodes dominate the estimate.
- The evaluated policies are smoothed so every episode gets weight: probability 0.9 on the
  policy's chosen action and `(1 - 0.9) / 7` on each of the other seven actions
  (`smoothed_pi_prob` in `eval/ope.py`, default `--smooth 0.9`). The numbers therefore describe
  the smoothed version, not the deterministic policy.
- DR: combines the weights with a value model (FQE) trained on the training logs only, so it is
  independent of the validation episodes and of the policy being scored. The FQE checkpoint
  (`models/transformer_fqe.pt`) is behavior FQE: it bootstraps the *logged* next action
  (SARSA-style Q for the behavior policy μ), not the candidate policy's own Q.
- Confidence intervals come from bootstrapping episodes: 200 resamples, percentile interval with
  `alpha=0.05` (95% CI) in `bootstrap_ci`.
- `dWIS` is the plain difference between a policy's WIS and `product_rule`'s WIS. It is not a
  paired bootstrap difference.

These numbers come from an earlier run of the evaluation and were not regenerated when this
folder was packaged. Higher is better; the unit is discounted return (reward scale of the logs).
The old policy's own mean discounted return on the validation episodes is -12.2.

### Overall (clip = 5, primary)

| Policy | ESS | WIS | 95% CI | DR | dWIS |
| --- | ---: | ---: | --- | ---: | ---: |
| transformer_fqi_acq | 121.8 | 47.131 | 35.068 to 54.820 | 6.121 | 45.295 |
| transformer_fqi | 95.1 | 44.750 | 31.287 to 56.450 | 9.936 | 42.914 |
| one_step_greedy | 125.5 | 36.806 | 28.210 to 46.018 | 12.001 | 34.970 |
| product_rule | 145.7 | 1.837 | -5.685 to 11.173 | -4.691 | n/a |
| transformer_awbc | 232.9 | 0.614 | -6.371 to 7.358 | -1.836 | -1.223 |
| simple_rule | 128.8 | -3.037 | -8.171 to 2.342 | -4.328 | -4.874 |
| always_escalate | 176.9 | -11.548 | -13.999 to -9.262 | -16.791 | -13.385 |
| always_check_in | 115.5 | -17.893 | -21.305 to -14.142 | -16.357 | -19.730 |
| transformer_bc | 285.4 | -19.589 | -22.488 to -16.350 | -16.187 | -21.426 |
| always_estimate_nudge | 174.4 | -35.350 | -41.476 to -29.731 | -36.272 | -37.187 |

### Sensitivity (clip = 20)

| Policy | ESS | WIS | CI | DR |
| --- | ---: | ---: | --- | ---: |
| transformer_fqi | 3.7 | 81.294 | 26.541 to 102.702 | 46.402 |
| transformer_fqi_acq | 78.0 | 34.406 | 23.711 to 43.669 | 11.339 |
| one_step_greedy | 44.0 | 25.232 | 14.289 to 37.053 | 14.206 |
| transformer_awbc | 157.9 | -0.662 | -6.721 to 5.306 | -1.662 |
| product_rule | 113.9 | -0.788 | -10.035 to 7.548 | -3.236 |
| simple_rule | 112.4 | -1.213 | -5.728 to 3.492 | -3.528 |
| always_escalate | 121.8 | -15.356 | -17.350 to -12.742 | -17.756 |
| transformer_bc | 123.4 | -16.098 | -18.612 to -13.225 | -15.714 |
| always_check_in | 2.3 | -23.978 | -30.735 to -15.217 | -17.667 |
| always_estimate_nudge | 47.6 | -36.546 | -42.431 to -30.780 | -40.368 |

With clip 20, `transformer_fqi` (ESS 3.7) and `always_check_in` (ESS 2.3) are dominated by a few
episodes and should be disregarded. The submitted policy stays stable (ESS 78). That stability
is the main reason it was preferred over `transformer_fqi`.

### Slices (clip = 5, WIS with ESS in parentheses; slices with ESS below 30 are hidden)

Slice definitions (from `eval/ope.py`, episode included if any step matches):
- `objection`: `customer_signal` in `{price_objection, competitor, ask_spouse, timing_delay}`
- `high_relationship`: `relationship_score >= 0.7`
- `unhappy_or_risk`: `customer_signal == "unhappy"` or `service_risk_score > 0.75`
- `late_ignored`: `turn >= 2` and `ignored_count >= 2`

Sizes: objection n=2733, high relationship n=1931, unhappy or risk n=466, late ignored n=1715.

| Policy | Objection | High relationship | Unhappy or risk | Late ignored |
| --- | --- | --- | --- | --- |
| transformer_fqi_acq | 63.545 (64.3) | hidden (28.9) | hidden (28.6) | hidden (9.8) |
| transformer_fqi | 67.593 (41.6) | 0.446 (61.9) | 10.813 (31.0) | hidden (8.3) |
| one_step_greedy | 45.045 (52.0) | 8.112 (61.7) | hidden (29.2) | hidden (2.7) |
| product_rule | -3.152 (44.8) | 0.493 (70.6) | 11.261 (30.6) | hidden (12.0) |
| transformer_awbc | 7.938 (112.8) | -23.669 (77.2) | 7.995 (35.0) | hidden (21.0) |
| simple_rule | -8.498 (51.1) | -9.140 (86.3) | 8.660 (35.3) | hidden (14.4) |
| always_escalate | -13.485 (59.9) | -10.923 (62.0) | hidden (29.1) | -24.017 (32.2) |
| always_check_in | -21.238 (41.5) | -17.385 (76.4) | -31.569 (46.1) | hidden (11.0) |
| transformer_bc | -20.071 (171.3) | -25.016 (96.6) | 7.015 (37.6) | hidden (17.7) |
| always_estimate_nudge | -22.033 (235.5) | hidden (26.8) | hidden (5.8) | -48.268 (32.7) |

### What the results support

- The learned value policies beat `product_rule` in WIS (non-overlapping intervals at clip 5)
  and in DR, and at both clip settings for the submitted policy. They beat all imitation and
  always-one-action baselines clearly.
- The gain is concentrated in objection cases: 63.5 for the submitted policy versus -3.2 for
  `product_rule` (ESS 64 and 45). This matches the log pattern that offering scheduling
  returns far more than asking the objection question in those states, which `product_rule`
  always asks. The log pattern may partly reflect who received which action.
- On high-relationship customers the learned policy `transformer_fqi` (0.4) matches
  `product_rule` (0.5). Behavior cloning (-25.0) and advantage-weighted cloning (-23.7) do
  badly, which fits the old policy having used sales actions on relationship customers.
- For unhappy or risky customers all policies that include the escalation rule score about
  +7 to +11, while `always_check_in` scores -31.6.

### What the results do not support

1. The size of the gain. WIS gives about +45 over `product_rule`; DR gives about +11 for the
   submitted policy. The two estimators agree for policies near the old behavior and disagree
   for the learned ones. The honest statement is "better, by an amount between about +10 and +45".
2. A winner among the learned policies. Their intervals overlap, DR ranks `one_step_greedy`
   first and WIS ranks `transformer_fqi_acq` first.
3. A multi-step advantage. `one_step_greedy` is statistically close to the multi-step
   policies, so the data do not show that planning beyond one step helps.
4. Clean numbers. The `transformer_fqi` and `transformer_fqi_acq` checkpoints were each chosen
   as the best epoch by validation WIS, so their WIS is biased upward by the selection. There is
   no separate test set. DR uses a value model trained only on the training logs and is the more
   conservative figure.
5. Any statement about customers who ignore several touches. That slice has an ESS below 30
   for every learned policy and could not be evaluated. The only evidence is descriptive:
   in the logs, parking after two or more ignores returned -1.0 versus -16.1 for a nudge.

Ablations that were not run: removing the recency features (days since last inbound or
outbound message), and a support threshold of 0.05 versus 0.01. They are not claimed here.

## 4. Limits of learning from old logs

- Logs show the outcome only for the action the old policy took. For every other action in
  the same situation there is no result, so the value of untried actions is extrapolated.
- The old policy made clear mistakes that imitation copies. Of 1,828 logged escalations, 1,634
  were for customers who were neither unhappy nor above a 0.75 risk score, and they averaged
  -18.6. Meanwhile only 194 of 3,000 unhappy or high-risk steps were escalated.
- Action selection by the old policy depends on information the logs only partly record, which
  biases the apparent effect of actions (selection bias).
- Some actions are rare in the logs (park 1.7% of steps, escalate 2.4%, membership touch 4.8%,
  wait 6.9%), so estimates for them rest on little data and their importance weights are large
  and clipped, which biases estimates for policies that use them more.
- Off-policy estimates assume the logged action probabilities are correct and that no hidden
  factor drives both the old policy's choices and the outcomes. Neither can be checked here.
- The evaluation compares a smoothed version of each policy, and uses a clip on the weights.
  Both change the quantity being estimated.

## 5. Fields visible at decision time, and leakage

Visible at decision time and used by the model (from `submission/policy.py` encode path):
- Categories: `topic`, `equipment_type`, `season`, `track_hint`, `heat`, `estimate_value_bucket`,
  `membership_status`, `customer_signal`, `weather_extreme`.
- Numbers: `turn`, `day`, `ignored_count`, `touch_count`, `user_patience`, `urgency_score`,
  `price_sensitivity_score`, `service_risk_score`, `relationship_score`.
- Earlier steps of the same episode: their observations and the actions taken (from `history`).

Also present in the observation but not used as direct inputs: `case_id` (only used to
match history to the episode), `previous_actions` and `last_action` (used when rebuilding a
synthetic history if the logged history does not match), `days_since_last_inbound`,
`days_since_last_outbound`. These are visible at decision time and could be used; the model
does not read the two recency fields as features.

Not available at decision time (would be leakage if used as inputs):
- `reward` of the current or later steps
- `next_observation`
- `done`
- everything in `info`: `outcome`, `cost_cents`, `latency_ms`, `invalid_action`
- `action_prob` (exists only in the logs)

The noisy fields (`track_hint`, and the urgency, price sensitivity, risk and relationship scores)
are visible but can be wrong; the policy treats them as evidence, not truth.

## 6. Three most likely failure modes

1. Extrapolation to rarely tried actions. The value head can be wrong where the old policy
   seldom acted, and the held-out simulator may reward actions the logs say little about. The
   support filter (threshold 0.01) reduces but does not remove this. Sign to watch: the action
   mix moving far from the logged mix, or sales actions on customers with weak positive signals.
2. Noisy score fields leading to the wrong tone. If the relationship or risk scores are
   misestimated, the policy can push sales at relationship customers (opt-outs rise with
   sales actions: 5.6% at turn 0 to 14.8% at turn 3 in the logs, and 13.1% versus 7.5% for
   high versus low relationship scores), over-park, or under-escalate in cases below the 0.75
   rule threshold.
3. Input or load failure. A different history format, a missing field, or a torch or version
   problem could break the model path. The policy tolerates malformed inputs and falls back to
   the rule policy with a warning, but the fallback has lower estimated value and a rise in the
   fallback rate would signal an integration problem.

## 7. If this went to 5% of production traffic

Rollout plan:
1. Shadow mode first: run the policy alongside the live one, log what it would do, and compare
   action mixes and flagged cases before sending any messages.
2. Then 5% of traffic in low-risk segments, with the escalation rule in front, and the old
   policy as the control arm.
3. Hold at 5% until the stopping rules below have enough data, then widen only on review.

Control rates from the training logs (per episode), to compare against:

| Measure | Rate |
| --- | ---: |
| Episode ends in opt-out | 24.5% |
| Booked appointment | 8.1% |
| Sold estimate | 5.5% |
| Positive review | 4.5% |
| Membership converted | 2.2% |
| Escalated | 7.6% |
| Parked | 5.3% |
| Reaches the step limit with no outcome | 42.2% |
| Average steps per episode | 3.11 |
| Unhappy or high-risk steps that were escalated | 6.5% (194 of 3,000) |

What to monitor:
- Opt-out rate (primary safety measure), booking plus sale rate (primary benefit measure).
- Share of unhappy or high-risk customers escalated (should be 100%, since a rule enforces it).
- Escalations that were not needed, messages sent per customer, and action mix against the
  control arm.
- Fallback rate, errors and latency.

Proposed rollback triggers (these are proposals to be tuned; they are not derived from data
beyond the control rates above):
- Opt-out rate higher than control by 3 percentage points or more, with at least 2,000
  episodes in each arm (about two standard errors of the difference at the 24.5% control rate).
- Booking plus sale rate lower than control by 2 percentage points or more, with the same
  sample size (about two standard errors at the 13.6% control rate).
- Any unhappy or high-risk customer not escalated, at any time.
- Fallback or error rate above 1% of calls.
- Action mix drift: any action's share changing by more than half its expected share for two
  consecutive days.

Safer alternatives to exploring on real customers:
- Offline evaluation first (as done here), with more careful estimators and a held-out test set.
- Shadow mode and replay of recent conversations.
- If exploration is needed, randomize only among actions already judged safe for that customer
  (for example never randomizing into sales actions for relationship customers), keep
  the escalation rule outside the randomization, cap the exploration share, and log the
  action probabilities so future data supports better off-policy estimates.

## Appendix: reproducibility

- Evaluation command used: `python3 -m eval.run_ope --q-ckpt models/transformer_fqe.pt
  --policies product_rule,simple_rule,always_check_in,always_estimate_nudge,always_escalate,
  transformer_bc,transformer_awbc,transformer_fqi,transformer_fqi_acq,one_step_greedy`
  (smoothing 0.9, clip 5 and 20, slices hidden when ESS is below 30). Numbers in section 3 are
  from that run and were not regenerated while packaging.
- Training command and checkpoint hash: see `README.md`.
