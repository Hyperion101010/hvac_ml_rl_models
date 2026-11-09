# Report: HVAC follow-up policy

## Summary

The policy has one fixed rule that always runs first: if the customer is unhappy or the
service-risk score is high, the case goes to a person. For everything else, a trained model picks
the action. If the model can't run, a simple hand-written rule takes over so the policy always
gives a valid answer.

I didn't have a simulator, so I estimated how well it does by replaying the old system's logs. By
those estimates the trained model beats my best hand-written rule on two separate checks. How much
better is uncertain: somewhere between roughly +10 and +45 points of reward. Copying what the old
system did turned out to be no better than the hand-written rules, and often worse.

Three caveats matter, and I explain them in section 3. I didn't set aside a separate test set, I
picked the final model version using the same validation data I report on, and I couldn't show that
planning more than one step ahead helps. A fourth, smaller one: the figures come from an earlier
run, on an earlier version of the policy code (see the note under section 1).

## 1. When each action is used

The escalation rule comes first. Everything else is the model's call. The table compares how often
the old system used each action with how often my policy used it on the validation data.

| Action | Old system | My policy | When it gets used |
| --- | ---: | ---: | --- |
| Escalate to human | 2.4% | 4.3% | Unhappy or high-risk customers (fixed rule). The old system missed most of these and escalated plenty of customers who didn't need it. |
| Park thread | 1.7% | 16.8% | Often for relationship-type customers and for customers who have ignored our messages. |
| Wait | 6.9% | 47.3% | The most common choice, especially for relationship-type customers and for expiring or lapsed memberships. |
| Offer scheduling | 13.8% | 15.0% | When a customer raises a price, competitor, spouse or timing concern about an open estimate, it often moves toward booking. |
| Ask about the objection | 21.1% | 12.8% | Used less than the old system did. In the logs, offering scheduling did much better than asking. |
| Membership touch | 4.8% | 2.5% | Rarely, even for expiring or lapsed memberships. |
| Estimate nudge | 31.4% | 1.3% | Almost never. In the logs it led to far more opt-outs than sales. |
| Check in | 17.9% | 0.0% | Never. In the logs it led to nearly as many opt-outs as positive reviews. |

In plain terms, this is a quiet policy. It waits or parks about two thirds of the time, which is a
lot, and I come back to that in section 6.

*Note on these shares.* I measured them on an earlier version of the policy code, which may also have
included a rule to park customers after two ignored messages. The submitted file does not have that
rule, so the Park and Wait shares for the submitted file could differ somewhat. The other rows should
be close. The log comparisons are descriptive only: the old system picked actions for reasons the
logs only partly show, so they don't prove that one action causes better results than another.

## 2. What I compared against

| Policy | What it stands for |
| --- | --- |
| Always check in | Only friendly follow-ups |
| Always nudge the estimate | Only sales pressure |
| Always escalate | Handing everything to a person |
| Simple rule (`simple_rule`) | The hand-written rule that came with the task |
| Product rule (`product_rule`) | A hand-written rule based on the task description: escalate unhappy or risky customers, park after repeated ignores, ask about objections, offer scheduling on hot deals, check in with relationship customers |
| Behavior cloning (`transformer_bc`) | Copies what the old system did |
| Advantage-weighted cloning (`transformer_awbc`) | Copies the old system, but pays more attention to cases that went well |
| One-step model (`one_step_greedy`) | A trained model that only looks at the immediate result |
| Multi-step model v1 (`transformer_fqi`) | A trained model that looks ahead over the whole conversation |
| Multi-step model v2 (`transformer_fqi_acq`) | The same idea with a different way of scoring actions. This is the one I submit. |

Everything was scored the same way on the same 5,000 validation conversations.

## 3. How well does it do?

**How I measured it.** I replayed the logged conversations and gave more weight to the ones where
what the old system did matches what the policy would have done. To get stable numbers I scored a
slightly softened version of each policy (90% on its chosen action, 10% spread over the others) and
capped very large weights. As a second opinion I used a separate prediction model trained only on
the training logs. The ranges are 95% ranges from resampling the conversations. Higher is better,
and the old system averages -12.2 on this data. These numbers come from an earlier run and I didn't
re-run them when I packaged the folder.

**Overall**

| Policy | Estimated score | Likely range | Second opinion |
| --- | ---: | --- | ---: |
| Multi-step model v2 (submitted) | 47.1 | 35.1 to 54.8 | 6.1 |
| Multi-step model v1 | 44.8 | 31.3 to 56.5 | 9.9 |
| One-step model | 36.8 | 28.2 to 46.0 | 12.0 |
| Product rule | 1.8 | -5.7 to 11.2 | -4.7 |
| Advantage-weighted cloning | 0.6 | -6.4 to 7.4 | -1.8 |
| Simple rule | -3.0 | -8.2 to 2.3 | -4.3 |
| Always escalate | -11.5 | -14.0 to -9.3 | -16.8 |
| Always check in | -17.9 | -21.3 to -14.1 | -16.4 |
| Behavior cloning | -19.6 | -22.5 to -16.4 | -16.2 |
| Always nudge the estimate | -35.4 | -41.5 to -29.7 | -36.3 |

I also repeated the evaluation with a looser cap on the weights. The submitted policy held up
(score 34.4, range 23.7 to 43.7). Multi-step model v1 and "always check in" did not: their results
ended up depending on a handful of conversations, so I'd ignore them. That stability is the main
reason I chose v2 over v1.

**By customer type** ("n/a" means there wasn't enough data to trust the number)

| Policy | Objection | High relationship | Unhappy or risky |
| --- | ---: | ---: | ---: |
| Multi-step model v2 (submitted) | 63.5 | n/a | n/a |
| Multi-step model v1 | 67.6 | 0.4 | 10.8 |
| One-step model | 45.0 | 8.1 | n/a |
| Product rule | -3.2 | 0.5 | 11.3 |
| Advantage-weighted cloning | 7.9 | -23.7 | 8.0 |
| Simple rule | -8.5 | -9.1 | 8.7 |
| Always escalate | -13.5 | -10.9 | n/a |
| Always check in | -21.2 | -17.4 | -31.6 |
| Behavior cloning | -20.1 | -25.0 | 7.0 |
| Always nudge the estimate | -22.0 | n/a | n/a |

"Objection" means the customer raised a price, competitor, spouse or timing concern. "High
relationship" means a relationship score of 0.7 or more. "Unhappy or risky" means unhappy, or a
service-risk score above 0.75. For two of these three groups there wasn't enough data for the
submitted policy itself, so I lean on v1, which behaves similarly. I also couldn't evaluate
customers who keep ignoring us for any of the trained policies.

**What the results show.** The trained models beat the hand-written rules on both checks, and they
clearly beat copying the old system and the "always do one thing" policies. Most of the gain comes
from objection cases. The product rule always asks about the objection first, but the logs show
that offering scheduling does far better. For relationship customers the best trained policy only
matches the product rule, while the two copying approaches do badly, which fits the old system's
habit of pushing sales on those customers. For unhappy or risky customers every policy with the
escalation rule does about equally well, and "always check in" does badly.

**What they don't show.**
- *How big the gain is.* The main estimate says about +45 over the product rule, and the second
  opinion says about +11. They agree for policies that act like the old system and differ for the
  trained ones. My guess is that the second opinion undersells gains that build up over several
  steps, but I didn't test that.
- *Which trained model is best.* Their ranges overlap, and the two checks rank them differently.
- *That looking further ahead helps.* The one-step model does about as well as the multi-step ones.
- *Clean numbers.* I picked each multi-step model's best training round using this same validation
  data, which flatters their scores, and I have no separate test set. The second opinion is less
  affected by this.
- *Anything about customers who keep ignoring us.* The logs suggest parking them is better than
  nudging them, but I couldn't measure it directly for the trained policies.
- *The exact submitted file.* The numbers come from an earlier version of the policy code (see the
  note under section 1). I didn't re-run the evaluation on the final file.

I also didn't test removing the "days since last contact" fields, or using a stricter filter (0.05
instead of 0.01) on which actions the model may consider. I make no claims about either.

## 4. Limits of learning from old logs

The biggest limit is that the logs only tell you what happened for the action the old system took.
For every other action in the same situation there is no result, so the value of those actions is
a guess.

On top of that, the old system made clear mistakes that copying would repeat. Most of its
escalations were for customers who were neither unhappy nor high-risk, and it escalated only a
small fraction of the ones who were. It also chose actions for reasons the logs only partly show,
so some actions look better or worse than they really are. Some actions are rare in the logs
(parking, escalating, membership touches, waiting), which makes estimates about them less reliable.

Finally, my estimates assume the logged probabilities are correct and that nothing hidden affected
both the old system's choices and the outcomes. I can't check either. And since I scored a softened
version of each policy with capped weights, the numbers describe that version, not the exact policy.

## 5. What the policy can see, and what would be leakage

**Visible at decision time and used:** the topic, equipment type, season, a hint about the customer
type, how hot the lead is, estimate size, membership status, what the customer just said, a weather
flag, the turn and day, ignored messages, touches so far, remaining patience, the urgency,
price-sensitivity, service-risk and relationship scores, and the earlier steps of the same
conversation along with the action taken at each.

**Visible but not used as inputs:** the case ID (only used to match up history), the list of
previous actions, and the two "days since last inbound/outbound message" fields.

**Not available at decision time, so using them would be leakage:** the reward, the next observation,
the done flag, the outcome record (outcome, cost, latency, invalid-action flag), and the old
system's action probability.

The customer-type hint and the four scores are noisy and can be wrong, so the policy treats them as
evidence, not as facts.

## 6. Three most likely failure modes

1. **It's too quiet.** On the version I measured, it waited almost half the time, parked far more
   than the old system did, and never checked in. The logs reward this because waiting rarely
   causes an opt-out. But if the real environment penalizes silence, or a parked customer is worth
   more than my short window shows, it will underperform. It can also go wrong where the old system
   rarely tried an action. Because the submitted file has no fixed rule for customers who keep
   ignoring us, that case is left to the model (or to the backup rule, which parks them after two
   ignored messages), so I'd watch the action mix and the share of wait and park.
2. **Noisy scores push it into the wrong tone.** If the relationship or risk scores are off, it
   could push sales at a relationship customer (in the logs, opt-outs after a sales message grew
   with each step and were higher for high-relationship customers), park too early, or miss an
   escalation that falls just under the 0.75 threshold.
3. **Something breaks around the model.** A different data format, a missing field or a software
   version issue could stop the model from running. The policy handles bad input and falls back to
   the hand-written rule, which scores lower, so a rising fallback rate would be the warning sign.

## 7. If this ran on 5% of real traffic

**Rollout.** I'd start in the background, without sending anything, and compare its choices with
the live system. Then I'd move 5% of low-risk traffic onto it with the escalation rule on, and
leave the rest on the old system as the comparison group. I'd stay at 5% until there is enough data
on the measures below.

**Reference rates.** In the training logs, about one in four conversations ends in an opt-out,
about 8% end in a booked appointment and 5.5% in a sale.

**What I'd watch.** The opt-out rate is the main safety measure, and booked plus sold is the main
benefit. I'd also track unneeded escalations, messages per customer, the share of actions that are
wait or park, and how often the fallback or errors occur.

**When I'd roll back.** These thresholds are my proposals and aren't derived from the data:
- the opt-out rate is 3 points or more above the comparison group, or booked plus sold is 2 points
  or more below it, with at least 2,000 conversations in each group;
- any unhappy or high-risk customer isn't escalated, at any time;
- the fallback or error rate goes above 1% of calls;
- any action's share moves by more than half its expected level for two days in a row.

**Safer ways to learn than experimenting on customers.** Keep checking offline first, with a proper
test set, and replay recent conversations. If live experiments are really needed, randomize only
among actions already judged safe for that customer, keep the escalation rule outside the
randomization, cap how much traffic it touches, and record the probabilities used so the data is
easier to analyze later.

## Reproducing the numbers

The evaluation script and the extra models used for the comparisons (behavior cloning,
advantage-weighted cloning, the one-step model and the second-opinion model) are not part of this
folder. They are kept in the project history on the `exp-changes` branch, at commit `9e28094`. From
that commit, the figures come from:

```bash
python3 -m eval.run_ope --q-ckpt models/transformer_fqe.pt --policies product_rule,simple_rule,always_check_in,always_estimate_nudge,always_escalate,transformer_bc,transformer_awbc,transformer_fqi,transformer_fqi_acq,one_step_greedy
```

The training command and the model checksum are in `README.md`.