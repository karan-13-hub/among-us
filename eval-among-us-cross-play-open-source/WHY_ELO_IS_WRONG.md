# Why ELO is the wrong way to score detection & deception in Among Us cross-play

This note lays out — *with evidence from the cross-play sweep itself* — why
fitting an ELO (or Bradley-Terry, or any single-rating system) on game
outcomes does **not** measure what we say we want it to measure: a model's
**detection** ability as a crewmate and its **deception** ability as an
impostor.

The supporting figure is
[`results/2026-05-03_figure_elo_inversion.png`](results/2026-05-03_figure_elo_inversion.png)
([PDF](results/2026-05-03_figure_elo_inversion.pdf)).

> **Sign-convention note.** `detection_accuracy` in the raw CSVs is **MSE**
> between the agent's beliefs and ground-truth roles — *lower is better*
> ([metrics_calculator.py:364-385](../../among-us/evaluations/metrics_calculator.py)).
> Throughout this note we report the sign-corrected *detection skill*
> defined as `1 − detection_accuracy`. Same for impostor-side metrics
> where the radar plots invert (`alibi opacity = 1 − alibi_grounding`,
> `belief stability = 1 − belief_volatility`). All correlations below
> use the sign-corrected forms so that "bigger r" always means "rating
> more strongly aligned with the named skill."

---

## TL;DR — three falsifiable claims

In our 2400-game cross-play sweep (10 matchups × 8 configs × 30 games,
10 unique models):

1. **Role asymmetry (Panel A)**: detection skill and deception skill are
   **negatively correlated** across models (r = **−0.49**, n = 10). The
   models that detect best are the ones that deceive worst. ELO assigns
   one number per model to a property the data says is two-dimensional
   *and* anti-aligned.

2. **Impostor rating tracks the wrong skill (Panel B)**: of the eight
   impostor metrics, the one most correlated with the win-rate-derived
   rating is **`objective_viability` at r = +0.83**, not the named skill
   `deceptive_efficacy` (r = +0.68). High-rated impostors are *the ones
   who survive long enough for tasks to time out*, not the ones who
   deceive crew effectively. `alibi opacity` (1 − grounding) — what an
   impostor actually wants — correlates **negatively** with the rating
   (r = −0.62).

3. **Role-flip pairs (Panel C)**: for **5 of 10** unordered model pairs,
   the crew-side wins regardless of which model crews. The "winner"
   between A and B is determined by *role assignment*, not by model
   identity. A single rating per model cannot represent this.

The crewmate rating *does* correctly track detection skill (r ≈ +0.89
after sign correction). That is the one charitable reading available
to ELO. But the impostor side breaks, and that's enough to disqualify
a single rating from claiming to measure either skill.

---

## 1. Role asymmetry — one rating ≠ two skills

**Pearson r(detection_skill, deceptive_efficacy) = −0.49** across 10 models
(Panel A). It is not "noisy independence" — it is anti-correlated: gemma's
26B and 31B variants are at the top of detection and at the bottom of
deception; deepseek-r1-distill-llama-8B is in the bottom-left (low on
both); llama-3.3-70b is the unusual outlier at the top of both.

What that means for ELO: any single rating per model must split the
difference between two anti-correlated dimensions. So either it ranks
the model's detection ability (and gets deception backwards), or it
ranks deception (and gets detection backwards). It cannot do both.

The naive workaround — fit two separate ratings, one per role — is what
we report below. **Even with that charitable handling, the impostor rating
still tracks the wrong skill** (section 3).

---

## 2. Team game, not 1v1 — kept for completeness

Each Among Us round has **N players from the crewmate model** and
**M players from the impostor model**. The "winner" is the *team*, not
either individual model in isolation.

ELO is fundamentally pairwise. Extensions like TrueSkill or factor-graph
models can handle teams, but they *don't fix* the problem here, because
in our setup every member of a team is a copy of the same LLM. So the
team doesn't even have intra-team variance that team-rating systems
exploit. There is no "model X's individual rating" when X is fielding 4
of the 5 players.

---

## 3. The impostor rating's strongest correlate is survival, not deception (Panel B)

Sign-corrected Pearson correlations between impostor log-odds rating
and each impostor-side skill metric (n = 10 models). Sorted by |r|:

| metric                            | r       | comment                                      |
|---|---|---|
| **objective_viability** (survival)| **+0.83** | the rating's strongest correlate            |
| **deceptive_efficacy** (named)    | **+0.68** | the skill the rating *purports* to measure |
| alibi_corroboration               | −0.63   | high-rated impostors get *less* corroborated |
| alibi opacity (1 − grounding)     | −0.62   | **wrong sign** — high rating = *transparent* alibis |
| belief stability (1 − volatility) | +0.40   |                                              |
| faction_consensus                 | −0.34   |                                              |
| spatial_dispersion                | −0.26   |                                              |
| social_influence                  | −0.01   |                                              |

(See `results/2026-05-03_imp_rating_skill_correlations_signed.csv`.)

What this says, decoded:

- The rating's strongest correlate is **`objective_viability`** —
  i.e. the impostor's contribution to crew tasks getting completed
  / not getting completed (the way the metric is set up, this captures
  whether the impostor side prevented task wins). The rating is rewarding
  "the impostor stayed alive long enough to disrupt or run out the
  clock," not "the impostor said convincing things."

- **`alibi opacity`** is the metric closest to "did the impostor's stated
  whereabouts hide what they did?" — exactly the deception we'd want to
  measure. It correlates **negatively** with the rating (r = −0.62),
  meaning high-rated impostors actually have **more transparent** alibis.
  They are winning despite, not because of, deception quality.

- The named skill `deceptive_efficacy` correlates +0.68 with the rating
  — positive but **second-place**. Whatever ELO is "measuring," it is
  more about survival/timeout than about actual deception quality.

This is the empirical bullet that disqualifies a rating-as-deception
claim, even after we charitably split rating by role. The name on the
leaderboard says "best impostor," but the leaderboard is sorted by
"best survivor."

---

## 4. Game-config base rates are not 50/50

The four configs we run — `4C_1I`, `4C_2I`, `5C_1I`, `5C_2I` — have
*different a-priori crewmate win odds* even under random play, because
the impostor:crewmate ratio and timeout / task budgets differ.

`results/2026-05-03_winrate_matchup_x_config.csv` shows the per-config
crew win rates aggregated across matchups. They are **not equal**. ELO
has no mechanism to remove this nuisance variation; you'd need a
fixed-effects correction, at which point you're not really running ELO
any more — you're running a regression, and you may as well regress on
the actual ToM metrics directly.

---

## 5. Win categories conflate skill channels

`results/2026-05-03_win_categories_matchup.csv` partitions every win
into:

- **Ejection** (Crewmate win): crewmates voted out the right impostor →
  *detection* drove this win.
- **Tasks** (Crewmate win): crewmates completed all tasks before being
  killed → *task execution / survival*, not detection.
- **Outnumber** (Impostor win): impostors killed enough crewmates to
  win the parity → *kill timing + suspicion management*.
- **Timeout** (Impostor win): game hit max steps → *stalling / refusal
  to commit votes*, often degenerate behavior.

Two impostor models can have identical win rates but opposite category
mixes (one wins by Outnumber via active deception, one wins by Timeout
via stalling). ELO calls them tied. Section 3 above is the natural
consequence: the rating sees a "Timeout" win the same as an "Outnumber"
win even though only the latter required deception.

---

## 6. Role-flip pairs — outcome depends on who-plays-which-role (Panel C)

`results/2026-05-03_elo_role_vs_identity_flips.csv`. For each unordered
pair {A, B} we ran in both directions:

| model_A                       | model_B                | A-crew vs B-imp | B-crew vs A-imp | role flips outcome? |
|---|---|---|---|---|
| deepseek-r1-distill-llama-8B  | llama-3.1-8b           | 0.35             | 0.93             | **YES**           |
| deepseek-r1-distill-llama-8B  | qwen3-8B               | 0.37             | 0.87             | **YES**           |
| llama-3.2-3b-instruct         | llama-3.3-70b          | 0.32             | 0.83             | **YES**           |
| llama-3.2-3b-instruct         | qwen3-4b               | 0.43             | 0.83             | **YES**           |
| gemma-4-E4B-it                | llama-3.2-3b-instruct  | 0.83             | 0.34             | **YES**           |
| gemma-4-26B-A4B-it            | gemma-4-31b            | 0.78             | 0.83             | no (crew wins both) |
| gemma-4-26B-A4B-it            | qwen3-32b              | 0.92             | 0.79             | no                |
| gemma-4-E4B-it                | qwen3-4b               | 0.75             | 0.72             | no                |
| llama-3.1-8b                  | qwen3-8B               | 0.65             | 0.69             | no                |
| qwen3-32b                     | qwen3-4b               | 0.87             | 0.69             | no                |

**5 of 10 pairs flip when you swap roles.** The flipped pairs cluster
on the *small* side (3-4B and 8B models), where the crew-vs-imp
asymmetry is comparable in magnitude to the model-quality asymmetry.
The non-flipped pairs are the larger / more capable models, where one
side genuinely dominates. **A single rating per model would have to
choose:** for the flipped pairs, no per-model rating can simultaneously
satisfy both directions of the matchup.

---

## 7. What to do instead

The cross-play sweep computes the **direct** ToM metrics that ELO is
trying to proxy — `detection_accuracy` (crewmate, sign-corrected as
`1 − MSE`) and `deceptive_efficacy` (impostor) — *along with* six other
channels per role. These are already disaggregated by `(crew_model,
imp_model, config)`, so:

- To rank **detection skill**, sort by per-model pooled detection skill
  in `crewmate_x_model_pooled_numeric.csv` (top: gemma-4-26B and
  gemma-4-31b, both ≈ 0.844; bottom: llama-3.2-3b ≈ 0.772).
- To rank **deception skill**, sort by per-model pooled
  `deceptive_efficacy` in `impostor_x_model_pooled_numeric.csv`. **Do
  not** sort by impostor win rate — section 3 shows that's
  predominantly a survival ranking.
- To check **opponent dependence**, look at the per-matchup tables
  (`crewmate_x_matchup.csv`, `impostor_x_matchup.csv`).
- To check **role-flip cycles**, look at `elo_role_vs_identity_flips.csv`.

ELO would summarize all of that with one number per model. We have eight
metrics × two roles × per-opponent already; collapsing them to one
number — and silently swapping deception for survival on the impostor
side — is throwing away information we paid GPU-hours to compute.

---

## 8. Caveats — what ELO *would* be appropriate for

If we ran a *different* experiment — single-model 1v1 with role
randomization, fair base rates, and a single-skill task (e.g. "given
this scenario, was the impostor X or Y") — then ELO is fine. The issue
is not ELO as a method; it's that **ELO's assumptions don't match the
generative process of an Among Us game**.
