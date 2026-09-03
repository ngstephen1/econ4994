# Core Simulation Scenarios

## Purpose

Each scenario is a controlled synthetic world with a known mechanism. A run
must name one focal protected attribute, a reference category, and the
categories being compared. Other demographic fields remain available for
auditing but are not silently treated as equivalent estimands.

The future implementation should build every scenario from the same population
and approval components. Scenario switches should activate declared upstream
shifts, a declared direct effect, both, or neither. This prevents four unrelated
generators from drifting apart.

Numerical treatments are defined separately in
[simulation_calibration.md](simulation_calibration.md) and
[effect_sizes.yaml](../configs/simulation/effect_sizes.yaml). Scenario files
select a level without duplicating treatment values.

## Mechanism matrix

| Scenario key | Upstream group shifts | Direct demographic term | Primary ground-truth expectation |
|---|---:|---:|---|
| fair_baseline | Off | Off | Group effects center on zero across replications |
| direct_discrimination | Off | On | Conditional disparity remains after correct controls |
| upstream_inequality | On | Off | Raw disparity arises through modeled mediators |
| mixed_mechanism | On | On | Adjustment removes only the modeled upstream portion |

## Scenario 1: Fair baseline

### Definition

- Categories of the focal protected attribute may have unequal population
  shares.
- Financial, loan, and context variables use the same generating distributions
  across focal categories.
- The focal protected attribute has no direct term in the latent approval
  equation.
- No proxy path tied to the focal attribute is enabled.

### Expected qualitative result

Raw approval-rate gaps and adjusted group estimates may be nonzero in one
finite sample, especially for a small group, but they should fluctuate around
zero over repeated simulations. Persistent or systematic gaps would indicate a
generator defect, an undeclared dependency, or analysis misspecification.

### Required invariant

Holding all generated underwriting inputs fixed, changing only the focal
protected category must not change approval_probability_true.

## Scenario 2: Direct discrimination

### Definition

- Financial, loan, and context distributions remain comparable across focal
  categories.
- A category-specific direct term is added to the latent approval score:

      z_i = z_financial,i + delta(D_i)

- White is the zero-effect reference. Black receives the configured mild,
  moderate, or strong negative log-odds term. Other race categories receive
  zero in the first experiment.

### Expected qualitative result

The raw disparity should reflect the direct pathway. A correctly specified
model that controls for generated borrower and loan characteristics should
continue to show a group difference. The exact percentage-point difference
will vary with baseline applicant risk even when the log-odds penalty is
constant.

### Required invariant

For otherwise identical records, the difference in latent scores between a
focal category and the reference category must equal the configured direct
effect exactly.

## Scenario 3: Upstream inequality

### Definition

- Race shifts only the declared upstream distributions of annual income,
  credit score, and liquid assets for Black applicants relative to White.
- Each enabled shift must identify the affected variable and distribution
  parameter. Undeclared group shifts are prohibited.
- The demographic direct term in the approval equation is exactly zero.

### Expected qualitative result

Groups should have different raw approval rates because their generated
financial or contextual inputs differ. Adding the complete, correctly
specified mediating controls should substantially reduce the estimated group
gap. It need not become numerically zero in one sample, and a logistic
coefficient comparison alone is not a valid decomposition because of
non-collapsibility.

### Required invariant

Holding all generated underwriting inputs fixed, changing only the focal
protected category must not change approval_probability_true.

## Scenario 4: Mixed mechanism

### Definition

- Declared upstream distribution shifts are enabled.
- A declared direct demographic term is also added to the latent approval
  score.
- The two mechanisms are stored separately in configuration and run metadata.

### Expected qualitative result

The raw group disparity combines upstream and direct pathways. Correct
adjustment should reduce the raw probability gap but leave a conditional
signal associated with the direct term. The remaining adjusted probability gap
will not generally equal the inserted log-odds coefficient.

### Required invariant

Tests must independently verify the configured upstream distribution shifts and
the exact direct contribution to the latent score.

## Planned model comparisons

Later analysis will compare three predeclared statistical specifications:

- Model A: approval as a function of the focal demographic group.
- Model B: Model A plus borrower financial controls.
- Model C: Model B plus loan and property controls.

Context controls should be added as a separate, explicitly motivated
sensitivity specification because they may be mediators or protected-attribute
proxies. The models are not implemented in this phase.

## Scenario-level validation plan

For every scenario, future automated tests should check:

- The scenario name resolves to the intended pair of mechanism switches.
- The direct-effect map has a declared protected attribute and reference
  category whenever the direct path is enabled.
- Upstream shifts identify only supported variables and parameters.
- A fair or direct-only run cannot contain upstream group shifts.
- A fair or upstream-only run cannot contain a direct demographic effect.
- Repeated large samples exhibit the expected direction of configured upstream
  shifts without requiring exact realized group means.
- Counterfactual rescoring that changes only group category obeys the scenario
  invariant.

## Interpretation limits

“Discrimination” is appropriate here only for the direct synthetic mechanism
that the researcher intentionally inserts. A synthetic result shows how a
method behaves under that assumption. It does not demonstrate that a real
lender or the mortgage market uses the same mechanism.
