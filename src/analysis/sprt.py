"""
Sequential Probability Ratio Test (SPRT) for zero-CRI falsifiability.

This implements Wald's SPRT applied to the VLUI paper's core claim:

    Theorem 2 (Zero Cross-Region Interference):
        For LCHI, CRI_{i→j} = 0  for all i ≠ j.

    Null hypothesis  H₀: |CRI_{i→j}| ≤ ε   (practically zero / ε-isolated)
    Alternative      H₁: |CRI_{i→j}| ≥ δ   (detectable interference)

Each window of the benchmark produces a CRI observation for every (i,j) pair.
The SPRT consumes those observations sequentially and updates log-likelihood
ratios, making a decision (Accept H₀ / Reject H₀ / Continue collecting) as
soon as sufficient evidence accumulates.

Why SPRT over a fixed-sample test?
  - You don't know in advance how many windows are available.
  - SPRT is optimal (Wald's theorem): it minimises expected sample size at
    both H₀ and H₁ while controlling Type-I and Type-II error rates.
  - A continuous "accept/reject" verdict displayed live makes the zero-CRI
    claim empirically falsifiable throughout the benchmark, not just at the end.

Model assumption:
  Under H₀ each |CRI_{i→j}| observation is drawn from a half-normal
  distribution with mean ≈ 0 (zero interference + measurement noise σ).
  Under H₁ it is drawn from a half-normal with mean = δ.
  The log-likelihood ratio increment per observation is:

      llr += log f(x; δ, σ) − log f(x; 0,  σ)
           = (x·δ − δ²/2) / σ²          (half-normal simplification)

Boundaries:
  Accept H₀ when LLR ≤ log(β / (1−α))      → lower boundary B
  Reject H₀ when LLR ≥ log((1−β) / α)      → upper boundary A
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class SPRTDecision(str, Enum):
    CONTINUE  = "continue"   # not enough evidence yet
    ACCEPT_H0 = "accept_h0"  # evidence strongly supports zero CRI
    REJECT_H0 = "reject_h0"  # evidence strongly rejects zero CRI


@dataclass
class PairSPRT:
    """SPRT for a single (i, j) region pair.

    Attributes
    ----------
    i, j        : region indices (i ≠ j)
    delta       : minimum detectable effect (H₁ mean of |CRI|)
    sigma       : assumed noise std dev of |CRI| observations
    alpha       : Type-I  error bound (false rejection of H₀)
    beta        : Type-II error bound (false acceptance of H₀)
    llr         : accumulated log-likelihood ratio
    n_obs       : number of observations consumed
    decision    : current decision
    obs_history : raw |CRI| observations (for audit / plotting)
    """
    i: int
    j: int
    delta: float
    sigma: float
    alpha: float
    beta: float
    llr: float = 0.0
    n_obs: int = 0
    decision: SPRTDecision = SPRTDecision.CONTINUE
    obs_history: list[float] = field(default_factory=list)

    # Wald boundaries (computed once; negative = accept, positive = reject)
    @property
    def boundary_reject(self) -> float:
        """log((1-β)/α)"""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def boundary_accept(self) -> float:
        """log(β/(1-α))"""
        return math.log(self.beta / (1.0 - self.alpha))

    def update(self, cri_obs: float) -> SPRTDecision:
        """Feed one |CRI_{i→j}| observation and update the LLR.

        Uses the log-LR for a one-sided Gaussian test:
            llr += (|x| * δ - δ²/2) / σ²
        """
        if self.decision != SPRTDecision.CONTINUE:
            return self.decision  # already decided; keep decision frozen

        x = abs(cri_obs)
        self.obs_history.append(x)
        self.n_obs += 1
        # Log-likelihood ratio increment (Gaussian, variance σ²)
        self.llr += (x * self.delta - self.delta ** 2 / 2.0) / (self.sigma ** 2)

        if self.llr >= self.boundary_reject:
            self.decision = SPRTDecision.REJECT_H0
        elif self.llr <= self.boundary_accept:
            self.decision = SPRTDecision.ACCEPT_H0

        return self.decision

    def to_dict(self) -> dict:
        return {
            "i": self.i,
            "j": self.j,
            "llr": round(self.llr, 4),
            "n_obs": self.n_obs,
            "decision": self.decision.value,
            "boundary_reject": round(self.boundary_reject, 4),
            "boundary_accept": round(self.boundary_accept, 4),
            "pct_to_reject": min(1.0, self.llr / self.boundary_reject) if self.boundary_reject > 0 else 0.0,
        }


@dataclass
class IndexSPRT:
    """Manages SPRT for every off-diagonal (i,j) pair of an index.

    Parameters
    ----------
    n_regions  : number of partition regions (m)
    delta      : minimum detectable CRI effect (default: 0.05 elasticity)
    sigma      : noise std dev — set to observed noise level; default 0.10
    alpha      : target Type-I  error rate (default 5 %)
    beta       : target Type-II error rate (default 5 %)
    """
    n_regions: int
    delta: float = 0.05
    sigma: float = 0.10
    alpha: float = 0.05
    beta: float  = 0.05

    def __post_init__(self) -> None:
        self._pairs: dict[tuple[int, int], PairSPRT] = {}
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if i != j:
                    self._pairs[(i, j)] = PairSPRT(
                        i=i, j=j,
                        delta=self.delta,
                        sigma=self.sigma,
                        alpha=self.alpha,
                        beta=self.beta,
                    )

    def update_from_matrix(self, cri_matrix: list[list[float]]) -> None:
        """Feed a full CRI matrix snapshot (one window of observations)."""
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if i != j:
                    self._pairs[(i, j)].update(cri_matrix[i][j])

    @property
    def all_pairs(self) -> list[PairSPRT]:
        return list(self._pairs.values())

    @property
    def verdict(self) -> SPRTDecision:
        """Overall index verdict:
        - REJECT_H0   if any pair has been rejected
        - ACCEPT_H0   if all pairs have been accepted
        - CONTINUE     otherwise
        """
        decisions = {p.decision for p in self._pairs.values()}
        if SPRTDecision.REJECT_H0 in decisions:
            return SPRTDecision.REJECT_H0
        if all(d == SPRTDecision.ACCEPT_H0 for d in decisions):
            return SPRTDecision.ACCEPT_H0
        return SPRTDecision.CONTINUE

    @property
    def n_accepted(self) -> int:
        return sum(1 for p in self._pairs.values() if p.decision == SPRTDecision.ACCEPT_H0)

    @property
    def n_rejected(self) -> int:
        return sum(1 for p in self._pairs.values() if p.decision == SPRTDecision.REJECT_H0)

    @property
    def n_pairs(self) -> int:
        return len(self._pairs)

    @property
    def mean_llr(self) -> float:
        """Mean LLR across all off-diagonal pairs."""
        if not self._pairs:
            return 0.0
        return sum(p.llr for p in self._pairs.values()) / len(self._pairs)

    def to_dict(self) -> dict:
        """Serialisable snapshot for the API."""
        return {
            "verdict": self.verdict.value,
            "n_pairs": self.n_pairs,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_continue": self.n_pairs - self.n_accepted - self.n_rejected,
            "mean_llr": round(self.mean_llr, 4),
            "boundary_reject": round(self._pairs[(0, 1)].boundary_reject, 4),
            "boundary_accept": round(self._pairs[(0, 1)].boundary_accept, 4),
            "llr_history": [],   # filled by benchmark runner per update
            "pairs": [p.to_dict() for p in self._pairs.values()],
        }
