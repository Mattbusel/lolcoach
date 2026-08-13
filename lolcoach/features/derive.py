"""Stage 2b entry point: derive coaching features for every ingested match.

This module is the stable API the dataset stage calls. The estimation itself
lives in the focused modules beside it, which is where the actual reasoning
is documented and unit tested:

============================  ===============================================
:mod:`lolcoach.features.geometry`    Map constants, lane projection, zones.
:mod:`lolcoach.features.waves`       Minion spawn model and wave-state
                                     inference (crash / freeze / slow push).
:mod:`lolcoach.features.lane`        Trades, lane priority, recall value.
:mod:`lolcoach.features.jungle`      Path reconstruction and a Markov model
                                     for jungle prediction.
:mod:`lolcoach.features.objectives`  Spawn timers and objective control.
:mod:`lolcoach.features.macro`       Death causes, fights, positioning,
                                     rotations.
:mod:`lolcoach.features.pipeline`    Runs all of the above per match and
                                     writes the feature tables.
============================  ===============================================

Riot samples participant frames roughly once per minute, so every inference
carries a confidence and the dataset stage drops low-confidence rows rather
than turning a weak estimate into confident advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..logging_utils import get_logger
from ..storage import Database
from .pipeline import build_features

log = get_logger("features")

#: Tables written by the feature pass, in dependency order.
FEATURE_TABLES = ("wave_states", "lane_states", "recalls", "objective_events",
                  "jungle_paths", "rotations", "fights", "deaths")


@dataclass
class FeatureResult:
    """Row counts produced by one derivation pass."""

    matches: int = 0
    waves: int = 0
    lanes: int = 0
    recalls: int = 0
    objectives: int = 0
    jungle_steps: int = 0
    rotations: int = 0
    fights: int = 0
    deaths: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, other: "FeatureResult") -> None:
        for key in ("matches", "waves", "lanes", "recalls", "objectives",
                    "jungle_steps", "rotations", "fights", "deaths"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.errors.extend(other.errors)

    @property
    def total_rows(self) -> int:
        return (self.waves + self.lanes + self.recalls + self.objectives
                + self.jungle_steps + self.rotations + self.fights + self.deaths)


def derive_all(cfg: Config, *, limit: int | None = None,
               force: bool = False) -> FeatureResult:
    """Derive features for every match that has a timeline.

    Matches whose features already exist are skipped unless ``force`` is set,
    which makes this cheap to re-run after each incremental download.
    """
    db = Database(cfg.paths.db)
    counts = build_features(db, limit=limit, force=force)
    db.set_meta("last_feature_derivation", {
        "matches": counts.get("matches", 0),
        "failed": counts.get("failed", 0),
    })
    db.optimize()
    db.close()

    result = FeatureResult(
        matches=counts.get("matches", 0),
        waves=counts.get("wave_states", 0),
        lanes=counts.get("lane_states", 0),
        recalls=counts.get("recalls", 0),
        objectives=counts.get("objective_events", 0),
        jungle_steps=counts.get("jungle_paths", 0),
        rotations=counts.get("rotations", 0),
        fights=counts.get("fights", 0),
        deaths=counts.get("deaths", 0),
    )
    failed = counts.get("failed", 0)
    if failed:
        result.errors.append(f"{failed} matches failed feature derivation")
    log.info("derived %d feature rows across %d matches",
             result.total_rows, result.matches)
    return result


def derive_match(db: Database, match_id: str) -> FeatureResult:
    """Derive features for a single match (used by tests and ad-hoc analysis)."""
    from .jungle import JunglePredictor
    from .pipeline import process_match

    predictor = JunglePredictor.from_json(db.get_meta("jungle_model"))
    item_costs = {int(r[0]): int(r[1] or 0)
                  for r in db.query("SELECT item_id, total_gold FROM items")}
    counts = process_match(db, match_id, predictor, item_costs)
    db.set_meta("jungle_model", predictor.to_json())
    return FeatureResult(
        matches=1 if counts else 0,
        waves=counts.get("wave_states", 0), lanes=counts.get("lane_states", 0),
        recalls=counts.get("recalls", 0), objectives=counts.get("objective_events", 0),
        jungle_steps=counts.get("jungle_paths", 0),
        rotations=counts.get("rotations", 0), fights=counts.get("fights", 0),
        deaths=counts.get("deaths", 0))
