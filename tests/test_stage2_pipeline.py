import csv
import json

from fixtures import write_synthetic

from lolcoach.config import Config
from lolcoach.dataset import prepare_dataset
from lolcoach.features import derive_all
from lolcoach.ingest import ingest_all
from lolcoach.storage import Database


def _write_oracle_fixture(cfg: Config) -> None:
    fields = [
        "gameid", "position", "participantid", "side", "champion", "result",
        "kills", "deaths", "assists", "totalgold", "total cs", "damagetochampions",
        "visionscore", "goldat10", "xpat10", "csat10", "gamelength", "league", "patch",
    ]
    directory = cfg.paths.raw / "oracles_elixir"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for side, result in (("Blue", "1"), ("Red", "0")):
            for number, position in enumerate(("top", "jng", "mid", "bot", "sup"), 1):
                writer.writerow({
                    "gameid": "fixture-pro-1", "position": position,
                    "participantid": str(number if side == "Blue" else number + 5),
                    "side": side, "champion": "Ahri", "result": result,
                    "kills": "1", "deaths": "1", "assists": "2", "totalgold": "7000",
                    "total cs": "100", "damagetochampions": "5000", "visionscore": "20",
                    "goldat10": "3500", "xpat10": "4000", "csat10": "75",
                    "gamelength": "1800", "league": "Fixture League", "patch": "16.16",
                })


def test_stage2_ingestion_features_and_dataset_are_resumable(tmp_path):
    cfg = Config(data_root=str(tmp_path))
    write_synthetic(cfg.paths.source("riot"))
    _write_oracle_fixture(cfg)

    first = ingest_all(cfg)
    assert [result.matches for result in first] == [1, 1]
    assert [result.participants for result in first] == [10, 10]
    assert [result.frames for result in first] == [260, 10]

    derived = derive_all(cfg, force=True)
    assert derived.errors == []
    assert derived.waves > 0
    assert derived.lanes > 0
    assert derived.recalls > 0
    assert derived.objectives == 3
    assert derived.jungle_steps > 0
    assert derived.deaths == 3

    db = Database(cfg.paths.db)
    before = db.counts()
    db.close()
    assert before["matches"] == 2
    assert before["participants"] == 20
    assert before["frames"] == 270
    assert before["events"] == 22

    second = ingest_all(cfg)
    assert [result.matches for result in second] == [0, 0]
    assert [result.skipped for result in second] == [1, 1]
    db = Database(cfg.paths.db)
    assert db.counts() == before
    db.close()

    dataset = prepare_dataset(cfg, limit=25, refresh_features=False)
    assert dataset.examples == 25
    records = [json.loads(line) for line in dataset.paths["train"].read_text(encoding="utf-8").splitlines()]
    assert records
    assert all(record["messages"][-1]["role"] == "assistant" for record in records)