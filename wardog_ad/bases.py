"""Faction HQ reference points in the same coordinates as the navigation assets."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path


@dataclass(frozen=True)
class TeamBase:
    id: str
    name: str
    color: str
    point: tuple[float, float]
    sigma_m: float
    source: str
    custom: bool = False


@dataclass(frozen=True)
class ManualPosition:
    map_id: str
    point: tuple[float, float]
    timestamp: float
    source: str
    sigma_m: float


@lru_cache(maxsize=1)
def _registry():
    return json.loads((Path(__file__).parent / 'data' / 'team_bases.json').read_text(encoding='utf-8'))


def team_bases(assets, cfg):
    entry = _registry()['maps'].get(assets.map_id)
    if entry is None:
        return []
    overrides = cfg.base_overrides.get(assets.map_id, {})
    result = []
    for row in entry['bases']:
        custom = row['id'] in overrides
        point = tuple(overrides[row['id']]) if custom else assets.game_to_map(row['game_point'])
        if not assets.contains(point):
            raise ValueError(f'invalid base coordinates: {assets.map_id}/{row["id"]}')
        result.append(TeamBase(row['id'], row['name'], row['color'], point, row['sigma_m'],
                               'user calibration' if custom else entry['source'], custom))
    return result
