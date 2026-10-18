"""Map assets (base image, SIFT features, road network, game-coordinate transform).

The data is read from a local checkout of greatredo/wardogs-navigator, which
publishes the map imagery, pre-computed SIFT features and the hand-curated road
network.  Nothing is copied into this project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ASSETS = Path(__file__).resolve().parents[1] / 'third_party' / 'wardogs-navigator' / 'assets'


@dataclass
class MapAssets:
    map_id: str
    name: str
    width: int
    height: int
    root: Path
    info: dict
    _project: dict | None = field(default=None, repr=False)

    @property
    def image_path(self) -> Path:
        return self.root / self.info['image']

    def load_image(self) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(str(self.image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'cannot read map image {self.image_path}')
        return image

    def load_features(self):
        with np.load(self.root / self.info['features'], allow_pickle=False) as data:
            return np.float32(data['points']), np.float32(data['descriptors'])

    def project(self) -> dict:
        if self._project is None:
            self._project = json.loads((self.root / self.info['project']).read_text(encoding='utf-8'))
        return self._project

    @property
    def roads(self) -> list:
        return self.project()['roads']

    @property
    def meters_per_pixel(self) -> float:
        return float(self.project().get('meters_per_pixel', 5.0))

    def _game_transform(self):
        path = self.root / 'game-coordinates.json'
        if not path.exists():
            return None, None
        data = json.loads(path.read_text(encoding='utf-8'))['maps'].get(self.map_id)
        if not data:
            return None, None
        return np.asarray(data['game_to_map'], dtype=float), data['bounds']

    def game_to_map(self, point) -> tuple[float, float]:
        matrix, bounds = self._game_transform()
        if matrix is None:
            raise ValueError(f'no game coordinate transform for {self.map_id}')
        x, y = map(float, point)
        if not (bounds['minX'] <= x <= bounds['maxX'] and bounds['minY'] <= y <= bounds['maxY']):
            raise ValueError(f'game coordinate {point} outside map bounds {bounds}')
        mx, my = matrix @ [x, y, 1.0]
        return float(mx), float(my)

    def map_to_game(self, point) -> tuple[float, float]:
        matrix, _ = self._game_transform()
        if matrix is None:
            raise ValueError(f'no game coordinate transform for {self.map_id}')
        gx, gy = np.linalg.solve(matrix[:, :2], np.asarray(point, dtype=float) - matrix[:, 2])
        return float(gx), float(gy)

    def contains(self, point) -> bool:
        return 0 <= point[0] < self.width and 0 <= point[1] < self.height and all(map(math.isfinite, point))


def available_maps(root: Path | str = DEFAULT_ASSETS) -> list[str]:
    root = Path(root)
    return [m['id'] for m in json.loads((root / 'maps.json').read_text(encoding='utf-8'))]


def load_map(map_id: str, root: Path | str = DEFAULT_ASSETS) -> MapAssets:
    root = Path(root)
    registry = root / 'maps.json'
    if not registry.exists():
        raise FileNotFoundError(
            f'{registry} not found. Clone the assets first:\n'
            '  git clone --depth 1 https://github.com/greatredo/wardogs-navigator third_party/wardogs-navigator')
    for info in json.loads(registry.read_text(encoding='utf-8')):
        if info['id'] == map_id:
            return MapAssets(map_id, info['name'], int(info['width']), int(info['height']), root, info)
    raise ValueError(f'unknown map {map_id!r}; available: {available_maps(root)}')
