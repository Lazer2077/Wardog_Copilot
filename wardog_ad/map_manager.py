"""Conservative cross-map identification from minimap pixels only."""
from dataclasses import dataclass
import logging
import math

from .assets import available_maps, load_map
from .localization import Fix, MinimapLocalizer

log = logging.getLogger('wardog_ad')


@dataclass
class MapSelection:
    map_id: str | None
    fix: Fix
    ready: bool
    reason: str


class MapManager:
    def __init__(self, assets, locator, tuning, *, candidates=None):
        self.t = tuning
        self.assets = {assets.map_id: assets}
        self.locators = {assets.map_id: locator}
        self.active = assets.map_id
        self.ready = False
        self.pending = None
        self.count = 0
        self.previous = None
        self.next_scan = -math.inf
        self.reason = 'identifying map'
        self.candidates = candidates
        self.loaded = False
        self.diagnostics = {}
        self.diagnostics_at = None

    def _load(self):
        if self.loaded:
            return
        if self.candidates is not None:
            for assets, locator in self.candidates:
                self.assets[assets.map_id] = assets
                self.locators[assets.map_id] = locator
        else:
            root = self.assets[self.active].root
            for name in available_maps(root):
                if name in self.assets:
                    continue
                try:
                    assets = load_map(name, root)
                    locator = MinimapLocalizer.from_assets(assets)
                    # Fail before selecting a map with unusable road assets.
                    assets.roads
                    self.assets[name], self.locators[name] = assets, locator
                except Exception:
                    log.exception('map candidate unavailable: %s', name)
        self.loaded = True

    def _quality(self, fix):
        return (fix.valid and fix.confidence >= self.t.min_confidence
                and fix.inliers >= self.t.min_inliers and 0 <= fix.error <= 2.0
                and .025 < fix.scale < 6)

    def _reset(self, reason):
        self.ready = False
        self.pending, self.count, self.previous = None, 0, None
        self.reason = reason

    def update(self, image, now, anchor=(.5, .5), north_up=True):
        self._load()
        if image is None:
            self.diagnostics = {}
            self.diagnostics_at = now
            self._reset('no minimap; map unconfirmed')
            return MapSelection(None, Fix(reason=self.reason), False, self.reason)

        fix = self.locators[self.active].locate(image, anchor, north_up)
        if self.ready and not self._quality(fix):
            self._reset('map localization lost; identifying again')
            self.next_scan = now
        if now < self.next_scan:
            return MapSelection(self.active if self.ready else None, fix, self.ready, self.reason)
        self.next_scan = now + max(.1, self.t.scan_interval_s)

        ranked = []
        self.diagnostics_at = now
        for name, locator in self.locators.items():
            # Full-map verification prevents an old local tracker from biasing identity.
            locator.reset()
            candidate = locator.locate(image, anchor, north_up)
            score = (candidate.confidence * math.sqrt(max(0,candidate.inliers)) / (1+candidate.error)
                     if self._quality(candidate) else 0.)
            self.diagnostics[name] = dict(valid=candidate.valid, confidence=candidate.confidence,
                inliers=candidate.inliers, error=candidate.error, scale=candidate.scale,
                rotation=candidate.rotation, score=score, reason=candidate.reason)
            if self._quality(candidate):
                ranked.append((score, name, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] < ranked[1][0] * self.t.score_ratio):
            self._reset('unknown or ambiguous map')
            return MapSelection(None, Fix(reason=self.reason), False, self.reason)

        _, name, best = ranked[0]
        if self.ready and name == self.active:
            return MapSelection(name, best, True, self.reason)
        self.ready = False
        consistent = self.pending == name and self.previous is not None
        if consistent:
            old_time, old = self.previous
            moved = math.hypot(best.x - old.x, best.y - old.y) * self.assets[name].meters_per_pixel
            consistent = (moved <= 50 + 60 * max(0, now - old_time)
                          and .7 <= best.scale / old.scale <= 1.4)
        self.count = self.count + 1 if consistent else 1
        self.pending, self.previous = name, (now, best)
        self.reason = f'identifying {name} ({self.count}/{self.t.confirm_scans})'
        if self.count >= max(1, self.t.confirm_scans):
            self.active, self.ready = name, True
            self.reason = f'map confirmed: {name}'
        return MapSelection(name if self.ready else None, best, self.ready, self.reason)
