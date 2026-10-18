"""Infer faction from several small friendly markers, never from zone tint/CONTROL."""
from dataclasses import dataclass

import cv2
import numpy as np

TEAM_NAMES = {'valkyra': '红队 Valkyra', 'manticore': '绿队 Manticore', 'lonestar': '蓝队 Lonestar'}
HUES = {'valkyra': ((0, 12), (165, 179)), 'manticore': ((35, 85),), 'lonestar': ((90, 125),)}


@dataclass(frozen=True)
class TeamReading:
    faction_id: str | None = None
    confidence: float = 0.
    timestamp: float = 0.
    source: str = 'minimap_markers'
    reason: str = 'waiting for friendly markers'
    markers: int = 0


def detect_team(image, now):
    if image is None or min(image.shape[:2]) < 60:
        return TeamReading(timestamp=now, reason='no minimap')
    # Normalize icon sizes across resolutions. Scoreboards / CONTROL text below
    # the terrain are deliberately excluded, as are large coloured zone borders.
    h,w = image.shape[:2]
    image = cv2.resize(image, (300, max(1, round(h*300/w))))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height,width = hsv.shape[:2]
    votes = {}
    for faction,ranges in HUES.items():
        mask = np.zeros((height,width),np.uint8)
        for low,high in ranges:
            mask |= cv2.inRange(hsv,(low,95,125),(high,255,255))
        mask[:5] = 0
        mask[round(height*.9):] = 0
        mask[:,:5] = mask[:,-5:] = 0
        _,_,stats,_ = cv2.connectedComponentsWithStats(mask,8)
        votes[faction] = sum(1 for x,y,bw,bh,area in stats[1:]
            if 5 <= area <= 200 and 3 <= bw <= 22 and 3 <= bh <= 22
            and .5 <= bw/bh <= 2 and .25 <= area/(bw*bh) <= .85)
    ranked = sorted(votes, key=votes.get, reverse=True)
    winner = ranked[0]
    n,total = votes[winner],sum(votes.values())
    if n < 3 or n < 2*votes[ranked[1]] or n/max(1,total) < .75:
        return TeamReading(timestamp=now, reason='too few or conflicting friendly markers', markers=n)
    return TeamReading(winner, n/total*min(1,n/5), now, markers=n, reason='candidate')


class TeamDetector:
    def __init__(self):
        self.context = None
        self.pending = None
        self.count = 0
        self.first = self.last_sample = -1e9
        self.reading = TeamReading()

    def update(self, image, now, context):
        if context != self.context:
            self.__init__()
            self.context = context
        if now-self.last_sample < .15:
            return self.reading
        self.last_sample = now
        candidate = detect_team(image,now)
        if candidate.faction_id is None:
            self.pending,self.count = None,0
        else:
            if candidate.faction_id != self.pending:
                self.pending,self.count,self.first = candidate.faction_id,0,now
            self.count += 1
            if self.count >= 3 and now-self.first >= .3:
                self.reading = TeamReading(candidate.faction_id,candidate.confidence,now,
                    reason='confirmed across frames',markers=candidate.markers)
        if now-self.reading.timestamp > 3.:
            self.reading = TeamReading(timestamp=now, reason='waiting for stable friendly markers')
        return self.reading
