"""Summarize recorded control gates; does not interact with the game."""
import json
from pathlib import Path

source = max(Path('logs').glob('flight_*.jsonl'),key=lambda p:p.stat().st_mtime)
rows = []
for line in source.read_text(encoding='utf-8').splitlines():
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
fields = ('profile','engaged','mission','status','message')
changes,last = [],None
for row in rows:
    key = tuple(row.get(f) for f in fields)
    if key != last:
        changes.append((round(row['t']-rows[0]['t'],1),key))
        last = key
print(source,'rows',len(rows))
for change in changes[-35:]:
    print(change)
cfg = json.loads(Path('config.json').read_text(encoding='utf-8'))
print('config',{k:cfg.get(k) for k in ('profile','map_id','auto_map','destinations','truck_minimap','keys')})
