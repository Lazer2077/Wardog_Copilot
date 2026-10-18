"""Offline, low-resolution behavior cloning. Never sends keyboard events.

python -m wardog_ad.imitation --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

KEYS = ('W', 'A', 'S', 'D', 'SPACE')
VIEWS = {'first_person': 0, 'third_person': 1}
FEATURES = ['speed/30', 'speed_valid', 'goal_right/50', 'goal_forward/50',
            'goal_valid', 'first_person', 'third_person']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def features(row):
    """Missing navigation stays masked; never fabricate a straight command."""
    f = np.zeros(7, np.float32)
    f[5 + VIEWS[row['camera_view']]] = 1
    nav = row.get('navigation') or {}
    fresh = row.get('label_validity', {}).get('navigation_fresh', False)
    # Navigation must have been available when this image was acquired.
    causal = nav.get('processed_t', float('inf')) <= row['t']
    if fresh and causal:
        speed = nav.get('speed') or {}
        v = speed.get('signed_ms')
        if v is not None and np.isfinite(v):
            f[0:2] = np.clip(v / 30, -2, 2), 1
        target = nav.get('local_target_m')
        if isinstance(target, (tuple, list)) and len(target) == 2 and np.isfinite(target).all():
            f[2:5] = np.clip(target[0] / 50, -2, 2), np.clip(-target[1] / 50, -2, 2), 1
    return f


def eligible(row, vehicle):
    keys = set(row.get('observed_keys', []))
    validity = row.get('label_validity', {})
    return (row.get('camera_view') in VIEWS and validity.get('camera_confirmed', False)
            and validity.get('input_fresh', False) and 0 <= row.get('keys_age_s', 99) <= .1
            and row.get('keys_t', float('inf')) <= row['t']
            and bool(vehicle) and not keys.intersection({'ALT', 'TAB', 'ESC', 'M', 'F0', 'C', 'R'})
            and not {'W', 'S'} <= keys and not {'A', 'D'} <= keys)


def make_samples(rows, good, history=4, step=.2, tolerance=.13):
    """Choose history by real timestamps, with hard breaks at excluded frames/gaps."""
    samples, segment = [], []
    def finish():
        if not segment:
            return
        times = np.array([rows[i]['t'] for i in segment])
        for pos, i in enumerate(segment):
            targets = times[pos] - np.arange(history - 1, -1, -1) * step
            chosen = []
            for target in targets:
                k = int(np.searchsorted(times[:pos + 1], target))
                choices = [j for j in (k - 1, k) if 0 <= j <= pos]
                if not choices:
                    break
                j = min(choices, key=lambda j: abs(times[j] - target))
                if abs(times[j] - target) > tolerance:
                    break
                chosen.append(segment[j])
            if len(chosen) == history and len(set(chosen)) == history:
                samples.append(chosen)
    for i, row in enumerate(rows):
        boundary = segment and (not 0 < row['t'] - rows[segment[-1]]['t'] <= .45
                                or row['camera_view'] != rows[segment[-1]]['camera_view'])
        if not good[i] or boundary:
            finish()
            segment = []
        if good[i]:
            segment.append(i)
    finish()
    return np.asarray(samples, dtype=np.int64).reshape(-1, history)


def split_samples(samples, times, guard=2.):
    """Contiguous time holdouts; every history frame belongs to its split."""
    lo, hi = times[samples[0, -1]], times[samples[-1, -1]]
    a, b = lo + .65 * (hi - lo), lo + .82 * (hi - lo)
    start, end = times[samples[:, 0]], times[samples[:, -1]]
    return {'train': np.flatnonzero(end < a - guard),
            'val': np.flatnonzero((start > a + guard) & (end < b - guard)),
            'test': np.flatnonzero(start > b + guard)}


def prepare(args):
    cv2.setNumThreads(1)
    source, out = Path(args.session).resolve(), Path(args.output).resolve()
    if out.exists():
        raise ValueError('Output already exists; choose a new directory to preserve provenance')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('Session is incomplete; finish collection before preparing')
    raw = (source / 'frames.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    vehicle = args.vehicle or manifest.get('metadata', {}).get('vehicle')
    if not vehicle or vehicle == 'unknown':
        raise ValueError('Supply --vehicle using a human-confirmed vehicle label')
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in source.glob('*.avi')}
    out.mkdir(parents=True)
    height = round(args.width * rows[0]['height'] / rows[0]['width'])
    reuse = getattr(args, 'reuse_cache', None)
    if reuse:
        cached = json.loads((Path(reuse) / 'dataset.json').read_text(encoding='utf-8'))
        if (cached['frames_sha256'] != hashlib.sha256(raw).hexdigest()
                or cached['width'] != args.width or cached['height'] != height
                or cached['decoded'] != len(rows)
                or cached['source_videos'] != {p.name:list(s) for p,s in before.items()}):
            raise ValueError('Reuse cache does not match this source')
        shutil.copyfile(Path(reuse) / 'images.npy', out / 'images.npy')
        frames = np.load(out / 'images.npy', mmap_mode='r')
    else:
        frames = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8,
                                         shape=(len(rows), height, args.width, 3))
    decoded = np.zeros(len(rows), bool)
    if reuse:
        decoded[:] = True
    for name in ([] if reuse else sorted({r['video'] for r in rows})):
        lookup = {r['video_frame']: i for i, r in enumerate(rows) if r['video'] == name}
        cap = cv2.VideoCapture(str(source / name))
        frame_no = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame_no in lookup:
                i = lookup[frame_no]
                frames[i] = cv2.cvtColor(cv2.resize(image, (args.width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                decoded[i] = True
            frame_no += 1
        cap.release()
    frames.flush()
    if any((p.stat().st_size, p.stat().st_mtime_ns) != stat for p, stat in before.items()):
        raise ValueError('Source videos changed during preparation; discard output and retry')
    good = decoded & np.array([eligible(r, vehicle) for r in rows])
    event_path = source / 'events.jsonl'
    if event_path.exists():
        events = [json.loads(x) for x in event_path.read_text(encoding='utf-8').splitlines()]
        # Vehicle overrides must not accidentally override the original transition gate.
        for event in events:
            if event['type'] in {'camera_label', 'focus', 'pause', 'pause_for_ui_or_seat_change'}:
                good &= np.array([not event['t'] - .25 <= r['t'] <= event['t'] + .75 for r in rows])
    review = None
    if getattr(args, 'review', None):
        review = json.loads(Path(args.review).read_text(encoding='utf-8'))
        if review['frames_sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError('Review belongs to another recording')
        for interval in review['exclude']:
            good &= np.array([not interval['start_s'] <= r['t'] <= interval['end_s'] for r in rows])
    samples = make_samples(rows, good)
    if len(samples) < 30:
        raise ValueError('Too few valid temporal samples')
    times = np.array([r['t'] for r in rows])
    splits = split_samples(samples, times)
    if any(len(v) < 10 for v in splits.values()):
        raise ValueError('Temporal split too small; collect more continuous driving')
    labels = np.array([[k in r.get('observed_keys', []) for k in KEYS] for r in rows], dtype=np.float32)
    feat = np.stack([features(r) if r.get('camera_view') in VIEWS else np.zeros(7, np.float32) for r in rows])
    np.savez(out / 'labels.npz', actions=labels, features=feat, samples=samples, times=times, **splits)
    report = dict(schema_version=1, source=str(source), vehicle=vehicle,
                  vehicle_label_source='human CLI confirmation' if args.vehicle else 'manifest',
                  frames_sha256=hashlib.sha256(raw).hexdigest(), source_videos={p.name:list(s) for p,s in before.items()},
                  events_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest() if event_path.exists() else None,
                  review=review,
                  width=args.width, height=height, history=4, history_step_s=.2, max_gap_s=.45,
                  keys=KEYS, features=FEATURES, decoded=int(decoded.sum()), indexed=len(rows),
                  eligible_frames=int(good.sum()), samples=len(samples),
                  splits={k:len(v) for k,v in splits.items()}, guard_s=2,
                  camera_counts=dict(Counter(rows[s[-1]]['camera_view'] for s in samples)),
                  navigation_speed_samples=int(feat[samples[:, -1], 1].sum()),
                  navigation_goal_samples=int(feat[samples[:, -1], 4].sum()),
                  limitation='Single-session temporal holdout; does not measure unseen map/vehicle generalization. Human data has not been certified collision-free.')
    write_json(out / 'dataset.json', report)
    # A contact sheet makes black frames, menus and view labels reviewable.
    selected = samples[np.linspace(0, len(samples)-1, 24, dtype=int), -1]
    sheet = np.zeros((6 * (height+24), 4 * args.width, 3), np.uint8)
    for j, i in enumerate(selected):
        y, x = (j//4)*(height+24), (j%4)*args.width
        sheet[y:y+height, x:x+args.width] = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        cv2.putText(sheet, f'{times[i]:.1f}s {"".join(k for k,v in zip(KEYS,labels[i]) if v)}', (x+3,y+height+17), cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    cv2.imwrite(str(out / 'review.jpg'), sheet)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_model():
    import torch
    from torch import nn
    class Driver(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Conv2d(3, 16, 5, 2), nn.ReLU(),
                nn.Conv2d(16, 24, 3, 2), nn.ReLU(), nn.Conv2d(24, 32, 3, 2), nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(), nn.Linear(480, 64), nn.ReLU())
            self.memory = nn.GRU(71, 64, batch_first=True)
            self.head = nn.Linear(64, 5)
        def forward(self, images, features):
            b,t,c,h,w = images.shape
            vision = self.encoder(images.reshape(b*t,c,h,w)).reshape(b,t,64)
            state,_ = self.memory(torch.cat((vision, features), dim=-1))
            return self.head(state[:, -1])
    return Driver()


def metrics(truth, probabilities):
    pred = probabilities >= .5
    truth = truth.astype(bool)
    tp = (pred & truth).sum(0)
    fp = (pred & ~truth).sum(0)
    fn = (~pred & truth).sum(0)
    f1 = 2 * tp / np.maximum(2*tp + fp + fn, 1)
    supported = truth.sum(0) > 0
    return dict(exact_match=float(np.all(pred == truth, axis=1).mean()),
                macro_f1_supported=float(f1[supported].mean()) if supported.any() else 0.,
                per_key={k:dict(f1=float(f1[i]), support=int(truth[:,i].sum()),
                               precision=float(tp[i]/max(tp[i]+fp[i],1)),
                               recall=float(tp[i]/max(tp[i]+fn[i],1))) for i,k in enumerate(KEYS)},
                conflicting_axes=float(((pred[:,0]&pred[:,2]) | (pred[:,1]&pred[:,3])).mean()))


def train(args):
    import random
    import torch
    from torch.utils.data import Dataset, DataLoader
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root, out = Path(args.dataset), Path(args.output)
    if out.exists():
        raise ValueError('Run output exists; choose a new directory')
    meta = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    data = np.load(root / 'labels.npz')
    images = np.load(root / 'images.npy', mmap_mode='r')
    class Sequences(Dataset):
        def __init__(self, split):
            self.indices = data[split]
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, n):
            sequence = data['samples'][self.indices[n]]
            rgb = np.array(images[sequence], dtype=np.float32).transpose(0,3,1,2) / 255.
            return torch.from_numpy(rgb), torch.from_numpy(data['features'][sequence]), torch.from_numpy(data['actions'][sequence[-1]])
    device = torch.device(args.device)
    model = build_model().to(device)
    loaders = {k:DataLoader(Sequences(k), batch_size=args.batch_size, shuffle=k=='train', num_workers=0) for k in ['train','val','test']}
    targets = data['actions'][data['samples'][data['train'], -1]]
    counts = targets.sum(0)
    weights = np.clip((len(targets)-counts)/np.maximum(counts,1), 1, 8)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001)
    out.mkdir(parents=True)
    run_config = dict(epochs=args.epochs, batch_size=args.batch_size, threads=args.threads,
                      device=str(device), seed=args.seed, lr=args.lr, torch_version=str(torch.__version__),
                      positive_class_weights=weights.tolist(), train_positive_counts=counts.tolist(),
                      unsupported_keys=[k for k,n in zip(KEYS,counts) if n == 0])
    write_json(out / 'run.json', run_config)
    best, history = -1., []
    def evaluate(split):
        model.eval()
        predictions, truth = [], []
        with torch.inference_mode():
            for x,f,y in loaders[split]:
                predictions.append(model(x.to(device),f.to(device)).sigmoid().cpu().numpy())
                truth.append(y.numpy())
        p,y = np.concatenate(predictions), np.concatenate(truth)
        return metrics(y,p),p,y
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for x,f,y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device), f.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(x)
        score,_,_ = evaluate('val')
        history.append(dict(epoch=epoch+1, train_loss=total/len(loaders['train'].dataset), validation=score))
        print(json.dumps(history[-1]), flush=True)
        if score['macro_f1_supported'] > best:
            best = score['macro_f1_supported']
            torch.save(dict(model=model.state_dict(), architecture='tiny_cnn_gru_v1', dataset=meta,
                            seed=args.seed, threshold=.5, epoch=epoch+1, run=run_config), out / 'best.pt')
        write_json(out / 'history.json', history)
    ckpt = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    score,p,y = evaluate('test')
    sample_ids = data['samples'][data['test']]
    # Past human controls are an offline baseline ONLY, never model inputs.
    persistence = data['actions'][sample_ids[:, -2]]
    majority = np.broadcast_to((targets.mean(0)>=.5).astype(float), y.shape)
    result = dict(best_epoch=ckpt['epoch'], test=score,
                  persistence=metrics(y,persistence), majority=metrics(y,majority),
                  device=str(device), dataset=meta,
                  deployable=False, note='Offline imitation experiment; no closed-loop validation, no automatic key output. Unsupported actions and views are untrained.')
    write_json(out / 'evaluation.json', result)
    np.savez(out / 'test_predictions.npz', probabilities=p, truth=y, frame_indices=sample_ids[:,-1])
    print(json.dumps(result, ensure_ascii=False, indent=2))


def predict(args):
    """Replay an already prepared recording; write probabilities, never drive."""
    import torch
    torch.set_num_threads(args.threads)
    root = Path(args.dataset)
    meta = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    trained = checkpoint['dataset']
    if checkpoint['architecture'] != 'tiny_cnn_gru_v1':
        raise ValueError('Unsupported architecture')
    if meta['vehicle'] != trained['vehicle']:
        raise ValueError('Vehicle differs from training data')
    if (meta['width'],meta['height'],meta['history']) != (trained['width'],trained['height'],trained['history']):
        raise ValueError('Image geometry/history differs from training')
    if set(meta['camera_counts']) - set(trained['camera_counts']):
        raise ValueError('Recording includes a camera view never seen in training')
    model = build_model().eval()
    model.load_state_dict(checkpoint['model'])
    images = np.load(root / 'images.npy', mmap_mode='r')
    data = np.load(root / 'labels.npz')
    samples = data['samples'][data[args.split]]
    with Path(args.output).open('x', encoding='utf-8') as stream, torch.inference_mode():
        for begin in range(0,len(samples),32):
            batch = samples[begin:begin+32]
            x = torch.from_numpy(np.array(images[batch], dtype=np.float32).transpose(0,1,4,2,3)/255.)
            f = torch.from_numpy(data['features'][batch])
            probabilities = model(x,f).sigmoid().numpy()
            for sequence,p in zip(batch,probabilities):
                stream.write(json.dumps(dict(frame_index=int(sequence[-1]),t=float(data['times'][sequence[-1]]),
                                            probabilities=dict(zip(KEYS,map(float,p)))), ensure_ascii=False)+'\n')
    print(f'Wrote {len(samples)} offline predictions to {args.output}; no key output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('session')
    p.add_argument('--output', required=True)
    p.add_argument('--vehicle')
    p.add_argument('--width', type=int, default=160)
    p.add_argument('--review', help='JSON exclusions tied to frames SHA256')
    p.add_argument('--reuse-cache', help='Reuse verified full decoded image cache')
    p.set_defaults(run=prepare)
    p = sub.add_parser('train')
    p.add_argument('dataset')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=.001)
    p.set_defaults(run=train)
    p = sub.add_parser('predict')
    p.add_argument('dataset')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train','val','test'], default='test')
    p.add_argument('--threads', type=int, default=4)
    p.set_defaults(run=predict)
    args = parser.parse_args()
    if getattr(args, 'width', 160) < 64:
        parser.error('width must be at least 64')
    args.run(args)


if __name__ == '__main__':
    main()
