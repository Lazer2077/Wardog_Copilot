"""Offline TCP checkpoint probe. Never imports or calls the game's input APIs.

Requires the upstream OpenDriveLab/TCP checkout in third_party/TCP and PyTorch.
The public checkpoint used here is a community reproduction, not author weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wardog_ad.tcp_adapter import decode_beta, TruckOutputMapper
SOURCE = 'https://github.com/OpenDriveLab/TCP'
WEIGHTS = 'https://huggingface.co/maxim-igenbergs/tcp-carla-repro'


def load_model(checkpoint, repo):
    sys.path.insert(0, str(repo.resolve()))
    from TCP.config import GlobalConfig
    import TCP.model as upstream
    # All parameters are loaded strictly below. Avoid an extra ImageNet download.
    original = upstream.resnet34
    upstream.resnet34 = lambda **kwargs: original(pretrained=False)
    try:
        model = upstream.TCP(GlobalConfig())
    finally:
        upstream.resnet34 = original
    # Lightning stores a callback class as a dict key. A local inert placeholder
    # permits reading that metadata without importing/executing Lightning code.
    placeholder = type('ModelCheckpoint', (),
                       {'__module__': 'pytorch_lightning.callbacks.model_checkpoint'})
    with torch.serialization.safe_globals([placeholder]):
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = loaded.get('state_dict', loaded)
    state = {key.removeprefix('model.'): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def prepare(image, crop):
    if crop == 'windshield':
        w, h = image.size
        image = image.crop((int(w * .29), int(h * .27), int(w * .84), int(h * .67)))
    image = image.resize((900, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255
    arr = (arr - [.485, .456, .406]) / [.229, .224, .225]
    return image, torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)


def predict(model, tensor, speed, command, target):
    point = torch.tensor([target], dtype=torch.float32)
    state = torch.zeros(1, 9)
    state[0, 0] = speed / 12
    state[0, 1:3] = point[0]
    state[0, 3 + command - 1] = 1
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(tensor, state, point)
        action = model._get_action_beta(prediction['mu_branches'], prediction['sigma_branches'])[0]
    elapsed = (time.perf_counter() - start) * 1000
    acceleration, steering = action.tolist()
    alpha = prediction['mu_branches'][0].tolist()
    beta = prediction['sigma_branches'][0].tolist()
    decoded = decode_beta(alpha, beta)
    if not np.allclose([decoded['acceleration'], decoded['steer']], [acceleration, steering], atol=1e-6):
        raise RuntimeError('TCP action decoder differs from upstream')
    mean_diagnostic = decode_beta(alpha, beta, method='mean')
    # Match a fresh agent's trajectory PID and status=0 blend. Reset between
    # independent static scenarios so their PID histories cannot contaminate.
    from TCP.model import PIDController
    cfg = model.config
    model.turn_controller = PIDController(cfg.turn_KP, cfg.turn_KI, cfg.turn_KD, cfg.turn_n)
    model.speed_controller = PIDController(cfg.speed_KP, cfg.speed_KI, cfg.speed_KD, cfg.speed_n)
    ts, tt, tb, metadata = model.control_pid(prediction['pred_wp'].clone(),
                                           torch.tensor([speed]), point.clone())
    throttle, brake = max(0, acceleration), max(0, -acceleration)
    hybrid = dict(steer=float(np.clip(.3 * steering + .7 * ts, -1, 1)),
                  throttle=float(np.clip(.3 * throttle + .7 * tt, 0, .75)),
                  brake=float(np.clip(.3 * brake + .7 * tb, 0, 1)))
    if hybrid['brake'] > .5:
        hybrid['throttle'] = 0.
    mapper = TruckOutputMapper()
    # Dry run only. These axes never reach a keyboard; speed/navigation are assumed.
    mapped = mapper.map(hybrid, speed_ms=speed, now=0., prediction_at=0., speed_at=0., enabled=True)
    return dict(latency_ms=round(elapsed, 2), steer=steering,
                throttle=throttle, brake=brake,
                trajectory_control=dict(steer=float(ts), throttle=float(tt), brake=float(tb),
                                        desired_speed_ms=float(metadata['desired_speed'])),
                startup_hybrid=hybrid,
                beta_mode=decoded, beta_mean_diagnostic=mean_diagnostic,
                mapped_startup_dry_run=mapped,
                waypoints=prediction['pred_wp'][0].tolist(),
                # This is the auxiliary speed estimate, NOT a commanded speed.
                auxiliary_pred_speed=prediction['pred_speed'][0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/tcp/best_model.ckpt')
    parser.add_argument('--repo', type=Path, default=ROOT / 'third_party/TCP')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/tcp_probe')
    parser.add_argument('--last-only', action='store_true', help='Only probe the most recent recorded image')
    parser.add_argument('--crops', nargs='+', choices=['full', 'windshield'], default=['full', 'windshield'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    warnings.filterwarnings('ignore', message='dropout2d: Received a 2-D input')
    model = load_model(args.checkpoint, args.repo)
    args.output.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in (args.session / 'samples.jsonl').read_text().splitlines()]
    samples = [s for s in samples if s.get('screen_mode') != 'menu']
    if args.last_only:
        samples = samples[-1:]
    if not samples:
        raise ValueError('No driving screenshots in session')
    # These are hypothetical route/speed inputs, not recovered ground truth.
    scenarios = [('follow_stopped', 0., 4, (0., -10.)),
                 ('straight_5ms', 5., 3, (0., -10.)),
                 ('left_5ms', 5., 1, (-6., -10.)),
                 ('right_5ms', 5., 2, (6., -10.))]
    results = []
    tiles = []
    for sample in samples:
        source = Image.open(args.session / sample['image']).convert('RGB')
        for crop in args.crops:
            image, tensor = prepare(source, crop)
            for scenario, speed, command, target in scenarios:
                result = predict(model, tensor, speed, command, target)
                result.update(image=sample['image'], crop=crop, scenario=scenario,
                              vehicle=sample.get('vehicle', 'unknown'),
                              camera_view=sample.get('camera_view', 'unknown'),
                              speed_ms_assumed=speed, command=command, target_assumed=target,
                              hypothetical_navigation=True, actuated=False)
                results.append(result)
                if sample == samples[0] and scenario in ('follow_stopped', 'straight_5ms'):
                    tile = Image.new('RGB', (900, 320), '#171e28')
                    tile.paste(image, (0, 0))
                    draw = ImageDraw.Draw(tile)
                    draw.text((10, 265), f'{crop} / {scenario}: steer={result["steer"]:.3f} '
                              f'throttle={result["throttle"]:.3f} brake={result["brake"]:.3f}', fill='white')
                    draw.text((10, 288), 'Static OOD screenshot probe; hypothetical route; no game control', fill='#ffc76b')
                    tiles.append(tile)
    latency = [r['latency_ms'] for r in results[1:]]  # exclude initial warmup
    report = dict(model='TCP', code_source=SOURCE, checkpoint_source=WEIGHTS,
                  checkpoint_type='unofficial reproduction',
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  checkpoint_strict_load=True, device='cpu', torch_version=torch.__version__,
                  decoder_verified_against_upstream=True,
                  mapping_scope='Dry run of brake-priority forward-only W/S axes; no live controller hookup',
                  input_images=len(samples), predictions=len(results),
                  median_latency_ms=statistics.median(latency),
                  p95_latency_ms=float(np.percentile(latency, 95)),
                  source_session=str(args.session.resolve()),
                  vehicles=sorted({s.get('vehicle', 'unknown') for s in samples}),
                  camera_views=sorted({s.get('camera_view', 'unknown') for s in samples}),
                  coordinate_convention='TCP waypoint x is lateral; forward is negative y',
                  result_scope='Static domain-transfer smoke test only; no accuracy or closed-loop driving score',
                  limits=['No driving video or trajectory ground truth',
                          'Route commands and target points are hypothetical',
                          'Windshield crop is heuristic, not camera calibration',
                          'Front camera model; third-person input is outside the training camera setup',
                          'Hybrid reproduces only the fresh status=0 blend; no turning-history state machine',
                          'Latency covers the neural prediction and direct action decode, not preprocessing or PID'],
                  results=results)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    contact = Image.new('RGB', (900, len(tiles) * 320))
    for i, tile in enumerate(tiles):
        contact.paste(tile, (0, i * 320))
    contact.save(args.output / 'comparison.jpg')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
    for crop in args.crops:
        for scenario, *_ in scenarios:
            group = [r for r in results if r['crop'] == crop and r['scenario'] == scenario]
            print(crop, scenario, {key: round(statistics.mean(r[key] for r in group), 3)
                                  for key in ('steer', 'throttle', 'brake')})
            print('  startup_hybrid', group[0]['startup_hybrid'],
                  'trajectory_speed', group[0]['trajectory_control']['desired_speed_ms'])


if __name__ == '__main__':
    main()
