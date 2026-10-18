"""Durable status for a single Linux GPU experiment (no desktop controls)."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--output', default='models/imitation_run')
    parser.add_argument('--dataset', default='dataset')
    args = parser.parse_args()
    status_path = Path(args.output + '.status.json')
    if status_path.exists() or Path(args.output).exists():
        raise SystemExit('Run already exists')
    status_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'wardog_ad.imitation', 'train', args.dataset,
               '--output', args.output, '--epochs', str(args.epochs), '--device', 'cuda']
    status = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  pid=os.getpid(), gpu=args.gpu, command=command, state='running')
    status_path.write_text(json.dumps(status, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1')
    try:
        result = subprocess.run(command, env=env, check=False)
        status.update(state='completed' if result.returncode == 0 else 'failed', exit_code=result.returncode)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        status['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
