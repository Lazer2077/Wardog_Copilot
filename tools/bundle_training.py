"""Package a reviewed low-resolution dataset and offline learner for remote training."""
import argparse
import hashlib
from pathlib import Path
import tarfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset,output = Path(args.dataset).resolve(),Path(args.output).resolve()
    files = ['images.npy','labels.npz','dataset.json','review.jpg']
    for name in files:
        if not (dataset/name).is_file():
            raise ValueError(f'Missing prepared dataset file: {name}')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'x:gz',compresslevel=3) as bundle:
        for name in ['wardog_ad/__init__.py','wardog_ad/imitation.py','tools/run_training_job.py',
                     'tests/test_imitation.py','requirements-training.txt','docs/training.md']:
            bundle.add(root/name,arcname=name)
        for name in files:
            bundle.add(dataset/name,arcname='dataset/'+name)
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    print(output,output.stat().st_size,digest.hexdigest())


if __name__ == '__main__':
    main()
