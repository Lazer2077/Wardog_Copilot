"""Audit Git's staged snapshot for accidental local artifacts and broken doc links."""
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


def main():
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(['git','ls-files','-z'],cwd=root).decode('utf-8').split('\0')
    files = [p for p in files if p]
    errors,total = [],0
    forbidden_roots = {'data','logs','models','reports','terrain','audio_clips','third_party'}
    secret_patterns = [r'gh[pousr]_[A-Za-z0-9]{24,}',r'sk-[A-Za-z0-9_-]{32,}',
                       r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'AKIA[A-Z0-9]{16}']
    local_patterns = [r'[A-Za-z]:[\\/]Users[\\/]',r'/home/' + r'myid/',r'cscidl' + r'\d+\.']
    for name in files:
        path = Path(name)
        blob = subprocess.check_output(['git','show',':'+name],cwd=root)
        total += len(blob)
        if path.parts[0] in forbidden_roots or path.suffix.lower() not in ('.py','.md','.json','.txt','.cmd','.vbs',''):
            if name != '.gitignore':
                errors.append((name,'unexpected artifact or file type'))
        if name in ('config.json','config.backup.json','lock_audio.json'):
            errors.append((name,'local configuration'))
        if path.suffix == '.json' and name not in ('config.example.json','wardog_ad/data/team_bases.json'):
            errors.append((name,'unexpected data JSON'))
        if len(blob)>512*1024:
            errors.append((name,'file exceeds source-only size limit'))
        try:
            content = blob.decode('utf-8')
        except UnicodeDecodeError:
            errors.append((name,'not UTF-8 text'))
            continue
        if any(re.search(pattern,content) for pattern in secret_patterns):
            errors.append((name,'possible credential; value not printed'))
        if any(re.search(pattern,content,re.I) for pattern in local_patterns):
            errors.append((name,'local account/server information'))
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)',content):
                target = target.strip('<>').split('#')[0]
                if not target or re.match(r'[a-z]+:',target,re.I):
                    continue
                local = (path.parent/unquote(target)).as_posix()
                if local not in files and not any(f.startswith(local.rstrip('/')+'/') for f in files):
                    errors.append((name,'unpublished link: '+target))
    print(json.dumps(dict(files=len(files),bytes=total,errors=errors),indent=2,ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
