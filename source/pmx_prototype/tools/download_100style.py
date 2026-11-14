"""Download selected original 100STYLE files from the author's linked sources."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import concurrent.futures
import hashlib
import html
import json
from pathlib import Path
import re
import requests

DEST = Path('media/mocap/100style')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    page = requests.get('https://www.ianxmason.com/100style/', timeout=60)
    page.raise_for_status()
    links = re.findall(r'href="([^"]+)"[^>]*>([^<]+)</a>', page.text)
    wanted = {'Neutral_FW.bvh', 'Neutral_FR.bvh', 'Neutral_BW.bvh', 'Neutral_SW.bvh',
              'Neutral_ID.bvh', 'Neutral_TR1.bvh', 'Frame_Cuts.csv', 'Dataset_List.csv'}
    selected = {name: html.unescape(url) for url, name in links if name in wanted}
    assert set(selected) == wanted

    def fetch(item):
        name, url = item
        target = DEST / name
        if not target.exists():
            file_id = re.search(r'[?&]id=([^&]+)', url).group(1)
            response = requests.get('https://drive.usercontent.google.com/download',
                                    params={'id': file_id, 'export': 'download', 'confirm': 't'}, timeout=180)
            response.raise_for_status()
            data = response.content
            if name.endswith('.bvh'):
                assert data.lstrip().startswith(b'HIERARCHY'), (name, data[:200])
            else:
                assert b'<html' not in data[:1000].lower(), (name, data[:200])
            target.write_bytes(data)
        data = target.read_bytes()
        print(name, len(data), flush=True)
        return {'file': name, 'source': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, selected.items()))
    (DEST / 'SOURCES.json').write_text(json.dumps({
        'dataset': 'The 100STYLE Dataset', 'authors': ['Ian Mason', 'Sebastian Starke', 'Taku Komura'],
        'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'homepage': 'https://www.ianxmason.com/100style/', 'doi': 'https://doi.org/10.5281/zenodo.8127870',
        'files': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
