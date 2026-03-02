import os
from pathlib import Path

base = Path('EDFF_dataset')
for d in base.iterdir():
    if d.is_dir():
        count = len(list(d.glob("*.jpg")))
        print(f"{d.name}: {count}")
