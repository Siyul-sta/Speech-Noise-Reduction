"""Download only the fixed Train evaluation pairs and check their original hashes."""
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import argparse

HERE=Path(__file__).resolve().parent
FOLDERS={"clean":"1IzCZ371Oy6NqNiOr8DILOD7IIYJpfyOx","noisy":"1iWMO3AEsGqMWBhs6ob4gEparKz2kq-w4"}

def fetch(url):
    return subprocess.check_output(["curl","-fLsS","--retry","2","--max-time","60",url])

def save(task):
    path,file_id,expected=task
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
        return
    data=fetch("https://drive.usercontent.google.com/download?export=download&confirm=t&id="+file_id)
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError(f"Checksum mismatch for {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=HERE.parent/"noise_reduction_challenge")
    args=parser.parse_args()
    rows=[]
    for manifest in ("train_split.csv","extra.csv"):
        with (HERE/manifest).open() as handle:
            rows.extend(csv.DictReader(handle))
    tasks=[]
    for kind,folder in FOLDERS.items():
        html=fetch("https://drive.google.com/embeddedfolderview?id="+folder).decode()
        ids={name:file_id for file_id,name in re.findall(r'id="entry-([^"]+)".*?class="flip-entry-title">([^<]+)',html)}
        for row in rows:
            path=args.data/row[kind+"_path"]
            tasks.append((path,ids[path.name],row[kind+"_sha256"]))
    with ThreadPoolExecutor(12) as pool:
        list(pool.map(save,tasks))
    print(f"Verified {len(tasks)} audio files.")
