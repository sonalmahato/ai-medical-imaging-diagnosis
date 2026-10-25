import glob, os, yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm

P = yaml.safe_load(open("params.yaml"))
size = P["img_size"]
df = pd.read_csv("data/processed/splits.csv")
RAW = os.environ.get("RAW_DIR", "data/raw/nih_sample")
paths = {os.path.basename(p): p for p in glob.glob(f"{RAW}/**/*.png", recursive=True)}
out = "data/processed/images"
os.makedirs(out, exist_ok=True)
for name in tqdm(df["Image Index"]):
    img = Image.open(paths[name]).convert("L")      # some NIH PNGs have 4 channels
    img = img.resize((size, size), Image.LANCZOS)    # NIH PNGs are square, so no distortion
    img.save(os.path.join(out, name))