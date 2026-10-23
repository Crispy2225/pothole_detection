"""Category-matched illustrative photos. Never recorded event evidence."""
import base64
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from config import ROOT
FOLDERS={"pothole":"pothole_images","speed_breaker":"speedbreaker_images"}
EXTENSIONS={".jpg",".jpeg",".png",".webp",".avif"}

@lru_cache(maxsize=32)
def thumbnail(path, modified, size):
    with Image.open(path) as original:
        if original.width*original.height>30_000_000:raise ValueError("Image too large")
        image=ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((640,420))
        output=BytesIO();image.save(output,format="JPEG",quality=82)
    return "data:image/jpeg;base64,"+base64.b64encode(output.getvalue()).decode("ascii")


def sample_image(event,root=ROOT):
    folder=FOLDERS.get(event.get("prediction"))
    if not folder:return None
    directory=Path(root)/folder
    if not directory.is_dir():return None
    candidates=sorted(p for p in directory.iterdir() if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS and p.stat().st_size<=20*1024*1024)
    if not candidates:return None
    # Stable pseudorandom selection by report ID. Existing photos are not altered.
    start=int(sha256(str(event["id"]).encode()).hexdigest(),16)%len(candidates)
    for index in range(len(candidates)):
        path=candidates[(start+index)%len(candidates)]
        try:
            stat=path.stat()
            return {"key":folder+"/"+path.name,"url":thumbnail(str(path),stat.st_mtime_ns,stat.st_size)}
        except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError):continue
    return None
