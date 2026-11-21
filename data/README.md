# Data

`images/` holds the 71 images used in the paper. `manifest.csv` lists them with their size in bytes and SHA-256 hash.

```
data/images/
  cbsd68/101085.jpg ... 351093.jpg    the 68 CBSD68 images (481x321 or 321x481)
  Baboon.tiff                         standard image, 512x512
  Barbara.jpg                         standard image, 512x512
  Peppers.tiff                        standard image, 512x512
```

Check the images against the manifest, from the repository root:

```python
import csv, hashlib, pathlib

root = pathlib.Path("data/images")
for row in csv.DictReader(open("data/manifest.csv", encoding="utf-8")):
    ok = hashlib.sha256((root / row["file"]).read_bytes()).hexdigest() == row["sha256"]
    print(row["file"], "ok" if ok else "DIFFERENT")
```

## File names decide the seeds

The noise seed of a run is `42 + 100000 r + 100 i + iota`, where `i` is the 0-based position of the image in the lexicographic order of the file names of its dataset. The same file names are therefore required: a missing, added or renamed file in `cbsd68/` changes the noisy images of the files after it. The standard images have `i` = 0 (Baboon), 1 (Barbara) and 2 (Peppers). Images are read with OpenCV, so the pinned version in `requirements.txt` also fixes the decoding of the JPEG files.

## Sources

- CBSD68: the color version of the 68 test images selected by Roth and Black (2009) from the Berkeley Segmentation Dataset (Martin et al., 2001), used as a color denoising test set by Zhang et al. (2017).
  - D. Martin, C. Fowlkes, D. Tal, J. Malik, "A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics," Proc. ICCV 2001, vol. 2.
  - S. Roth, M. J. Black, "Fields of Experts," International Journal of Computer Vision 82(2) (2009) 205-229.
  - K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang, "Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising," IEEE Transactions on Image Processing 26(7) (2017) 3142-3155.
- Baboon, Barbara, Peppers: common standard test images.

These images are third-party material, included here so that the experiments can be repeated. The terms of use of their original sources apply, and the license of this repository does not cover them.

## Manifest columns

| Column | Meaning |
|---|---|
| `file` | path relative to `data/images/` |
| `group` | `cbsd68` or `standard` |
| `size` | size in bytes |
| `sha256` | SHA-256 hash of the file |
