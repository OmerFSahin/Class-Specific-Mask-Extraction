from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np
import SimpleITK as sitk


VALID_EXTS = (".nii", ".nii.gz", ".nrrd")


def is_valid_image_file(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(VALID_EXTS)


def find_scan_and_mask(files: List[Path]) -> Tuple[Path, Path]:
    """
    Expects exactly one scan and one mask in a patient folder,
    identified by '(SCAN)' and '(MASK)' in filename.
    """
    scan_files = [p for p in files if "(scan)" in p.name.lower()]
    mask_files = [p for p in files if "(mask)" in p.name.lower()]

    if len(scan_files) != 1 or len(mask_files) != 1:
        raise ValueError(f"found {len(scan_files)} scan(s), {len(mask_files)} mask(s)")

    return scan_files[0], mask_files[0]


def binarize_label(mask_img: sitk.Image, label: int) -> sitk.Image:
    """
    If mask pixel type is float, cast to UInt8 first (as requested),
    then create a binary mask for (mask == label).
    Output type is UInt8 with values {0,1}, geometry preserved.
    """
    # Read as array for explicit control
    mask_arr = sitk.GetArrayFromImage(mask_img)

    # If float -> cast to uint8 before comparison (your requirement)
    if np.issubdtype(mask_arr.dtype, np.floating):
        mask_arr = mask_arr.astype(np.uint8, copy=False)

    bin_arr = (mask_arr == label).astype(np.uint8)

    bin_img = sitk.GetImageFromArray(bin_arr)
    bin_img.CopyInformation(mask_img)
    return bin_img


def process_patient(patient_in: Path, patient_out: Path, label: int) -> None:
    files = [p for p in patient_in.iterdir() if p.is_file() and is_valid_image_file(p)]
    scan_path, mask_path = find_scan_and_mask(files)

    patient_out.mkdir(parents=True, exist_ok=True)

    # 1) copy scan as-is
    shutil.copy2(scan_path, patient_out / scan_path.name)

    # 2) binarize mask for selected label
    mask_img = sitk.ReadImage(str(mask_path))
    bin_img = binarize_label(mask_img, label=label)
    sitk.WriteImage(bin_img, str(patient_out / mask_path.name))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract a single label from a multi-class labelmap and save as a binary mask (0/1)."
    )
    p.add_argument("--input-dir", type=Path, default=Path("data/input"), help="Root input folder with patient subfolders.")
    p.add_argument("--output-dir", type=Path, default=Path("data/output"), help="Root output folder.")
    p.add_argument("--label", type=int, required=True, choices=[1, 2, 3], help="Which label to keep (1, 2, or 3).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patient_folders = [d for d in sorted(args.input_dir.iterdir()) if d.is_dir()]
    if not patient_folders:
        print(f"No patient folders found under: {args.input_dir}")
        return 1

    for patient_in in patient_folders:
        try:
            process_patient(patient_in, args.output_dir / patient_in.name, label=args.label)
        except Exception as e:
            print(f"[SKIP] {patient_in.name}: {e}")
            continue

        print(f"[OK] {patient_in.name}: scan copied, mask binarized for label={args.label}")

    print("All done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
