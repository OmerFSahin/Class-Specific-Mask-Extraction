# Class-Specific Mask Extraction — Reliable Multi-Label Handling

A lightweight preprocessing utility for extracting a selected class from a multi-class medical image labelmap.

The script converts the chosen label into a binary mask (0 and 1) while preserving spatial metadata.

Tested on the HVSMR-2.0 3D cardiac MRI dataset.  
The dataset is not included in this repository due to licensing and size considerations.

---

## 📂 Expected Folder Structure

data/input/
patient_001/
patient_001(SCAN).nrrd
patient_001(MASK).nrrd
patient_002/

Output will be written to:
data/output/<patient_id>/


---

##  Example Multi-Class Labels

Example labelmap:

- 0 → background
- 1 → aorta 
- 2 → chambers 
- 3 → pulmonaries 

The user selects which label to retain.

---

##  Usage

Install requirements:

```bash
pip install -r requirements.txt
```

Extract label 3:

```bash
python src/extract_label.py \
  --input-dir data/input \
  --output-dir data/output \
  --label 3
```
--label 1
--label 2
--label 3

