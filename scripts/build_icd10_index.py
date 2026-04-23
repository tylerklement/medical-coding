#!/usr/bin/env python3
import json, csv
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"

def build_custom_codiesp_index():
    cm_out  = DATA_DIR / "icd10cm_descriptions.json"
    pcs_out = DATA_DIR / "icd10pcs_descriptions.json"
    
    cm_descs = defaultdict(set)
    pcs_descs = defaultdict(set)

    for split in ["train", "dev"]:
        for ftype in ["X", "D", "P"]:
            path = DATA_DIR / split / f"{split}{ftype}.tsv"
            if not path.exists(): continue
            
            with open(path) as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 4:
                        label = row[1]
                        code = row[2].upper().replace(".", "")
                        text = row[3].strip()
                        
                        if label == "DIAGNOSTICO":
                            cm_descs[code].add(text)
                        elif label == "PROCEDIMIENTO":
                            pcs_descs[code].add(text)

    # Join all unique synonyms into a single descriptive document string
    cm_final = {code: " | ".join(sorted(list(texts))) for code, texts in cm_descs.items()}
    pcs_final = {code: " | ".join(sorted(list(texts))) for code, texts in pcs_descs.items()}

    print(f"Extracted {len(cm_final)} Diagnosis codes")
    print(f"Extracted {len(pcs_final)} Procedure codes")

    with open(cm_out, "w", encoding="utf-8") as f:
        json.dump(cm_final, f, ensure_ascii=False, indent=2)
    with open(pcs_out, "w", encoding="utf-8") as f:
        json.dump(pcs_final, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_custom_codiesp_index()
    print("Done building custom dataset index.")
