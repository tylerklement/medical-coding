#!/usr/bin/env python3
import json, csv, os, sys
from io import BytesIO
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def build_custom_codiesp_index():
    cm_out  = DATA_DIR / "icd10cm_descriptions.json"
    pcs_out = DATA_DIR / "icd10pcs_descriptions.json"

    # We will build our index directly from the CodiEsp Train & Dev sets!
    # This completely eliminates ontology mismatches and length explosion.
    
    cm_descs = {}
    pcs_descs = {}

    for split in ["train", "dev"]:
        for ftype in ["X", "D", "P"]:
            path = DATA_DIR / split / f"{split}{ftype}.tsv"
            if not path.exists(): continue
            
            with open(path) as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    # Some files have code at index 2, text at index 3
                    # format: article_id, label, code, text, pos (for X)
                    if len(row) >= 4:
                        label = row[1]
                        code = row[2].upper().replace(".", "")
                        text = row[3].strip()
                        
                        if label == "DIAGNOSTICO":
                            cm_descs[code] = text
                        elif label == "PROCEDIMIENTO":
                            pcs_descs[code] = text
                    # format: article_id, code (for D, P)
                    elif len(row) >= 2 and ftype != "X":
                        pass # No text description available in D or P without X

    print(f"Extracted {len(cm_descs)} Diagnosis codes from Training data")
    print(f"Extracted {len(pcs_descs)} Procedure codes from Training data")

    with open(cm_out, "w", encoding="utf-8") as f:
        json.dump(cm_descs, f, ensure_ascii=False, indent=2)
    with open(pcs_out, "w", encoding="utf-8") as f:
        json.dump(pcs_descs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_custom_codiesp_index()
    print("Done building custom dataset index.")
