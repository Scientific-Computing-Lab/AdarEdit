#!/usr/bin/env python3
import csv
import json
import os
import sys


SYSTEM_MESSAGE = {
        "role": "system",
        "content": "Predict if the central adenosine (A) in the given RNA sequence context within an Alu element will be edited to inosine (I) by ADAR enzymes."
    }
    

def csv_to_jsonl(input_filename: str) -> None:
    """
    ממיר קובץ CSV אחד לקובץ JSONL באותו שם (עם סיומת שונה)
    """
    base_name = os.path.splitext(os.path.basename(input_filename))[0]  
    directory = os.path.dirname(input_filename)
    output_filename = os.path.join(directory, base_name + ".jsonl")   
    data = []

    with open(input_filename, mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None) 

        for row in csv_reader:
            if not row or all(not cell.strip() for cell in row):
                continue

            try:
                structure, L, R, y_n = row[:4]
            except ValueError:
                print(f"Skipping malformed row in {input_filename}: {row}")
                continue

            content = (
                f"L:{L}, A:A, R:{R}, Alu Vienna Structure:{structure}"
            )

            data.append(
                {
                    "messages": [
                        SYSTEM_MESSAGE,
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": y_n},
                    ]
                }
            )

    with open(output_filename, mode="w") as out_file:
        for entry in data:
            out_file.write(json.dumps(entry) + "\n")

    print(f"Saved JSONL: {output_filename}")


def convert_all_csv(root_dir: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                csv_path = os.path.join(dirpath, filename)
                csv_to_jsonl(csv_path)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    convert_all_csv(root)
