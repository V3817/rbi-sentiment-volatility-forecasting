import os
import pdfplumber
import pandas as pd

# Paths
MINUTES_DIR = "data/raw/minutes/"
SPEECHES_DIR = "data/raw/speeches/"
OUTPUT_FILE = "data/raw/rbi_communications_raw.csv"


def extract_text_from_folder(folder_path, doc_type):
    data = []
    # Loop through every file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            # Extract date from filename (e.g., "2024-02-08_minutes.pdf" -> "2024-02-08")
            date_str = filename.split("_")[0]

            print(f"Processing: {filename}...")

            try:
                full_text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        full_text += page.extract_text() + "\n"

                data.append({
                    "Date": date_str,
                    "Document_Type": doc_type,
                    "Raw_Text": full_text
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return data


# Run extraction
all_data = []
all_data.extend(extract_text_from_folder(MINUTES_DIR, "MPC_Minutes"))
all_data.extend(extract_text_from_folder(SPEECHES_DIR, "Speech"))

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\nSUCCESS! Saved {len(df)} documents to {OUTPUT_FILE}")
