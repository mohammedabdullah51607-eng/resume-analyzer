import os
import pdfplumber
import pandas as pd

data = []

base_path = "dataset"

for role in os.listdir(base_path):
    role_path = os.path.join(base_path, role)

    if os.path.isdir(role_path):
        for file in os.listdir(role_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(role_path, file)

                text = ""
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text += page.extract_text() or ""
                except:
                    continue

                data.append([text, role])

# Create CSV
df = pd.DataFrame(data, columns=["text", "role"])
df.to_csv("dataset.csv", index=False)

print("✅ dataset.csv created")