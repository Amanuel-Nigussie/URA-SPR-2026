import time
import torch
from transformers import AutoTokenizer, AutoModel

# load model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

model.eval()

# example code snippet
code = "import requests\nfrom lxml import html\nimport pandas as pd\nimport sqlite3\ndef task_func(webpage_url: str, database_name: str = \"my_database.db\") -> int:\n    try:\n        if webpage_url.startswith(\"file://\"):\n            with open(webpage_url[7:], \"r\", encoding=\"utf-8\") as file:\n                content = file.read()\n        else:\n            response = requests.get(webpage_url, timeout=5)\n            response.raise_for_status()\n            content = response.content\n\n        tree = html.fromstring(content)\n        rows = tree.xpath(\"//tr\")\n        data = [\n            [cell.text_content().strip() for cell in row.xpath(\".//td\")] for row in rows\n        ]\n\n        df = pd.DataFrame(data)\n        if df.empty:\n            return 0\n\n        conn = None\n        try:\n            conn = sqlite3.connect(database_name)\n            df.to_sql(\"my_table\", conn, if_exists=\"replace\", index=False)\n        finally:\n            if conn:\n                conn.close()\n\n        return len(df)\n\n    except requests.RequestException as e:\n        raise requests.RequestException(f\"Error accessing URL {webpage_url}: {e}\")\n    except sqlite3.DatabaseError as e:\n        raise sqlite3.DatabaseError(f\"Database error with {database_name}: {e}\")"


# tokenize
inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)

# measure time
start = time.time()

with torch.no_grad():
    outputs = model(**inputs)

embedding = outputs.last_hidden_state.mean(dim=1)

end = time.time()

print("Embedding shape:", embedding.shape)
print("Time for one embedding:", end - start, "seconds")