import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# 1. Setup - Load Model and Tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def get_sentiment_score(text):
    """
    Processes long text using a sliding window/chunking approach.
    Returns a single averaged sentiment score: prob(pos) - prob(neg).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0

    # Tokenize the entire text without truncation
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]

    # Define window size (512 - 2 special tokens = 510)
    chunk_size = 510
    chunks = input_ids.split(chunk_size)

    chunk_probs = []

    for chunk in chunks:
        # Add special tokens [CLS] and [SEP] back to each chunk
        chunk_with_special = torch.cat([
            torch.tensor([tokenizer.cls_token_id]),
            chunk,
            torch.tensor([tokenizer.sep_token_id])
        ]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(chunk_with_special)
            # Apply softmax to get probabilities
            probs = F.softmax(outputs.logits, dim=-1)
            chunk_probs.append(probs.cpu())

    # Average probabilities across all chunks
    # FinBERT labels: [0: positive, 1: negative, 2: neutral]
    avg_probs = torch.mean(torch.stack(chunk_probs), dim=0).squeeze()

    pos_score = avg_probs[0].item()
    neg_score = avg_probs[1].item()

    # Calculate net sentiment score
    return pos_score - neg_score


# 2. Load Data
raw_data_path = 'data/raw/rbi_communications_raw.csv'
df = pd.read_csv(raw_data_path)

# 3. Process with Progress Bar
print(f"Starting sentiment analysis on {len(df)} documents using {device}...")
tqdm.pandas(desc="Processing RBI Docs")
df['Sentiment_Score'] = df['Raw_Text'].progress_apply(get_sentiment_score)

# 4. Save Intermediate Result
output_dir = 'data/processed/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'rbi_sentiment_scores.csv')

# Keeping only necessary columns for the merge
final_df = df[['Date', 'Document_Type', 'Sentiment_Score']]
final_df.to_csv(output_path, index=False)

print(f"Step 1 Complete. Sentiment scores saved to {output_path}")
