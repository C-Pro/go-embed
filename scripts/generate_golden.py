import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def main():
    model_dir = "models/intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    test_inputs = [
        # Short queries & passages
        "query: how to implement consensus in distributed systems?",
        "passage: Consensus in distributed systems is the process of agreeing on a data value among multiple nodes or processes (e.g. Raft, Paxos).",
        "query: what is machine learning",
        "passage: Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn.",
        
        # Cross-lingual equivalence (English, Russian, Indonesian, German, Chinese)
        "query: How to implement consensus in distributed systems?",
        "query: Как реализовать консенсус в распределенных системах?",
        "query: Bagaimana cara mengimplementasikan konsensus dalam sistem terdistribusi?",
        "query: Wie implementiert man einen Konsens in verteilten Systemen?",
        "query: 如何在分布式系统中实现共识？",
        
        "query: The cat is sleeping on the couch.",
        "query: Кот спит на диване.",
        "query: Kucing itu sedang tidur di sofa.",
        "query: Die Katze schläft auf der Couch.",
        "query: 猫在沙发上睡觉。",

        # Negative pairs / unrelated topics
        "query: Kubernetes cluster deployment on bare metal",
        "passage: Authentic Italian tiramisu recipe with mascarpone, espresso, and ladyfingers.",
        "query: Low latency order matching engine",
        "passage: History of Renaissance painting and sculpture in Florence during the 15th century.",

        # Hard negatives (lexical overlap with contradictory meaning)
        "passage: The primary database node is healthy and accepting writes.",
        "passage: The primary database node crashed and lost data.",
        
        # Code snippet
        "query: func (c *Context) Embed(text string) ([]float32, error)",
        
        # Empty string
        "",
        "query: ",
        "passage: ",
        
        # Long text (repeated to test longer sequence)
        "query: " + "Transformer models use self-attention mechanisms to process sequential text data efficiently. " * 30,
    ]

    # Also build a 512-token sequence
    long_words = ("distributed consensus algorithm raft paxos state machine replication " * 50)[:1500]
    test_inputs.append("passage: " + long_words)

    records = []

    for text in test_inputs:
        encoded = tokenizer(
            text,
            max_length=512,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(**encoded)
            # outputs.last_hidden_state: [1, seq_len, 384]
            last_hidden_state = outputs.last_hidden_state

            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # L2 normalize
            l2_normalized = F.normalize(mean_pooled, p=2, dim=1)

        records.append({
            "text": text,
            "input_ids": input_ids[0].tolist(),
            "attention_mask": attention_mask[0].tolist(),
            "tokens": tokens,
            "seq_len": len(input_ids[0]),
            "pooled_embedding": [round(float(x), 8) for x in mean_pooled[0].tolist()],
            "embedding": [round(float(x), 8) for x in l2_normalized[0].tolist()],
        })

    os.makedirs("testdata", exist_ok=True)
    with open("testdata/golden.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Generated testdata/golden.json with {len(records)} test cases.")

if __name__ == "__main__":
    main()
