import os
import torch
from utils.utils import get_sparse_model, get_tokenizer

OUT = "allocation_results"   # the OUTPUT_PATH from the bash scripts
MODEL_PATH = "Qwen2.5-Math-7B-Instruct"

tokenizer = get_tokenizer(MODEL_PATH)
model = get_sparse_model(
    MODEL_PATH,
    device="auto",                                  # or "cpu" / "cuda:0"
    histogram_path=os.path.join(OUT, "histograms"), # required arg, even for topk
    sparse_mode="wina",                             # or "teal"
    mask_by="topk",                                 # or "threshold"
    transform=False,                                 # must match how you ran grab_acts
    torch_dtype=torch.float16,
)

model.load_greedy_sparsities(os.path.join(OUT, "lookup"), 0.6) 

model.eval()

# --- Run inference ---
question = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."

messages = [{"role": "user", "content": question}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=8192,
        do_sample=False,
    )

# Decode only the newly generated tokens
response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"Q: {question}")
print(f"A: {response}")

# Compute each token's entropy. Check the changing trend.