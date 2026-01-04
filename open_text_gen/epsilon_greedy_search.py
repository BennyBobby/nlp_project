import torch
import json
import random as rd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

device = Accelerator().device
print(f"Device : {device}")

if device.type == "cuda":
    print(f"Nom du GPU : {torch.cuda.get_device_name(device)}")
elif device.type == "cpu":
    print("Utilisation du processeur (CPU)")

model_name = "gpt2"
input_file = "wikitext/wikitext_greedy_gpt2-xl_256.jsonl"
source_name = input_file.split("/")[0]
all_k = [4, 6, 8, 10]
all_alpha = [0.5, 0.6, 0.7, 0.8]
alpha = all_alpha[3]
k = all_k[2]
max_len = 128
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


def eps_greedy_search_algorithm(model, tokenizer, prompt, alpha, k, max_len):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_response = tokenized_prompt
    is_ended = False
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(tokenized_prompt).logits
            next_token_logits = logits[:, -1, :]
        if rd.random() < alpha:  # partie exploitation
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        else:  # partie exploration
            _, top_k_indices = torch.topk(next_token_logits, k)
            random_indice = rd.randint(0, k - 1)
            next_token = top_k_indices[:, random_indice].unsqueeze(-1)
        prompt_response = torch.cat((prompt_response, next_token), dim=-1)
        if (
            next_token.item() == tokenizer.eos_token_id
        ):  # eos_token_id est le token de fin, si le token est celui de fin alors la boucle prend fin
            is_ended = True
            break
    tokenized_response = prompt_response[0][len(tokenized_prompt[0]) :]
    return (
        tokenizer.decode(tokenized_response, skip_special_tokens=True),
        is_ended,
    )


all_data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            all_data.append(json.loads(line.strip().strip("[] ,")))
        except:
            continue
print(f"{len(all_data)} préfixes")

results = []
for data in tqdm(all_data):
    prompt = data.get("prompt")
    if not prompt:
        continue
    (response, is_ended) = eps_greedy_search_algorithm(
        model, tokenizer, prompt, alpha, k, max_len
    )
    full_text = prompt + response
    results.append(
        {
            "ended": is_ended,
            "tokens": tokenizer.encode(prompt + response),
            "prefix_text": prompt,
            "generated_result": {"0": response},
            "prompt": prompt,
            "gen_text": prompt + response,
            "reference_text": data.get("gold_ref", ""),
        }
    )

output_filename = f"{source_name}_eps_{alpha}_k_{k}.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(results, f)
