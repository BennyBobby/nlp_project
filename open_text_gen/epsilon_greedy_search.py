import torch
import json
import random as rd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

def eps_greedy_search_algorithm(model, tokenizer, prompt, alpha, k, max_len):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    current_sequence = input_ids
    past_key_values = None  
    is_ended = False
    generated_tokens = []

    for i in range(max_len):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(current_sequence, use_cache=True)
            else:
                outputs = model(
                    current_sequence[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits
            past_key_values = outputs.past_key_values  # mise à jour du cache
            next_token_logits = logits[:, -1, :]

        if rd.random() < alpha:
            # exploitation
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        else:
            # exploration
            values, indices = torch.topk(next_token_logits, k)
            random_indice = rd.randint(0, k - 1)
            next_token = indices[:, random_indice].unsqueeze(-1)
            
        current_sequence = torch.cat((current_sequence, next_token), dim=-1)
        generated_tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            is_ended = True
            break

    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response_text, is_ended

if __name__ == "__main__":
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"Nom du GPU : {torch.cuda.get_device_name(device)}")
    elif device.type == "cpu":
        print("Utilisation du processeur (CPU)")

    model_name = "EleutherAI/gpt-neo-125M"
    input_file = "wikitext/wikitext_greedy_gpt2-xl_256.jsonl"
    source_name = input_file.split("/")[0]
    
    all_k = [4, 6, 8, 10]
    all_alpha = [0.5, 0.6, 0.7, 0.8]
    max_len = 128

    print(f"Chargement du modèle : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    all_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_data.append(json.loads(line.strip().strip("[] ,")))
            except:
                continue
                
    print(f"{len(all_data)} préfixes au total")
    all_data = all_data[:100]
    print(f"Nous prenons que les {len(all_data)} premiers préfixes")

    for alpha in all_alpha:
        for k in all_k:
            print(f"\nLancement : Alpha={alpha}, K={k}")
            results = []

            for data in tqdm(all_data, desc=f"Alpha {alpha} | K {k}"):
                prompt = data.get("prompt")
                if not prompt:
                    continue

                response, is_ended = eps_greedy_search_algorithm(
                    model, tokenizer, prompt, alpha, k, max_len
                )

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

            output_filename = f"gpt-neo-125M_{source_name}_eps_{alpha}_k_{k}.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            print(f"Fichier sauvegardé : {output_filename}")