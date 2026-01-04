import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

MODEL_NAME = "gpt2"
INPUT_FILE = r"wikitext\wikitext_greedy_gpt2-xl_256.jsonl"
ALPHA = 0.7
K = 5
MAX_GEN_LEN = 128


def epsilon_greedy_search(model, tokenizer, prompt_text, alpha, k, max_len):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    generated = input_ids
    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
        if random.random() < alpha:  # exploitation
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        else:  # exploration
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k)
            ix = random.randint(0, k - 1)
            next_token = top_k_indices[:, ix].unsqueeze(-1)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    # On ne retourne que la partie générée (on enlève le prompt)
    gen_ids = generated[0][len(input_ids[0]) :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

all_data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    full_content = f.read()

# On essaie de découper par objets JSON en cherchant les "prompt"
# Cette méthode marche même si le fichier est mal formaté
raw_objects = full_content.split('"prompt":')
for i in range(1, len(raw_objects)):
    try:
        # On reconstruit l'objet JSON
        obj_text = '{"prompt":' + raw_objects[i]
        # On cherche la fin de l'objet (on coupe au cas où il y a une virgule ou un crochet après)
        if "}" in obj_text:
            obj_text = obj_text[: obj_text.rfind("}") + 1]

        obj = json.loads(obj_text)
        all_data.append(obj)
    except Exception:
        continue

print(f"{len(all_data)} préfixes trouvés.")

results = []
for data in tqdm(all_data):
    if isinstance(data, dict) and "prompt" in data:
        prompt = data["prompt"]
        new_gen_text = epsilon_greedy_search(
            model, tokenizer, prompt, ALPHA, K, MAX_GEN_LEN
        )

        full_text = prompt + new_gen_text
        full_tokens = tokenizer.encode(full_text)
        output_data = {
            "prefix_text": prompt,
            "generated_result": {"0": new_gen_text},
            "gold_ref": data.get("gold_ref", ""),
            "prompt": prompt,
            "gen_text": prompt + new_gen_text,
            "tokens": full_tokens,
        }
        results.append(output_data)

output_filename = f"wikitext_eps_{ALPHA}_k_{K}.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
