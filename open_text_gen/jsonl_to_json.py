import json

input_file = "book_eps_0.7_k_5.jsonl"  # Mets le nom de ton fichier ici
output_file = "book_eps_0.7_k_5.json"

final_list = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            # Chaque ligne est une liste d'un élément [{...}]
            # On charge la ligne, et on prend le premier élément [0]
            try:
                data_list = json.loads(line)
                final_list.append(data_list[0])
            except Exception as e:
                print(f"Erreur sur une ligne : {e}")

# On sauvegarde tout dans une seule vraie liste JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_list, f, indent=4)

print(f"✅ Terminé ! Utilise '{output_file}' avec les scripts du prof.")
