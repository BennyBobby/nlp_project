import json
import os
import csv
import re


def aggregate_results():
    results_folder = "./"
    aggregated_data = []

    pattern = re.compile(
        r"gpt-neo-125m_(?P<dataset>book|wikinews|wikitext)_eps_(?P<alpha>[\d\.]+)_k_(?P<k>\d+)"
    )
    files = [
        f
        for f in os.listdir(results_folder)
        if f.endswith("_diversity_mauve_gen_length_result.json")
    ]

    for f_name in files:
        match = pattern.search(f_name.lower())
        if not match:
            continue

        dataset = match.group("dataset").capitalize()
        alpha = match.group("alpha")
        k = match.group("k")
        row = {
            "Stratégie": "Epsilon Greedy",
            "Modèle": "gpt-neo-125m",
            "Dataset": dataset,
            "Alpha": alpha,
            "K": k,
            "Cohérence": "N/A",
            "MAUVE": "N/A",
            "Diversité": "N/A",
            "Gen Len": "N/A",
        }
        try:
            with open(os.path.join(results_folder, f_name), "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list) and len(content) > 0:
                    data = content[0]
                    row["Diversité"] = data.get("diversity_dict", {}).get(
                        "prediction_div_mean", "N/A"
                    )
                    row["Gen Len"] = data.get("gen_length_dict", {}).get(
                        "gen_len_mean", "N/A"
                    )
                    row["MAUVE"] = data.get("mauve_dict", {}).get("mauve_mean", "N/A")
        except Exception as e:
            print(f"Erreur lecture métriques {f_name}: {e}")
        coh_f_name = f_name.replace(
            "_diversity_mauve_gen_length_result.json", "_opt-2.7b_coherence_result.json"
        )

        if os.path.exists(coh_f_name):
            try:
                with open(coh_f_name, "r", encoding="utf-8") as f:
                    coh_content = json.load(f)
                    if isinstance(coh_content, list) and len(coh_content) > 0:
                        row["Cohérence"] = coh_content[0].get("coherence_mean", "N/A")
            except Exception as e:
                print(f"Erreur lecture cohérence {coh_f_name}: {e}")

        aggregated_data.append(row)

    headers = [
        "Stratégie",
        "Modèle",
        "Dataset",
        "Alpha",
        "K",
        "Cohérence",
        "MAUVE",
        "Diversité",
        "Gen Len",
    ]
    output_file = "bilan_global_resultats_gpt-neo-125m.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(aggregated_data)

    print(f"Terminé ! Le fichier '{output_file}' est prêt.")


if __name__ == "__main__":
    aggregate_results()
