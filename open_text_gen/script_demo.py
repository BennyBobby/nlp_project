import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from epsilon_greedy_search import eps_greedy_search_algorithm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Démo Epsilon-Greedy (Device: {device}) ---")

    # 1. Sélection du modèle
    print("\nModèles disponibles :")
    print("1. GPT-2 Small")
    print("2. GPT-Neo 125M")
    print("3. OPT 125M")
    
    choix = input("Choisissez un modèle (1, 2 ou 3) : ")
    
    models = {
        "1": "gpt2",
        "2": "EleutherAI/gpt-neo-125M",
        "3": "facebook/opt-125m"
    }
    
    model_id = models.get(choix, "gpt2")
    
    print(f"\nChargement de {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # 2. Boucle de démo
    while True:
        print("\n" + "="*50)
        prompt = input("Entrez votre prompt (ou 'exit' pour quitter) : ")
        if prompt.lower() == 'exit':
            break
            
        try:
            alpha = float(input("Valeur de Alpha (exploitation) [0.0 - 1.0] : "))
            k = int(input("Valeur de K (exploration) : "))
            max_len = int(input("Longueur max de génération : "))
        except ValueError:
            print("Erreur : Veuillez entrer des nombres valides.")
            continue

        print("\n--- Génération en cours... ---")
        
        # Appel de ta fonction
        reponse, is_ended = eps_greedy_search_algorithm(
            model, tokenizer, prompt, alpha, k, max_len
        )

        print("\n[PROMPT]:", prompt)
        print("-" * 20)
        print("[GÉNÉRATION]:", reponse)
        print("-" * 20)
        print(f"Statut : {'Terminé par EOS' if is_ended else 'Atteint max_len'}")
        
    print("\nFin de la démo. Bonne chance pour ta soutenance !")

if __name__ == "__main__":
    main()