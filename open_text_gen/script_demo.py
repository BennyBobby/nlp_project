import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from epsilon_greedy_search import eps_greedy_search_algorithm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("1. GPT-2 Small")
    print("2. GPT-Neo 125M")
    print("3. OPT 125M")

    choix = input("Which model (1, 2 or 3): ")
    models = {"1": "gpt2", "2": "EleutherAI/gpt-neo-125M", "3": "facebook/opt-125m"}
    model_id = models.get(choix, "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    while True:
        prompt = input("Write the prompt (or write 'end'): ")
        if prompt.lower() == "end":
            break
        try:
            alpha = float(input("Alpha [0.0 - 1.0] : "))
            k = int(input("K : "))
            max_len = int(input("Longueur max de génération : "))
        except ValueError:
            print("Error")
            continue

        reponse, is_ended = eps_greedy_search_algorithm(
            model, tokenizer, prompt, alpha, k, max_len
        )
        print("\nPrompt:", prompt)
        print("Response :", reponse)
        print(f"Statut : {'EOS' if is_ended else 'max length reached'}")


if __name__ == "__main__":
    main()
