import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.config import ModelConfig
from model.transformer import TinyFrenchGPT


def generer_texte(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu"
):
    """
    Génère du texte à partir d'un prompt.
    """
    model.eval()

    # Tokeniser le prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)

    # Génération token par token
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids[:, -model.config.max_seq_len:]
            logits = model(input_ids_cond)
            next_token_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_values[-1]
                next_token_logits[next_token_logits < min_top_k] = float("-inf")
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Décoder la séquence complète
    texte_genere = tokenizer.decode(input_ids[0].tolist())
    return texte_genere


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    tokenizer_path = os.path.join(project_root, "tokenizer", "tokenizer.json")
    checkpoint_path = os.path.join(project_root, "checkpoints", "checkpoint_step_28000.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Erreur : Le checkpoint {checkpoint_path} n'existe pas.")
        print("Lance d'abord l'entraînement avec training/train_pretrain.py.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")

    # Charger le tokenizer
    print("Chargement du tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Charger le modèle
    print("Chargement du modèle...")
    config = ModelConfig()
    model = TinyFrenchGPT(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Modèle chargé !")

    # Générer du texte avec des prompts neutres
    prompts = [
        "La science",
        "Un ordinateur",
        "Le cerveau",
        "Une équation",
        "La matière"
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt : {prompt}")
        print(f"{'='*60}")
        texte = generer_texte(
            model, tokenizer, prompt,
            max_new_tokens=60,
            temperature=0.8,
            top_k=50,
            device=device
        )
        print(texte)


if __name__ == "__main__":
    main()