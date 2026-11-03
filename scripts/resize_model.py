import os
import torch

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    checkpoint_path = os.path.join(project_root, "checkpoints", "checkpoint_step_28000.pt")
    output_path = os.path.join(project_root, "checkpoints", "checkpoint_step_28000_resized.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Erreur : Le checkpoint {checkpoint_path} n'existe pas.")
        return

    print("Chargement du checkpoint ...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    old_state = checkpoint["model_state_dict"]

    # Ancienne et nouvelle taille de vocabulaire
    old_vocab = old_state["tok_emb.weight"].shape[0]
    new_vocab = 12003
    d_model = old_state["tok_emb.weight"].shape[1]
    print(f"Ancien vocabulaire : {old_vocab}, nouveau : {new_vocab}")

    # Calcul de la moyenne et de l'écart-typepour l'initialisation
    mean = old_state["tok_emb.weight"].mean().item()
    std = old_state["tok_emb.weight"].std().item()

    new_tok_emb = torch.empty(new_vocab, d_model, dtype=old_state["tok_emb.weight"].dtype)
    new_tok_emb[:old_vocab] = old_state["tok_emb.weight"]
    new_tok_emb[old_vocab:] = torch.normal(mean, std, size=(new_vocab - old_vocab, d_model))
    old_state["tok_emb.weight"] = new_tok_emb

    # --- Redimensionnement de lm_head.weight (si présent) ---
    # Comme lm_head partage ses poids avec tok_emb, on lui donne le même tenseur
    if "lm_head.weight" in old_state:
        old_state["lm_head.weight"] = new_tok_emb
    else:
        # Si la clé n'existe pas dans le checkpoint, on l'ajoute
        old_state["lm_head.weight"] = new_tok_emb

    # Sauvegarder
    checkpoint["model_state_dict"] = old_state
    torch.save(checkpoint, output_path)
    print(f"Checkpoint redimensionné sauvegardé : {output_path}")

if __name__ == "__main__":
    main()