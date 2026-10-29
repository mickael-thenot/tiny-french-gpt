import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.config import ModelConfig
from model.transformer import TinyFrenchGPT
from training.dataset import creer_dataloader


def main():
    # --- Chemins ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    corpus_path = os.path.join(project_root, 'data', "processed", "wikipedia_sciences_dump.txt")
    tokenizer_path = os.path.join(project_root, "tokenizer", "tokenizer.json")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Configuration ---
    config = ModelConfig()

    # --- Device (CPU ou GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")

    # --- Chargement du modèle ---
    print(f"Création du modèle...")
    model = TinyFrenchGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")

    # --- Chargement des données ---
    # On limite à 5 millions de tokens pour un premier test
    print("Chargement des données...")
    dataloader = creer_dataloader(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        config=config,
        max_tokens=5_000_000,
        batch_size=config.batch_size
    )

    # --- Optimiseur et scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Scheduler cosine : le learning rate diminue progressivement
    total_steps = len(dataloader) * 1 # 1 epoch pour le test
    scheduler = CosineAnnealingLR(optimizer, total_steps)

    # --- Loss ---
    # CrossEntropyLoss ignore le token <pad> (index 0 par convention)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # --- Boucle d'entraînement ---
    print("Début de l'entraînement...")
    model.train()
    step = 0
    start_time = time.time()

    for epoch in range(1): # 1 epoch pour le test
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward
            logits = model(input_ids) # (B, T, vocab_size)

            # Calcul de la loss
            # On reshape pour que CrossEntropyLoss comprenne les dimensions
            # logits : (B*T, vocab_size), targets : (B*T,)
            loss = criterion(
                logits.view(-1, config.vocab_size),
                target_ids.view(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (évite l'explosion des gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Mise à jour des poids
            optimizer.step()
            scheduler.step()

            step += 1

            # Affichage tous les 10 steps
            if step % 10 == 0:
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step} | Loss : {loss.item():.4f} | LR : {lr:.2e} | Temps: {elapsed:.1f}s")

            # Sauvegarde du checkpoint tous les 200 steps
            if step % 200 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, checkpoint_path)
                print(f"Checkpoint sauvegardé : {checkpoint_path}")

    print("Entraînement terminé !")

    # Sauvegarde finale
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Modèle final sauvegardé : {final_path}")



if __name__ == "__main__":
    main()




