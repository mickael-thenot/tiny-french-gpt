import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from model.config import ModelConfig


class TextDataset(Dataset):
    """
    Dataset pour le pré-entraînement : prédiction du token suivant.
    """
    def __init__(self, token_ids: torch.Tensor, max_seq_len: int):
        """
        Args:
            token_ids: Tenseur 1D de tous les tokens du corpus (déjà tokenisé).
            max_seq_len: Longueur maximale d'une séquence.
        """
        self.token_ids = token_ids
        self.max_seq_len = max_seq_len

    def __len__(self):
        # On peut créer une séquence à chaque position, sauf les dernières
        # où il n'y a pas assez de tokens pour former une séquence complète
        return len(self.token_ids) - self.max_seq_len - 1

    def __getitem__(self, idx):
        # Séquence d'entrée : tokens de idx à idx + max_seq_len
        input_ids = self.token_ids[idx:idx + self.max_seq_len]
        # Cible : tokens de idx + 1 à idx + max_seq_len + 1 (décalage de 1)
        target_ids = self.token_ids[idx + 1:idx + self.max_seq_len + 1]
        return input_ids, target_ids


def charger_et_tokeniser_corpus(
    corpus_path: str,
    tokenizer_path: str,
    max_tokens: int = None
) -> torch.Tensor:
    """
    Charge le corpus, le tokenise, et retourne un tenseur 1D de tokens.

    Args:
        corpus_path: Chemin vers le fichier texte.
        tokenizer_path: Chemin vers tokenizer.json.
        max_tokens: Nombre maximum de tokens à garder (None = tout).

    Returns:
        Tenseur 1D de tokens (torch.long).
    """
    print(f"Chargement du tokenizer depuis {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Lecture du corpus depuis {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        texte = f.read()

    print(f"Taille du corpus {len(texte):,} caractères")

    # Tokenisation par chunks pour éviter de saturer la mémoire
    print("Tokenisation en cours...")
    chunk_size = 10_000_000 # 10 Mo par chunk
    all_ids = []

    for i in range(0, len(texte), chunk_size):
        chunk = texte[i:i + chunk_size]
        encoded = tokenizer.encode(chunk)
        all_ids.extend(encoded.ids)
        print(f" {i + len(chunk):,} / {len(texte):,} caractères traités...")

    print(f"Nombre total de tokens {len(all_ids):,}")

    # Limiter si demandé
    if max_tokens is not None:
        all_ids = all_ids[:max_tokens]
        print(f"Nombre total de tokens : {len(all_ids):,}")

    # Convertir en tenseur
    token_ids = torch.tensor(all_ids, dtype=torch.long)
    return token_ids


def creer_dataloader(
    corpus_path: str,
    tokenizer_path: str,
    config: ModelConfig,
    max_tokens: int = None,
    batch_size: int = None,
) -> DataLoader:
    """
    Crée un DataLoader prêt pour l'entraînement.
    """
    if batch_size is None:
        batch_size = config.batch_size

    # Charger et tokeniser le corpus
    token_ids = charger_et_tokeniser_corpus(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens
    )

    # Créer le dataset
    dataset = TextDataset(token_ids, config.max_seq_len)
    print(f"Nombre de séquences disponibles : {len(dataset):,}")

    # Créer le DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, # Sur Windows, 0 est plus stable
        pin_memory=False
    )

    return dataloader


if __name__ == "__main__":
    # Test rapide
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    corpus_path = os.path.join(project_root, "data", "processed", "wikipedia_sciences_dump.txt")
    tokenizer_path = os.path.join(project_root, "tokenizer", "tokenizer.json")

    config = ModelConfig()

    # On limite à 1 million de tokens pour le test
    dataloader = creer_dataloader(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        config=config,
        max_tokens=1_000_000,
        batch_size=4
    )

    # Vérifier un batch
    print("\nVérification d'un batch...")
    for input_ids, target_ids in dataloader:
        print(f"input_ids shape : {input_ids.shape}")
        print(f"target_ids shape : {target_ids.shape}")
        print(f"Premiers tokens d'entrée : {input_ids[0][:10].tolist()}")
        print(f"Premiers tokens cibles : {target_ids[0][:10].tolist()}")
        break