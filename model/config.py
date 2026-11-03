from dataclasses import dataclass

@dataclass
class ModelConfig:
    # --- Taille du modèle ---
    vocab_size: int = 12003             # 12000 tokens de base + 3 spéciaux
    d_model: int = 384                  # Dimension d'embedding (la "largeur" du modèle)
    n_layers: int = 8                   # Nombre de blocs Transformer (la "profondeur")
    n_heads: int = 6                    # Nombre de têtes d'attention
    d_ff: int = 1536                    # Dimension de la couche Feed-Forward (souvent 4 * d_model)
    max_seq_len: int = 256              # Longueur maximale d'une séquence (contexte)
    dropout: float = 0.1           # Taux de dropout (régularisation)

    # --- Paramètres d'entraînement (optionnels, mais utiles ici) ---
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    def __post_init__(self):
        # Vérifications de cohérence
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) doit être divisible par n_heads ({self.n_heads})"
        assert self.d_model > 0 and self.n_layers > 0