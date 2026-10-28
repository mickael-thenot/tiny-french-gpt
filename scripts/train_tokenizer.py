import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_file = os.path.join(project_root, "data", "processed", "wikipedia_sciences_dump.txt")
    output_dir = os.path.join(project_root, "tokenizer")
    output_file = os.path.join(output_dir, "tokenizer.json")

    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {input_file} n'existe pas.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("Création du tokenizer BPE...")
    tokenizer = Tokenizer(models.BPE())

    # Pré-tokenizer : sépare sur les espaces et la ponctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Décodeur : pour reconstruire le texte à partir des tokens
    tokenizer.decoder = decoders.ByteLevel()

    # Entraîneur BPE
    trainer = trainers.BpeTrainer(
        vocab_size=12000, # Taille du vocabulaire
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
        show_progress=True
    )

    print("Entraînement sur le corpus...")
    tokenizer.train([input_file], trainer)

    print(f"Sauvegarde du tokenizer dans {output_file}...")
    tokenizer.save(output_file)

    print(f"\nTerminé ! Tokenizer sauvegardé dans : {output_file}")
    print(f"Taille du vocabulaire : {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    main()