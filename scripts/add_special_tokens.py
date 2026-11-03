import os

from tokenizers import Tokenizer, AddedToken

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    tokenizer_path = os.path.join(project_root, "tokenizer", "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        print(f"Erreur : Le tokenizer {tokenizer_path} n'existe pas.")
        return

    print("Chargement du tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size_avant = tokenizer.get_vocab_size()
    print(f"Taille du vocabulaire avant : {vocab_size_avant}")

    # Ajouter les tokens spéciaux à la fin
    special_tokens = ["<|user|>", "<|assistant|>", "<|end|>"]
    tokens_to_add = [AddedToken(tok, special=True) for tok in special_tokens]
    tokenizer.add_special_tokens(tokens_to_add)

    vocab_size_apres = tokenizer.get_vocab_size()
    print(f"Taille du vocabulaire après: {vocab_size_apres}")

    # Sauvegarder
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer sauvegardé dans {tokenizer_path}")

    # Vérifier les IDs
    for tok in special_tokens:
        print(f"  {tok} -> ID {tokenizer.token_to_id(tok)}")

if __name__ == "__main__":
    main()