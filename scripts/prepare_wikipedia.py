import os
import re
from datasets import load_dataset
from tqdm import tqdm

# Catégories scientifiques que l'on veut cibler
# (On va filtrer sur le titre ou le contenu pour ne garder que les sciences)
MOTS_CLES_SCIENCES = [
    "mathématique",
    "mathématiques",
    "informatique",
    "physique",
    "sciences cognitives",
    "sciences",
    "algorithmique",
    "algorithmes",
    "physicien",
    "chimiste",
    "mathématicien",
    "biologie"
]

def nettoyer_texte(texte):
    """Nettoie le texte brut de Wikipédia."""
    # Supprime les balises de type <ref>...</ref>
    texte = re.sub(r'<ref[^>]*>.*?</ref>', '', texte, flags=re.DOTALL)
    texte = re.sub(r'<[^>]+>', '', texte) # Supprime les autres balises HTML
    texte = re.sub(r'\s+', ' ', texte) # Remplace les espaces multiples
    return texte.strip()

def main():
    # On s'assure que le dossier de destination existe
    os.makedirs("data/processed", exist_ok=True)
    output_file = "data/processed/wikipedia_sciences.txt"

    print("Chargement du dataset Wikipédia (streaming)...")
    # On charge le dataset en streaming pour ne pas tout télécharger d'un coup
    dataset = load_dataset("wikimedia/wikipedia","20231101.fr", split="train", streaming=True)

    compteur = 0
    max_articles = 100 # On limite à 100 articles pour ce premier test

    print(f"Extraction des articles scientifiques (max {max_articles})...")

    with open(output_file, "w", encoding="utf-8") as f:
        for article in tqdm(dataset, desc="Traitement"):
            if compteur >= max_articles:
                break

            titre = article["title"]
            texte = article["text"]

            # On garde l'article si le titre contient un mot-clé scientifique
            # (Tu pourras affiner ce filtre plus tard)
            if any(mot in titre.lower() for mot in MOTS_CLES_SCIENCES):
                texte_propre = nettoyer_texte(texte)
                if len(texte_propre) > 200: # On ignore les articles trop courts
                    f.write(texte_propre + "\n\n")
                    compteur += 1

    print(f"\nTerminé ! {compteur} articles sauvegardés dans {output_file}")

if (__name__ == "__main__"):
    main()