import os
import re

def nettoyer_article(texte):
    """
    Nettoie le texte d'un article Wikipédia :
    - Supprime les sections de fin (Bibliographie, Notes, Liens externes, etc.)
    - Supprime les résidus de mise en forme
    - Formate les paragraphes avec des sauts de lignes
    """
    # 1. Supprimer les sections de fin
    pattern_sections_fin = re.compile(
        r'\s*(Notes et références|Bibliographie|Liens externes|Articles connexes|Voir aussi|Références|Sources)\s*',
        re.IGNORECASE
    )
    match = pattern_sections_fin.search(texte)
    if match:
        texte = texte[:match.start()]

    # 2. Supprimer les résidus de type "(en anglais)"
    texte = re.sub(r'\(en anglais\s*\)', '', texte, flags=re.IGNORECASE)
    texte = re.sub(r'\(en anglais [^)]*\)', '', texte, flags=re.IGNORECASE)

    # 3. Nettoyer les espaces
    texte = re.sub(r'[ \t]+', ' ', texte)
    texte = re.sub(r'\n\s*\n', '\n', texte)

    # 4. Ajouter un saut de ligne après les phrases
    texte = re.sub(r'(?<![A-Z])\.\s+(?=[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ])', '.\n', texte)

    # 5. Séparer les titres de section du corps du texte
    titres_sections = [
        "Étymologie",
        "Histoire",
        "Description",
        "Caractéristiques",
        "Définitions",
        "Utilisation",
        "Utilisations",
        "Exemples",
        "Applications",
        "Notes",
        "Annexes"
    ]
    for titre in titres_sections:
        texte = re.sub(rf'\s+({titre})\s+', rf'\n\1\n', texte)

    return texte.strip()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_file = os.path.join(project_root, "data", "processed", "wikipedia_sciences.txt")
    output_file = os.path.join(project_root, "data", "processed", "wikipedia_sciences_clean.txt")

    if not(os.path.exists(input_file)):
        print(f"Erreur : Le fichier {input_file} n'existe pas.")
        return

    print("Lecture du fichier brut...")
    with open(input_file, "r", encoding="utf-8") as f:
        contenu_brut = f.read()

    articles = contenu_brut.split("\n\n")
    print(f"Nombre d'articles : trouvés {len(articles)}")

    articles_propres = []

    print("Nettoyage des articles en cours...")
    for i, article in enumerate(articles):
        if len(article.strip()) < 100:
            continue

        article_propre = nettoyer_article(article)

        if len(article_propre) > 200:
            articles_propres.append(article_propre)

        if (i + 1) % 1000 == 0:
            print(f" {i + 1} articles traités...")

    print(f"Écriture du fichier nettoyé ({len(articles_propres)} articles conservés)...")
    with open(output_file, "w", encoding="utf-8") as f:
        for article in articles_propres:
            f.write(article + "\n\n")

    print(f"\nTerminé ! Fichier sauvegardé dans : {output_file}")

    with open(output_file, "r", encoding="utf-8") as f:
        apercu = f.read(500)
        print("\n--- Aperçu du début du fichier nettoyé ---")
        print(apercu)
        print("--------------------------------------------")

if __name__ == "__main__":
    main()

