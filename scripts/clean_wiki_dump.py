import os
import re
import html

# Catégories scientifiques que l'on veut cibler
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
    "biologie",
    "chimie",
    "mécanique",
    "astronomie",
    "neurosciences",
    "génétique"
]

def extraire_titre_et_texte(article_brut):
    match_titre = re.search(r'<doc[^>]*title="([^"]*)"[^>]*>', article_brut)
    if not match_titre:
        return None, None

    titre = match_titre.group(1)

    match_texte = re.search(r'<doc[^>]*>(.*?)</doc>', article_brut, re.DOTALL)
    if not match_texte:
        return titre, ""

    texte_brut = match_texte.group(1)

    # Supprimer le titre en début de texte s'il est répété
    # (wikiextractor inclut souvent le titre dans le contenu)
    texte_brut = texte_brut.strip()
    if texte_brut.startswith(titre):
        texte_brut = texte_brut[len(titre):].lstrip()

    return titre, texte_brut

def nettoyer_texte_wiki(texte):
    texte = html.unescape(texte)
    texte = re.sub(r'Section \d+', '', texte)
    texte = re.sub(r'<[^>]+>', ' ', texte)
    texte = re.sub(r'\{\{[^}]*\}\}', ' ', texte)
    texte = re.sub(r'[ \t]+', ' ', texte)
    texte = re.sub(r'\n\s*\n', '\n', texte)
    texte = re.sub(r'(?<![A-Z])\.\s+(?=[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ])', '.\n', texte)
    texte = re.sub(r'\s+', ' ', texte)
    texte = re.sub(r'\s+([,.;:!?])', r'\1', texte)
    texte = re.sub(r"(\w)\s*'\s*(\w)", r"\1'\2", texte)

    return texte.strip()

def est_scientifique(titre, texte):
    if any(mot in titre.lower() for mot in MOTS_CLES_SCIENCES):
        return True

    return any(mot in texte[:500].lower() for mot in MOTS_CLES_SCIENCES)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_dir = os.path.join(project_root, "data", "raw", "wiki_extracted")
    output_file = os.path.join(project_root, "data", "processed", "wikipedia_sciences_dump.txt")

    if not os.path.exists(input_dir):
        print(f"Erreur : Le dossier {input_dir} n'existe pas.")
        return

    # Chercher tous les fichiers (avec ou sans extension .txt)
    fichiers = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            fichiers.append(os.path.join(root, f))

    if not fichiers:
        print(f"Aucune fichier trouvé dans {input_dir}.")
        return

    print(f"Nombre de fichiers trouvés : {len(fichiers)}")

    articles_propres = []
    total_articles = 0

    print("Nettoyage et filtrage des articles en cours...")

    for i, fichier in enumerate(fichiers):
        with open(fichier, "r", encoding="utf-8") as f:
            contenu = f.read()
            articles = re.findall(r'<doc[^>]*>.*?</doc>', contenu, re.DOTALL)

            for article in articles:
                total_articles += 1
                if len(article.strip()) < 200:
                    continue

                titre, texte_brut = extraire_titre_et_texte(article)
                if titre is None:
                    continue

                if est_scientifique(titre, texte_brut):
                    article_propre = nettoyer_texte_wiki(texte_brut)
                    if len(article_propre) > 200:
                        article_final = f"{titre}\n\n{article_propre}"
                        articles_propres.append(article_final)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(fichiers)} fichiers traités... ({len(articles_propres)} articles conservés)")

    print(f"\nTotal articles lus : {total_articles}")
    print(f"Articles scientifiques conservés : {len(articles_propres)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Écriture du fichier final dans {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for article in articles_propres:
            f.write(article + "\n\n")

    print(f"\nTerminé ! Fichier sauvegardé dans : {output_file}")

    if articles_propres:
        print("\n--- Aperçu du début du fichier ---")
        print(articles_propres[0][:800])
        print("------------------------------------")

if __name__ == "__main__":
    main()