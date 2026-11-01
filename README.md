# tiny-french-gpt

Petit modèle de langage (GPT decoder-only, ~18.9M de paramètres) entraîné from scratch sur un corpus français (Wikipédia scientifique).

## Objectif

Construire de zéro un chatbot miniature : tokenizer BPE, pré-entraînement, SFT sur dialogues, génération.

## Pipeline

1. **Corpus** : dump Wikipédia français (`frwiki-latest-pages-articles.xml.bz2`, 6.8 Go)
2. **Extraction** : `wikiextractor` avec l'option `--links` pour préserver le texte des liens
3. **Nettoyage** : filtrage des articles scientifiques (177 891 articles), suppression des balises HTML, conservation des titres, correction de la ponctuation
4. **Tokenizer** : BPE entraîné sur le corpus (12 000 tokens)
5. **Modèle** : Transformer decoder-only (8 couches, 384 dimensions, 6 têtes d'attention)
6. **Entraînement** : pré-entraînement sur 5 millions de tokens (loss finale : ~2.5)

## Résultats du pré-entraînement

Après ~28 000 steps (loss ~2.5), le modèle génère du français grammaticalement correct avec un vocabulaire scientifique pertinent.

### Exemples de génération

**Prompt : "La science"**
> La science a pris la décision de faire le jour sans avoir une expérience de science expérimentale. Les sciences médicales et les sciences médicales ont tenté d'établir la chimie expérimentale et la médecine (en grec "bioos", «protection»), qui a été l'occasion de la conférence de Jansky

**Prompt : "Un ordinateur"**
> Un ordinateur quantique et de son langage de programmation quantique. Il est également possible de définir la notion de fonction en plusieurs groupes de combinaisons linéaires. Un modèle de calcul mécanique quantique est toujours un modèle très élaboré, dans le cadre d'un modèle de l'onde et de la théorie des champs, soit

**Prompt : "Le cerveau"**
> Le cerveau est un phénomène de synthèse des neurosciences. L'importance des neurosciences biologiques dans la nature est l'étude du fonctionnement du métabolisme des êtres vivants. Les neurosciences biologiques sont considérés comme étant des disciplines comme la science du processus et de la physiologie (Implification, neurologie, économie

**Prompt : "Une équation"**
> Une équation de la dynamique est de type "objet" et de volume "impulsion". De manière intéressante par ailleurs la dynamique de ces équations définit une courbe, un champ est de type "objet" qui s'y associe à chaque point formula_3. Si l'on est de type "objet"

**Prompt : "La matière"**
> La matière désigne donc un système constitué de particules de particules de masse et de masse, dont la masse est de la «charge» et du «blanche» (du grec "kka" +) semblable à la masse. Il existe aussi des particules de masse qui sont en interaction avec le champ électromagnétique qu

### Ce que le modèle a appris

- La grammaire française (accords, conjugaisons, structure des phrases)
- Le vocabulaire scientifique (neurosciences, quantique, équation, particules, masse, champ)
- Les enchaînements typiques de Wikipédia (définitions, exemples, références)

### Ce qu'il n'a pas encore appris

- La cohérence à long terme (répétitions, incohérences)
- La factualité (il invente parfois des choses)
- La capacité à répondre à des questions (il complète du texte, il ne dialogue pas)

## Structure

- `data/` : corpus brut et nettoyé
- `tokenizer/` : entraînement du tokenizer
- `model/` : architecture Transformer
- `training/` : pré-entraînement et SFT
- `scripts/` : préparation des données et génération
- `checkpoints/` : poids du modèle sauvegardés

## Prochaines étapes

- [ ] SFT sur des paires (question, réponse) pour transformer le modèle en chatbot
- [ ] Ajout de Project Gutenberg pour diversifier le style (littéraire vs encyclopédique)
- [ ] Augmentation du corpus (plus de tokens, plus d'epochs)

## Statut

Pré-entraînement terminé. Le modèle génère du texte cohérent. En cours de développement pour le SFT.