import os
import pandas as pd
import pprint
from ollama import Client
from time import sleep
from sklearn.metrics import classification_report

# Modifier ce chemin selon où tu places ce script
# LOCAL
#BASE_PATH = "C:/Users/ine28/OneDrive/Documents/Mastère_IA/Projet_Fil_Rouge/dataset_val"
# GPU TELECOM PARIS
BASE_PATH = "/home/infres/lalou-24/experiments/dataset_val"

languages = ["EN", "RU", "PT", "HI", "BG"]

# Nombre max d'entités à prédire par langue
nb_prediction = 91

# Few-shot examples pour le prompt
few_shot_examples = """
Tu es un assistant qui aide à annoter les entités nommés d'un texte selon 3 rôles : antagonist, protagonist, innocent.
Voici des exemples annotés avec le rôle des entités dans leur contexte :

Texte : "KHARKIV - Russia pounded over 30 villages and towns in Ukraine’s northeastern Kharkiv region after launching a ground offensive in the border region, forcing almost 6,000 to evacuate, the governor said on Monday."
Entité : "Russia"
Rôle : Antagonist

Texte : "Alsynov has previously criticized military mobilization in the region as “genocide” against the Bashkir people."
Entité : "Bashkir people"
Rôle : Innocent

Texte : "In this event, we show the world we exist: we are Ukraine, a real powerful, independent and democratic country,\" said Valeriy Sushkevych."
Entité : "Ukraine"
Rôle : Protagonist
"""

# Chargement et préparation des données
dataset_global = {}

for lang in languages:
    dataset = []
    print(f"Traitement de la langue : {lang}")
    lang_path = os.path.join(BASE_PATH, lang)

    mentions_path = os.path.join(lang_path, "subtask-1-annotations.txt")
    if not os.path.exists(mentions_path):
        print(f"Fichier d’annotations manquant pour {lang}")
        continue

    documents_path = os.path.join(lang_path, "subtask-1-documents")
    if not os.path.exists(documents_path):
        print(f"Dossier de documents manquant pour {lang}")
        continue

    annotations = pd.read_csv(
        mentions_path,
        sep="\t",
        header=None,
        engine='python',
        usecols=[0,1,2,3,4],
        names=["file", "entity", "start", "end", "true_role"]
    )

    text_cache = {}
    for file_name in annotations["file"].unique():
        text_file_path = os.path.join(documents_path, file_name)
        if os.path.exists(text_file_path):
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_cache[file_name] = f.read()
        else:
            print(f"Texte introuvable : {text_file_path}")

    grouped = annotations.groupby("file")

    for file_name, group in grouped:
        if file_name not in text_cache:
            continue
        text = text_cache[file_name]
        entities = []
        for _, row in group.iterrows():
            entities.append({
                "entity": row["entity"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "true_role": row["true_role"]
            })
        dataset.append({
            "file": file_name,
            "text": text,
            "entities": entities
        })
    dataset_global[lang] = dataset

# Initialisation client Ollama
client = Client(host='http://localhost:11434')

results = []
nb_prediction_total = 0

for lang in languages:
    print(f"\n--- Prédictions pour les entités en {lang} ---\n")
    count = 0

    for doc in dataset_global.get(lang, []):
        if count >= nb_prediction:
            break

        text = doc["text"]
        for entity in doc["entities"]:
            if count >= nb_prediction:
                break

            entity_text = entity["entity"]
            entity_start = entity["start"]
            entity_end = entity["end"]
            true_role = entity["true_role"]

            entity_snippet = text[max(0, entity_start - 150):min(len(text), entity_end + 150)]

            prompt = f"""{few_shot_examples}

Voici un nouveau cas à analyser :
Texte : \"{entity_snippet}\"
Entité : \"{entity_text}\" (position : {entity_start}-{entity_end} dans le texte complet)
Quel est le rôle de cette entité dans ce contexte ?

Répond uniquement par l'un des mots suivants : Protagonist, Antagonist, Innocent. 
Rôle :"""

            try:
                response = client.chat(
                    model="gemma3:12b",
                    messages=[{"role": "user", "content": prompt}]
                )
                predicted_role_raw = response['message']['content'].strip()
                valid_roles = {"protagonist", "antagonist", "innocent"}
                predicted_clean = predicted_role_raw.lower().split()[0]

                if predicted_clean not in valid_roles:
                    print(f"⚠️ Réponse inattendue : '{predicted_role_raw}' → ignorée ou marquée comme 'unknown'")
                    predicted_clean = "unknown"

                print(f"[{lang} - {count+1}] Entité: {entity_text} → Rôle prédit: {predicted_clean} | Rôle réel: {true_role}")

                results.append({
                    "lang": lang,
                    "file": doc["file"],
                    "entity": entity_text,
                    "start": entity_start,
                    "end": entity_end,
                    "true_role": true_role.lower(),
                    "predicted_role": predicted_clean
                })

            except Exception as e:
                print(f"Erreur à l'entité {count+1} ({lang}): {e}")

            count += 1
            nb_prediction_total += 1
            sleep(0.5)  # pour limiter la vitesse d'appels API

print(f"\n✅ Total d'entités traitées : {nb_prediction_total}")

# Sauvegarde résultats
results_df = pd.DataFrame(results)
results_df.to_csv("predictions_vs_truth.csv", index=False, encoding="utf-8")
print("\n📄 Résultats enregistrés dans predictions_vs_truth.csv")

# Rapport global
print("\n--- Rapport de classification global ---")
print(classification_report(
    results_df["true_role"],
    results_df["predicted_role"],
    digits=4
))

# Rapport par langue
for lang in languages:
    lang_df = results_df[results_df["lang"] == lang]
    if len(lang_df) == 0:
        continue
    print(f"\n--- Rapport pour {lang} ---")
    print(classification_report(
        lang_df["true_role"],
        lang_df["predicted_role"],
        digits=4
    ))
