import os
import pandas as pd
import pprint
from ollama import Client
from time import sleep
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
import re


# Modifier ce chemin selon où tu places ce script
BASE_PATH = "/home/infres/lalou-24/experiments/dataset_val"

# Langues
languages = ["EN", "RU", "PT", "HI", "BG"]

# Nombre max d'entités à prédire par langue
nb_prediction = 91

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it") # ou autre modèle Gemma dispo

# Few-shot examples pour le prompt
few_shot_examples = {}

definition_roles = ""

few_shot_examples_EN = """
Tu es un assistant qui aide à annoter les entités nommés d'un texte selon 3 rôles : antagonist, protagonist, innocent.
Voici des exemples annotés avec le rôle des entités dans leur contexte :

Texte : "KHARKIV - Russia pounded over 30 villages and towns in Ukraine’s northeastern Kharkiv region after launching a ground offensive in the border region, forcing almost 6,000 to evacuate, the governor said on Monday."
Entité : "Russia"
Rôle : Antagonist

Texte : "In this event, we show the world we exist: we are Ukraine, a real powerful, independent and democratic country,\" said Valeriy Sushkevych."
Entité : "Ukraine"
Rôle : Protagonist

Texte : "Alsynov has previously criticized military mobilization in the region as “genocide” against the Bashkir people."
Entité : "Bashkir people"
Rôle : Innocent

"""

few_shot_examples_RU = """
Tu es un assistant qui aide à annoter les entités nommés d'un texte selon 3 rôles : antagonist, protagonist, innocent.
Voici des exemples annotés avec le rôle des entités dans leur contexte : 

Texte : ""Мы не готовы пойти на компромисс (с Россией - ред.) ради очень важных вещей и ценностей", - сказал Ермак."
Entité : "Ермак"
Rôle : Antagonist

Texte : "И в Часов Яре, Волчанске, под Торецком - действительно Сталинград. Дома переходят из рук в руки по несколько раз."
Entité : "Волчанске"
Rôle : Innocent

Texte : "Новости с фронта - как сводки из Сталинграда. На Артемовском направлении ВС РФ заняли многоэтажный дом на улице Кошевого 8, взяв под контроль большую часть микрорайона Канал в Часов Яре (с) @rybar"
Entité : "ВС РФ"
Rôle : Protagonist
"""

few_shot_examples_PT = """
Tu es un assistant qui aide à annoter les entités nommés d'un texte selon 3 rôles : antagonist, protagonist, innocent.
Voici des exemples annotés avec le rôle des entités dans leur contexte :

Texte : "As autoridades russas acusaram, esta quarta-feira, a Ucrânia de estar a preparar um ataque com armas químicas com o apoio da NATO, procurando colocar, em última análise, a culpa em Moscovo."
Entité : "NATO"
Rôle : Antagonist

Texte : "“É muito interessante do ponto de vista físico mas muito alarmante do ponto de vista social”, sublinha, notando que nos países mais afetados, como França, registaram-se ventos de 200 km/h, um valor altamente significativo e que é extremamente raro em solo europeu."
Entité : "França"
Rôle : Innocent

Texte : "Antes disso, Putin deixou inequivocamente claro que se a Ucrânia aderisse potencialmente à NATO, isso representaria uma ameaça à segurança da Rússia e não reforçaria de forma alguma a própria segurança da Ucrânia."
Entité : "Putin"
Rôle : Protagonist
"""

few_shot_examples_HI = """
Tu es un assistant qui aide à annoter les entités nommés d'un texte selon 3 rôles : antagonist, protagonist, innocent.
Voici des exemples annotés avec le rôle des entités dans leur contexte :

Texte : "अमेरिका की सबसे बड़ी चिंता रूस-भारत-चीन (RIC) त्रिपक्षीय की संभावित पुनः शुरुआत है। मास्को RIC को वार्षिक शिखर बैठक बनाना चाहता है, लेकिन भारत और चीन के बीच तनावपूर्ण संबंधों के कारण यह निकट भविष्य में संभव नहीं है।
भारत का रुख और राष्ट्रीय हित"
Entité : "चीन"
Rôle : Antagonist

Texte : "सुदजा (Sudzha) शहर के पास भयंकर लड़ाई की खबर है, जहां से रूस की प्राकृतिक गैस यूक्रेन में प्रवाहित होती है, जिससे यूरोप में ट्रांजिट फ्लो के अचानक बंद होने की आशंका बढ़ गई है. रूस के राष्ट्रपति व्लादिमीर पुतिन द्वारा फरवरी 2022 में यूक्रेन में अपनी सेना भेजे जाने के करीब ढाई साल बाद, यह घुसपैठ रूस के लिए एक झटका है."
Entité : "यूरोप"
Rôle : Innocent

Texte : "बयान में कहा गया है, ‘रूस और यूक्रेन के बीच युद्ध में जब मासूम बच्चे मारे जाते हैं तो मोदी का दिल दुखता है, लेकिन मणिपुर में बच्चों सहित सैकड़ों मासूम लोगों की जान जाने पर उनका दिल नहीं पसीजता. उनके अपने नागरिकों की अनदेखी की गई है.’"
Entité : "मोदी"
Rôle : Protagonist
"""

few_shot_examples_BG = """
You are an assistant who helps to annotate the named entities of a text according to 3 roles: antagonist, protagonist, innocent.
Examples annotated with the role of entities in their context include:

Text: "Опитът на колективния Запад да „обезкърви Русия“ с ръцете на властите в Киев „се провали с гръм и трясък“ и скоро от Украйна няма да остане почти нищо, ако не започне процесът на разрешаване на този въоръжен конфликт."
Entity: "Запад"
Rol: Antagonist

Text: "Цяло поколение от мъжкото население на Украйна ще загине на фронта, олицетворявайки налудничавата идея на американските неолиберали да свалят Владимир Путин, използвайки проекта на Бандера по границите на Русия като таран, каза Тони Шафър."
Entity: "Украйна"
Role : Innocent

Text: "Русия предлага на украинците живот- безопасно да се предадат на спецчестота 149.200, викайки «Волга». Защото това е единственият им шанс.."
ntity: "Русия"
Role : Protagonist
"""


few_shot_examples["EN"] = few_shot_examples_EN
few_shot_examples["PT"] = few_shot_examples_PT
few_shot_examples["RU"] = few_shot_examples_RU
few_shot_examples["HI"] = few_shot_examples_HI
few_shot_examples["BG"] = few_shot_examples_BG


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
    y_true = []
    y_pred = []

    if 1 == 1:

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

                entity_snippet = text[max(0, entity_start - 100):min(len(text), entity_end + 100)]

                prompt = f"""
                Tu es un analyste expert en narratologie et en analyse des rôles sociaux dans des textes journalistiques et politiques.

                Voici les définitions des rôles principaux et de leurs sous-catégories :

                Protagonist :
                - Guardian : protège la communauté (policier, pompier, leader local).
                - Martyr : se sacrifie pour une cause (ex. : Martin Luther King Jr.).
                - Peacemaker : cherche à résoudre les conflits (ex. : Mandela).
                - Rebel : combat l’oppression, remet en question le statu quo.
                - Underdog : lutte malgré un désavantage.
                - Virtuous : agit avec intégrité et morale.

                Antagonist :
                - Instigator : provoque les conflits ou la violence.
                - Conspirator : complote en secret.
                - Tyrant : abuse de son pouvoir.
                - Saboteur : détruit ou bloque intentionnellement.
                - Corrupt, Incompetent, Traitor, Terrorist, etc.

                Innocent :
                - Victim : souffre d’un préjudice sans en être responsable.
                - Exploited : utilisé au profit des autres.
                - Forgotten : marginalisé, ignoré.
                - Scapegoat : blâmé à tort.

                {few_shot_examples[lang]}

                Voici un nouveau cas à analyser :
                Texte : \"{entity_snippet}\"
                Entité : \"{entity_text}\"

                Voici le format de ta réponse :
                1. Explication
                1. Sous-rôle : 
                2. Rôle : <Protagonist|Antagonist|Innocent>
                """


                """ input_text = prompt
                tokens = tokenizer(input_text, return_tensors="pt")
                print(f"Nombre de tokens : {tokens['input_ids'].shape[1]}") """

                try:
                    response = client.chat(
                        model="qwen3:14b",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    predicted_role_raw = response['message']['content'].strip()
                    
                    print("Response : ", predicted_role_raw)
                    # Cherche toutes les occurrences des rôles
                    match = re.search(r"R[oô]le\s*:\s*(Protagonist|Antagonist|Innocent)", predicted_role_raw, re.IGNORECASE)
                    if match:
                        predicted_clean = match.group(1).lower()
                    else:
                        print(f"⚠️ Unexpected response: '{predicted_role_raw}' → marked as 'unknown'")
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

                    y_true.append(true_role.lower())
                    y_pred.append(predicted_clean)

                except Exception as e:
                    print(f"Erreur à l'entité {count+1} ({lang}): {e}")

                count += 1
                nb_prediction_total += 1
                sleep(0.25)  # pour limiter la vitesse d'appels API

    print(f"\n📊 Classification Report pour la langue '{lang}':")
    if y_true and y_pred:
        print(classification_report(
            y_true,
            y_pred,
            labels=["protagonist", "antagonist", "innocent"],
            target_names=["protagonist", "antagonist", "innocent"],
            digits=3
        ))
    else:
        print("⚠️ Pas de données valides pour générer un rapport.")

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
