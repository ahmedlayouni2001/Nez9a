"""
Agent Publicitaire TV Tunisien - VERSION AVEC JOURS DE SEMAINE
=================================================================
100% GRATUIT - Gemini 2.0 Flash via Google AI Studio

Usage:
    python agent_gemini.py

Dependances:
    pip install google-genai pandas python-dotenv
"""

import os
import json
import time
import re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Load .env from project root (Nez9a/.env)
load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNEES
# ─────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent.parent / "frontend" / "tdrtyf" / "public" / "data" / "audience_ramadan.csv"
DF = pd.read_csv(CSV_PATH)

JOURS_ORDRE = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

BONUS_JOUR = {
    "Lundi":    1.00,
    "Mardi":    1.00,
    "Mercredi": 1.00,
    "Jeudi":    1.05,
    "Vendredi": 1.20,
    "Samedi":   1.30,
    "Dimanche": 1.25,
}
BONUS_FERIE = 1.10


# ─────────────────────────────────────────────────────────────
# 2. MOTEUR DE SCORING
# ─────────────────────────────────────────────────────────────

def compute_scores(target_ages, target_sex, target_rev,
                   target_jours=None,
                   inclure_feries=True,
                   w_age=0.4, w_sex=0.3, w_rev=0.3):
    df = DF.copy()

    if target_jours:
        df = df[df["contexte_jour"].isin(target_jours)]
    if not inclure_feries:
        df = df[df["est_ferie"] == 0]

    if df.empty:
        return pd.DataFrame()

    GROUP = ["nom_chaine", "contexte_jour", "heure", "est_ferie"]

    total = (df.groupby(GROUP)["nb_foyers"]
               .sum().reset_index()
               .rename(columns={"nb_foyers": "total_audience"}))

    def match_sum(col, vals, label):
        mask = df[col].isin(vals) if vals else pd.Series([True] * len(df))
        return (df[mask].groupby(GROUP)["nb_foyers"]
                .sum().reset_index().rename(columns={"nb_foyers": label}))

    r = total.copy()
    for sub in [match_sum("age_groupe_pred", target_ages, "age_f"),
                match_sum("sexe_pred",       target_sex,  "sex_f"),
                match_sum("revenu_pred",     target_rev,  "rev_f")]:
        r = r.merge(sub, on=GROUP, how="left")
    r = r.fillna(0)

    T = r["total_audience"].replace(0, 1)
    r["age_pct"] = r["age_f"] / T
    r["sex_pct"] = r["sex_f"] / T
    r["rev_pct"] = r["rev_f"] / T

    r["SPA"] = w_age * r["age_pct"] + w_sex * r["sex_pct"] + w_rev * r["rev_pct"]

    r["bonus_jour"]  = r["contexte_jour"].map(BONUS_JOUR).fillna(1.0)
    r["bonus_ferie"] = r["est_ferie"].apply(lambda x: BONUS_FERIE if x == 1 else 1.0)
    r["bonus_total"] = r["bonus_jour"] * r["bonus_ferie"]

    r["SVP"]      = r["total_audience"] * r["SPA"]
    r["SVP_pond"] = r["SVP"] * r["bonus_total"]

    mx = r["SVP_pond"].max()
    r["score_norm"] = (r["SVP_pond"] / mx * 100).round(1) if mx > 0 else 0

    r["jour_ordre"] = r["contexte_jour"].map(
        {j: i for i, j in enumerate(JOURS_ORDRE)}
    )
    return (r.sort_values(["score_norm", "jour_ordre"], ascending=[False, True])
             .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────
# 3. FONCTIONS OUTILS
# ─────────────────────────────────────────────────────────────

def top_slots_fn(target_ages, target_sex, target_rev,
                 target_jours=None, inclure_feries=True, n=5, **kw):
    result = compute_scores(target_ages, target_sex, target_rev,
                            target_jours=target_jours,
                            inclure_feries=inclure_feries, **kw)
    if result.empty:
        return []
    out = []
    for _, row in result.head(n).iterrows():
        out.append({
            "chaine":          row["nom_chaine"],
            "jour":            row["contexte_jour"],
            "heure":           int(row["heure"]),
            "score":           float(row["score_norm"]),
            "audience_totale": int(row["total_audience"]),
            "foyers_cibles":   int(row["SVP"]),
            "bonus_jour":      round(float(row["bonus_jour"]), 2),
            "est_ferie":       bool(row["est_ferie"]),
            "age_pct":         round(float(row["age_pct"]) * 100, 1),
            "sex_pct":         round(float(row["sex_pct"]) * 100, 1),
            "rev_pct":         round(float(row["rev_pct"]) * 100, 1),
        })
    return out


def evaluate_slot_fn(heure, chaine, jour, target_ages, target_sex, target_rev, **kw):
    scores = compute_scores(target_ages, target_sex, target_rev, **kw)
    if scores.empty:
        return {"found": False, "message": "Aucune donnee disponible"}

    mask = scores["heure"] == heure
    if jour:
        mask = mask & (scores["contexte_jour"].str.lower() == jour.lower())
    if chaine:
        mask = mask & (scores["nom_chaine"].str.lower() == chaine.lower())

    sub = scores[mask]
    if sub.empty:
        return {"found": False, "message": "Creneau introuvable dans les donnees"}

    b = sub.iloc[0]
    return {
        "found":           True,
        "chaine":          b["nom_chaine"],
        "jour":            b["contexte_jour"],
        "heure":           int(b["heure"]),
        "score":           float(b["score_norm"]),
        "audience_totale": int(b["total_audience"]),
        "foyers_cibles":   int(b["SVP"]),
        "bonus_jour":      round(float(b["bonus_jour"]), 2),
        "est_ferie":       bool(b["est_ferie"]),
        "age_pct":         round(float(b["age_pct"]) * 100, 1),
        "sex_pct":         round(float(b["sex_pct"]) * 100, 1),
        "rev_pct":         round(float(b["rev_pct"]) * 100, 1),
    }


def audience_stats_fn(heure=None, chaine=None, jour=None):
    df = DF.copy()
    if heure:  df = df[df["heure"] == heure]
    if chaine: df = df[df["nom_chaine"].str.lower() == chaine.lower()]
    if jour:   df = df[df["contexte_jour"].str.lower() == jour.lower()]
    if df.empty:
        return {"found": False}

    by_jour = (df.groupby("contexte_jour")["nb_foyers"]
                 .sum()
                 .reindex([j for j in JOURS_ORDRE if j in df["contexte_jour"].unique()])
                 .to_dict())

    return {
        "found":        True,
        "total_foyers": int(df["nb_foyers"].sum()),
        "by_jour":      by_jour,
        "by_age":       df.groupby("age_groupe_pred")["nb_foyers"].sum().to_dict(),
        "by_sex":       df.groupby("sexe_pred")["nb_foyers"].sum().to_dict(),
        "by_rev":       df.groupby("revenu_pred")["nb_foyers"].sum().to_dict(),
        "by_chain":     df.groupby("nom_chaine")["nb_foyers"].sum()
                          .sort_values(ascending=False).to_dict(),
    }


def compare_jours_fn(target_ages, target_sex, target_rev, heure=None, chaine=None, **kw):
    scores = compute_scores(target_ages, target_sex, target_rev, **kw)
    if scores.empty:
        return []
    mask = pd.Series([True] * len(scores))
    if heure:  mask = mask & (scores["heure"] == heure)
    if chaine: mask = mask & (scores["nom_chaine"].str.lower() == chaine.lower())
    sub = scores[mask].sort_values("jour_ordre")
    out = []
    for _, row in sub.iterrows():
        out.append({
            "jour":            row["contexte_jour"],
            "chaine":          row["nom_chaine"],
            "heure":           int(row["heure"]),
            "score":           float(row["score_norm"]),
            "bonus_jour":      round(float(row["bonus_jour"]), 2),
            "audience_totale": int(row["total_audience"]),
            "foyers_cibles":   int(row["SVP"]),
        })
    return out


# ─────────────────────────────────────────────────────────────
# 4. DECLARATION DES TOOLS
# ─────────────────────────────────────────────────────────────

JOURS_DESC = "Jours cibles : Lundi, Mardi, Mercredi, Jeudi, Vendredi, Samedi, Dimanche (vide = tous)"

get_best_slots_decl = types.FunctionDeclaration(
    name="get_best_slots",
    description=(
        "Retourne les N meilleurs creneaux publicitaires TV (chaine + jour + heure) "
        "pour une cible demographique. Le score integre l'audience, le ciblage ET "
        "un bonus selon le jour de la semaine (weekend > vendredi > semaine)."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "target_ages": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Tranches d'age : 18-24, 25-34, 35-44, 45+",
            ),
            "target_sex": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Sexe : H (Homme) ou F (Femme)",
            ),
            "target_rev": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Revenu : faible, moyen, eleve",
            ),
            "target_jours": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description=JOURS_DESC,
            ),
            "inclure_feries": types.Schema(
                type=types.Type.BOOLEAN,
                description="Inclure les jours feries ? (defaut True)",
            ),
            "w_age": types.Schema(type=types.Type.NUMBER, description="Poids age (defaut 0.4)"),
            "w_sex": types.Schema(type=types.Type.NUMBER, description="Poids sexe (defaut 0.3)"),
            "w_rev": types.Schema(type=types.Type.NUMBER, description="Poids revenu (defaut 0.3)"),
            "n": types.Schema(type=types.Type.INTEGER, description="Nombre de resultats (defaut 5)"),
        },
        required=["target_ages", "target_sex", "target_rev"],
    ),
)

evaluate_slot_decl = types.FunctionDeclaration(
    name="evaluate_slot",
    description=(
        "Evalue la pertinence d'un creneau precis (chaine + jour + heure) "
        "pour une cible publicitaire, avec verdict et bonus jour."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "heure": types.Schema(type=types.Type.INTEGER, description="Heure du creneau (0-23)"),
            "jour":  types.Schema(type=types.Type.STRING,  description=JOURS_DESC),
            "chaine":types.Schema(type=types.Type.STRING,  description="Nom de la chaine (optionnel)"),
            "target_ages": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                description="Tranches d'age : 18-24, 25-34, 35-44, 45+",
            ),
            "target_sex": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                description="Sexe : H ou F",
            ),
            "target_rev": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                description="Revenu : faible, moyen, eleve",
            ),
            "w_age": types.Schema(type=types.Type.NUMBER),
            "w_sex": types.Schema(type=types.Type.NUMBER),
            "w_rev": types.Schema(type=types.Type.NUMBER),
        },
        required=["heure", "target_ages", "target_sex", "target_rev"],
    ),
)

audience_stats_decl = types.FunctionDeclaration(
    name="get_audience_stats",
    description=(
        "Statistiques d'audience TV tunisienne. "
        "Filtres optionnels : heure, chaine, jour de semaine."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "heure":  types.Schema(type=types.Type.INTEGER, description="Heure (0 = toutes)"),
            "chaine": types.Schema(type=types.Type.STRING,  description="Nom chaine (vide = toutes)"),
            "jour":   types.Schema(type=types.Type.STRING,  description=JOURS_DESC),
        },
    ),
)

compare_jours_decl = types.FunctionDeclaration(
    name="compare_jours",
    description=(
        "Compare les scores d'un meme creneau (chaine + heure) sur tous les jours "
        "de la semaine. Utile pour savoir quel jour est le meilleur pour un slot donne."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "target_ages": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                description="Tranches d'age : 18-24, 25-34, 35-44, 45+",
            ),
            "target_sex": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                description="Sexe : H ou F",
            ),
            "target_rev": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING),
                description="Revenu : faible, moyen, eleve",
            ),
            "heure":  types.Schema(type=types.Type.INTEGER, description="Heure a comparer"),
            "chaine": types.Schema(type=types.Type.STRING,  description="Chaine a comparer"),
            "w_age":  types.Schema(type=types.Type.NUMBER),
            "w_sex":  types.Schema(type=types.Type.NUMBER),
            "w_rev":  types.Schema(type=types.Type.NUMBER),
        },
        required=["target_ages", "target_sex", "target_rev"],
    ),
)

TOOLS_DECL = types.Tool(function_declarations=[
    get_best_slots_decl,
    evaluate_slot_decl,
    audience_stats_decl,
    compare_jours_decl,
])

# ─────────────────────────────────────────────────────────────
# 5. EXECUTEUR DE TOOLS
# ─────────────────────────────────────────────────────────────

def run_tool(name: str, args: dict) -> dict:
    try:
        if name == "get_best_slots":
            result = top_slots_fn(
                target_ages=args["target_ages"],
                target_sex=args["target_sex"],
                target_rev=args["target_rev"],
                target_jours=args.get("target_jours") or None,
                inclure_feries=args.get("inclure_feries", True),
                n=args.get("n", 5),
                w_age=args.get("w_age", 0.4),
                w_sex=args.get("w_sex", 0.3),
                w_rev=args.get("w_rev", 0.3),
            )
            return {"slots": result}

        elif name == "evaluate_slot":
            return evaluate_slot_fn(
                heure=args["heure"],
                jour=args.get("jour", ""),
                chaine=args.get("chaine", ""),
                target_ages=args["target_ages"],
                target_sex=args["target_sex"],
                target_rev=args["target_rev"],
                w_age=args.get("w_age", 0.4),
                w_sex=args.get("w_sex", 0.3),
                w_rev=args.get("w_rev", 0.3),
            )

        elif name == "get_audience_stats":
            h = args.get("heure", 0)
            return audience_stats_fn(
                heure=h if h and h > 0 else None,
                chaine=args.get("chaine", "") or None,
                jour=args.get("jour", "") or None,
            )

        elif name == "compare_jours":
            result = compare_jours_fn(
                target_ages=args["target_ages"],
                target_sex=args["target_sex"],
                target_rev=args["target_rev"],
                heure=args.get("heure") or None,
                chaine=args.get("chaine", "") or None,
                w_age=args.get("w_age", 0.4),
                w_sex=args.get("w_sex", 0.3),
                w_rev=args.get("w_rev", 0.3),
            )
            return {"comparaison": result}

        return {"error": f"Tool inconnu : {name}"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# 6. AGENT CONVERSATIONNEL
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en planification publicitaire TV pour le marche tunisien.
Tu as acces a des donnees d'audience reelles de 15 chaines TV tunisiennes :
Attessia TV, Hannibal TV, Nessma TV, Watania 1, Watania 2, TF1 (sat), France 24 Arabe,
MBC1, MBC4, beIN Sports 1, beIN Sports 2, Cartoon Network, Dubai TV, El Hiwar, Telvza TV.

Heures disponibles : 0h-23h.
Tranches d'age : 18-24, 25-34, 35-44, 45+.
Sexe : H (Homme), F (Femme).
Revenu : faible, moyen, eleve.
Jours : Lundi, Mardi, Mercredi, Jeudi, Vendredi, Samedi, Dimanche.

FORMULE DE SCORING :
1. SPA = 0.4*age_pct + 0.3*sex_pct + 0.3*rev_pct
2. SVP = audience_totale * SPA
3. Bonus jour :
   - Lundi-Jeudi : x1.00 a x1.05
   - Vendredi    : x1.20
   - Samedi      : x1.30
   - Dimanche    : x1.25
   - Jour ferie  : +10% supplementaire
4. Score final = SVP * bonus_jour -> normalise 0-100

REGLES :
1. Si la cible est vague -> demande age, sexe, revenu naturellement.
2. Suggestion de creneaux -> get_best_slots, affiche TOP 5 avec jour + score.
3. Evaluer un creneau precis -> evaluate_slot, verdict clair.
4. Comparer un creneau sur plusieurs jours -> compare_jours.
5. Statistiques -> get_audience_stats.
6. Explique les scores simplement.
7. Mentionne toujours le jour dans tes recommandations.
8. Reponds TOUJOURS en francais."""


def run_agent():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\nERREUR : GEMINI_API_KEY non trouvee dans Nez9a/.env")
        exit(1)

    client = genai.Client(api_key=api_key)

    MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
    ]
    current_model = [0]

    def call_with_retry(contents, max_retries=4):
        for attempt in range(max_retries):
            model = MODELS[current_model[0]]
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=[TOOLS_DECL],
                    ),
                )
            except genai_errors.ClientError as e:
                err_str = str(e)
                if "429" not in err_str:
                    raise

                if "GenerateRequestsPerDayPerProject" in err_str:
                    if current_model[0] + 1 < len(MODELS):
                        old_m = MODELS[current_model[0]]
                        current_model[0] += 1
                        new_m = MODELS[current_model[0]]
                        print(f"\n  Quota journalier epuise sur '{old_m}' -> bascule sur '{new_m}'")
                        continue
                    print("\n  Quota epuise sur tous les modeles gratuits. Attendez demain.")
                    raise

                match = re.search(r"retryDelay.*?(\d+)s", err_str)
                wait = int(match.group(1)) + 2 if match else 20
                print(f"\n  Limite temporaire -> attente {wait}s (tentative {attempt+1}/{max_retries})...")
                time.sleep(wait)

        raise RuntimeError(f"Echec apres {max_retries} tentatives.")

    print("\n" + "=" * 65)
    print("   AGENT PUBLICITAIRE TV TUNISIEN  (Gemini - GRATUIT)")
    print("   Scoring par JOUR DE SEMAINE integre")
    print("=" * 65)
    print(f"  Modele actif : {MODELS[current_model[0]]}")
    print(f"  CSV : {CSV_PATH.name}")
    print("=" * 65)
    print("Exemples :")
    print("  'Je veux faire de la pub pour une voiture'")
    print("  'Samedi 20h sur Nessma TV est bien pour du parfum ?'")
    print("  'Meilleur creneau en semaine pour les femmes 25-34 ?'")
    print("  'Compare Hannibal TV 21h tous les jours pour les jeunes'")
    print("Tapez 'quit' pour quitter.\n")

    history = []

    while True:
        user_input = input("Vous : ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)]
        ))

        print("  [Calcul en cours...]")

        while True:
            response = call_with_retry(history)

            tool_calls = []
            text_parts = []

            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(part.function_call)
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            history.append(response.candidates[0].content)

            if not tool_calls:
                print(f"\nAgent : {''.join(text_parts)}\n")
                break

            tool_results = []
            for fc in tool_calls:
                print(f"  [tool: {fc.name}...]")
                result = run_tool(fc.name, dict(fc.args))
                tool_results.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response=result,
                    )
                )

            history.append(types.Content(role="tool", parts=tool_results))


if __name__ == "__main__":
    run_agent()
