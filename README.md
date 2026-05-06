# 📡 OfflinePulse Tunisia
### Mesure & Visualisation Temps Réel de la Consommation Média Offline — TV · Radio · OOH

> *"À l'heure où chaque clic digital est mesuré à la milliseconde, les médias offline tunisiens naviguent encore à l'aveugle. OfflinePulse change la donne."*

---

## 🎯 Le Problème

En Tunisie, les médias offline — **TV, Radio et Affichage Extérieur (OOH)** — représentent la majorité des budgets publicitaires, mais leur mesure repose encore sur :

- des **panels représentatifs limités** (Sigma Conseil, études déclaratives)
- des **estimations a posteriori** livrées avec plusieurs jours, voire semaines de décalage
- une **opacité totale** sur les profils d'audience réels

Cette latence est incompatible avec les exigences d'un marché publicitaire qui se pilote désormais à l'instant T. **OfflinePulse Tunisia** sort l'offline de cette temporalité figée et le fait entrer dans celle du dashboard temps réel — sans attendre les opérateurs.

---

## 🏗️ Architecture Globale

Le projet est divisé en **trois modules indépendants et complémentaires**, chacun couvrant un canal média :

```
offlinepulse-tunisia/
│
├── 📺  mediapulse/          # MODULE 1 — Intelligence Audience TV (RPD + ML)
│   ├── dashboard/app.py
│   ├── agent/agent_gemini.py
│   └── models/
│
├── 📻  radio-scraper/       # MODULE 2 — Audience Radio Temps Réel (Icecast + LLM)
│   └── scraper_tunisia_v5.py
│
└── 🪧  billboard/           # MODULE 3 — Intelligence OOH (Computer Vision + Détection Fraude)
    ├── roi_drawer.py
    ├── test.py
    ├── cv_engine.py
    └── yolov8n.pt
```

| Module | Canal | Technologie Clé | Données Source | Latence |
|---|---|---|---|---|
| MediaPulse | 📺 Télévision | RPD · Random Forest · Gemini | IPTV Box (TT, Ooredoo, Smart TV) | ~3,5 sec |
| Radio Scraper | 📻 Radio | Icecast · Whisper · LLaMA 3.3 | Flux web des stations | ~60 sec |
| Billboard Intelligence | 🪧 OOH | YOLOv8 · Hashing perceptif | Caméra / vidéo billboard | ~1 sec |

---

## MODULE 1 — 📺 MediaPulse Tunisia
### Intelligence d'Audience TV par RPD & Machine Learning

### Le Contexte

Médiamétrie n'opère pas en Tunisie. Les annonceurs (**SFBT, Délice, Tunisair…**) investissent sans savoir qui regarde quoi, à quelle heure, quel jour. **MediaPulse comble ce vide grâce au Return Path Data.**

### Comment ça fonctionne

#### 📡 Collecte RPD en temps réel

Le **Return Path Data** est le flux d'événements envoyé automatiquement par chaque décodeur IPTV ou Smart TV à chaque interaction : allumage, changement de chaîne, extinction. **Aucun boîtier additionnel — l'infrastructure existe déjà.**

| Source | Technologie | Couverture |
|---|---|---|
| Tunisie Telecom | Box IPTV | > 800 000 foyers |
| Ooredoo | 4G Home IPTV | > 400 000 foyers |
| Smart TVs | ACR / HbbTV | Samsung, LG, TCL… |

#### 🧠 Pipeline Machine Learning

```
RPD bruts → Feature Engineering (25+ features) → 4 × Random Forest
     → Inférence démographique (âge, sexe, revenu, enfants)
          → Audience qualifiée par chaîne × heure × jour
```

Les modèles se **fine-tunent en continu** sur chaque nouvelle vague de données RPD.  
Cible : **70% de précision au lancement → 85%+ en production réelle.**

#### 📊 Données synthétiques, règles réelles

En attendant les données opérateurs, **5 000 foyers fictifs** ont été générés selon des règles précises :
- Distribution géographique par gouvernorat
- Corrélations revenu/région
- Comportements TV par profil démographique
- Habitudes horaires tunisiennes
- Calendrier (vendredi, weekend, jours fériés)

> Ces données peuvent être remplacées par des données RPD réelles **sans modifier l'architecture**.

### Ce que voit l'utilisateur

#### 1 — Métriques Live *(mis à jour toutes les 3,5 sec)*
- **Total Viewers Live** — audience totale en ce moment
- **Most Watched Now** — chaîne n°1 à l'instant T
- **Peak Zapping Hour** — heure la plus active de la journée
- Histogramme par chaîne + courbe de zapping horaire (6h–24h)

#### 2 — Audience Profile
Sélectionnez une chaîne, un jour, une heure → **3 donuts s'affichent instantanément** : répartition par **âge**, **genre** et **revenu** de l'audience à ce créneau.

#### 3 — AI Recommendation
Décrivez votre publicité en langage naturel → le système retourne le **meilleur créneau** : chaîne + heure + jour + score /100 + reach estimé en foyers.

*[📸 Capture du dashboard principal]*

*[📸 Capture de la vue Audience Profile]*

### 🤖 Le Chatbot Publicitaire (Gemini 2.0 Flash)

L'agent conversationnel répond à des questions comme :

```
"Meilleur créneau pour toucher des femmes 25-34, revenu moyen ?"
"Samedi 21h sur Attessia est-il bon pour une boisson ?"
"Compare Hannibal TV à 21h sur tous les jours de la semaine"
```

**Formule de scoring :**

```
SPA  = 0.4×age_match + 0.3×sexe_match + 0.3×revenu_match
SVP  = audience_totale × SPA
Score final = SVP × bonus_jour  →  normalisé 0–100
```

**Bonus jour :** Samedi ×1.30 · Vendredi ×1.20 · Dimanche ×1.25 · Semaine ×1.00–1.05 · Férié +10%

*[📸 Capture du chatbot]*

### Démarrage rapide — MediaPulse

```bash
git clone https://github.com/TON_USERNAME/mediapulse-tunisia.git
cd mediapulse-tunisia
pip install -r requirements.txt
echo "GEMINI_API_KEY=AIza..." > .env
streamlit run dashboard/app.py          # Dashboard
python agent/agent_gemini.py            # Chatbot CLI
```

---

## MODULE 2 — 📻 Radio Scraper Tunisia
### Mesure et Profilage Temps Réel de l'Audience Radio

### Le Contexte

En Tunisie, la mesure d'audience radio repose sur des **panels Sigma Conseil** livrés avec plusieurs jours de décalage. Ce module propose une approche complémentaire : **du temps réel via les flux web des stations.**

### Architecture du Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Parallèle (12 workers)                               │
│  ┌──────────────────────────┐    ┌──────────────────────────┐  │
│  │ probe_station × 10       │    │ fetch_radiobrowser_index │  │
│  │ ├─ resolve_stream()      │    │ (4 mirrors fallback)     │  │
│  │ ├─ liveness check        │    └──────────────────────────┘  │
│  │ ├─ probe_icecast_json    │              ↓                    │
│  │ ├─ probe_shoutcast_v2    │    attach_radiobrowser            │
│  │ ├─ probe_shoutcast_v1    │    (clickcount + trend)           │
│  │ └─ probe_html_scrape     │                                   │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Séquentielle (anti rate-limit Groq)                  │
│    ├─ capture_audio_clip(15s)      → fichier mp3 temporaire     │
│    ├─ transcribe_audio()           → Groq Whisper-large-v3-turbo│
│    ├─ extract_keywords()           → mots-clés AR/FR            │
│    └─ predict_audience()           → llama-3.3-70b → JSON       │
│                       ↓                                          │
│           breakdown âge/genre/CSP/persona                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    console (table + détails) + CSV
```

### Ce que fait chaque cycle (60 sec par défaut)

1. **Compte les auditeurs** connectés au stream web de chaque station (quand le serveur l'expose)
2. **Récupère** les métriques de popularité depuis radio-browser.info (clicks, trend 24h)
3. **Capture 15 s** du flux audio en direct
4. **Transcrit** avec Groq Whisper-large-v3-turbo (arabe ou français)
5. **Prédit le profil démographique** avec llama-3.3-70b en combinant profil éditorial statique + mots-clés transcrits
6. **Logge** une ligne CSV par station par cycle

### Stations couvertes (10 stations)

| Station | FM | Type éditorial |
|---|---|---|
| Mosaïque FM | 94.9 MHz | Privée généraliste leader |
| Shems FM | 88.7-107.6 | Privée généraliste |
| Radio Jawhara FM | 89.4 / 102.5 / 104.4 / 107.3 | Privée régionale Sahel |
| Radio IFM | 100.6 MHz | Humour & musique |
| Diwan FM | 97.3 MHz (Sfax) | Régionale Sfax |
| Express FM | 103.6 / 104.0 MHz | Économique |
| Radio Zitouna FM | various | Religieuse islamique |
| KnOOz FM | 90.6 MHz | Musique & jeux |
| Radio Nationale Tunisienne | various | Publique généraliste |
| RTCI | 98.0 MHz | Publique internationale (FR/EN/ES) |

### Exemple de sortie — Profil d'audience IA

```
🎯  AUDIENCE PROFILES — predicted by llama-3.3-70b-versatile
────────────────────────────────────────────────────────────────────

  ▶ Mosaïque FM                       4230 auditeurs            (ok)
     NOW PLAYING : Saber Rebaï — Maalem
     TRANSCRIPT  : النشرة الاقتصادية لهذا الصباح، الدينار التونسي...
     KEYWORDS    : النشرة, الاقتصادية, الدينار, التونسي
     TOPIC       : Économie tunisienne (news)
     GENDER      : ♂ 58%   ♀ 42%
     AGE         : 15-24: 8% │ 25-34: 24% │ 35-44: 32% │ 45-54: 22% │ 55+: 14%
     SOCIO       : CSP+: 35% │ Moyen: 50% │ CSP-: 15%
     PERSONA     : Cadre tunisien de 35-44 ans suivant l'actualité économique du matin
```

*[📸 Capture de la sortie console]*

### Format CSV — Consommable directement

Le fichier `audience_log.csv` contient par station et par cycle : `timestamp`, `station`, `status`, `listeners`, `peak`, `clickcount`, `clicktrend`, `now_playing`, `transcript`, `panel_topic`, `tone`, `men_pct`, `women_pct`, `age_15_24`…`age_55_plus`, `csp_plus`…`csp_minus`, `persona`.

> **Directement consommable par Excel, Power BI, Tableau, pandas.**

### Démarrage rapide — Radio Scraper

```bash
# Installation (pas de dépendances lourdes)
pip install requests groq
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Lancement standard (cycle toutes les 60 s)
python scraper_tunisia_v5.py

# Variantes
python scraper_tunisia_v5.py --watch 30          # cycle toutes les 30 s
python scraper_tunisia_v5.py --no-ai             # listener count seul
python scraper_tunisia_v5.py --watch 0 --json    # one-shot JSON pour dashboard
```

---

## MODULE 3 — 🪧 Billboard Intelligence
### Comptage Audience & Détection de Fraude OOH par Computer Vision

### Le Contexte

Les panneaux publicitaires extérieurs (OOH) sont totalement aveugles : aucune donnée sur combien de personnes passent devant, ni si l'affichage a réellement été diffusé. **Billboard Intelligence répond à ces deux questions en temps réel, à partir d'une simple caméra.**

### Ce que fait le module

À partir d'un flux vidéo (fichier ou caméra live) :

1. **Compte les personnes** — YOLOv8 détecte et traque chaque personne frame par frame. Chaque nouvel ID de tracking ajoute 1 au compteur cumulatif, sans double-comptage.
2. **Détecte la fraude billboard** — Prend un hash perceptif de la zone billboard à la frame 30 (référence). Toutes les 2 secondes, re-hashe la même zone et compare. Si la différence visuelle dépasse le seuil → fraude déclarée avec timestamp exact.
3. **Produit une table live dans le terminal** — Toutes les secondes : timestamp, personnes vues, statut fraude.
4. **Génère un rapport JSON de session** — `session_report.json` avec le log complet seconde par seconde + résumé final.
5. **Envoie les métriques au backend** — Via WebSocket (`ws://localhost:8000/ws/cv`) pour affichage dashboard en temps réel.

### Comment fonctionne la détection de fraude

```
Frame 30 → hash perceptif de la ROI billboard = RÉFÉRENCE
     ↓
Toutes les 2 sec → nouveau hash → distance de Hamming avec référence
     ↓
distance > seuil (15) → FRAUD + timestamp verrouillé
```

> Un rectangle **vert** autour du billboard devient **rouge** dès qu'une fraude est détectée — visible directement dans la fenêtre vidéo.

### Exemple de sortie live

```
+---------------------+---------+--------------+
| Timestamp           | Persons | Fraud Status |
+---------------------+---------+--------------+
| 2026-05-02 14:32:01 |       3 | NOTHING      |
| 2026-05-02 14:32:55 |       7 | FRAUD        |
+---------------------+---------+--------------+

[FraudDetector] *** FRAUD DETECTED at 2026-05-02 14:32:55 ***
```

### Rapport JSON de session

```json
{
  "summary": {
    "video_start": "2026-05-02 14:32:00",
    "video_end":   "2026-05-02 14:33:45",
    "total_people": 47,
    "fraud_detected_at": "2026-05-02 14:32:55"
  },
  "per_second": [
    { "timestamp": "2026-05-02 14:32:01", "persons": 3, "fraud_status": "nothing" },
    ...
  ]
}
```

*[📸 Capture du dashboard billboard]*

*[📸 Capture de la détection de fraude (rectangle rouge)]*

### Paramètres configurables

| Constante | Défaut | Rôle |
|---|---|---|
| `YOLO_CONF_THRESHOLD` | `0.4` | Seuil de confiance détection |
| `PROCESS_EVERY_N_FRAMES` | `3` | 1 frame traitée sur N (vitesse/précision) |
| `FRAUD_CHECK_INTERVAL_SEC` | `2` | Fréquence de re-hash billboard |
| `FRAUD_HASH_THRESHOLD` | `15` | Distance de Hamming seuil fraude |
| `PUSH_INTERVAL_SEC` | `1.0` | Fréquence push WebSocket |

### Démarrage rapide — Billboard Intelligence

```bash
cd Billboard
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Étape 1 : définir la zone billboard (ROI)
python roi_drawer.py
# → cliquer-glisser sur la zone, appuyer sur 'p' pour sauvegarder

# Étape 2 : lancer le pipeline
python test.py --source path/to/video.mp4
python test.py --source 0                         # webcam live
python test.py --source 0 --no-window             # mode headless
```

---

## 💡 Points de Force du Projet

### 🔄 Temps Réel Unifié sur 3 Canaux
Premier système en Tunisie (et au Maghreb) à mesurer simultanément TV, Radio et OOH avec une latence inférieure à la minute — sans aucun panel déclaratif.

### 🧠 IA Multi-Couches
- **Machine Learning supervisé** (Random Forest, 25+ features) pour la TV
- **LLM conversationnel** (Gemini 2.0 Flash) pour les recommandations publicitaires
- **Speech-to-text + profilage LLM** (Whisper + LLaMA 3.3 70B) pour la radio
- **Computer Vision** (YOLOv8) pour le comptage OOH

### 📡 Infrastructure Existante, Zéro Matériel Additionnel
Le RPD TV exploite les flux déjà émis par les box IPTV. Le scraper radio exploite les flux Icecast/SHOUTcast existants. Billboard utilise n'importe quelle caméra. **Aucun capteur à déployer.**

### 🔐 Détection de Fraude OOH Inédite
La vérification d'intégrité par hash perceptif du contenu billboard est unique sur le marché tunisien — elle garantit aux annonceurs que leur publicité a bien été diffusée.

### 🗺️ Architecture Modulaire & Scalable
Chaque module est indépendant, interconnectable via WebSocket/CSV/API. Ajout d'une station radio : 5 lignes de config. Extension Maghreb : la logique de probe est générique.

### 🛡️ Conformité & Éthique
Données agrégées et anonymisées — conformes à la **Loi n°2004-63** et aux **standards INTT**. Profils démographiques inférés, jamais de données personnelles collectées.

---

## 💼 Modèle Économique (MediaPulse TV)

| Plan | Prix/mois | Inclus |
|---|---|---|
| Starter | 500 DT | 2 chaînes, chatbot, 3 mois d'historique |
| Business | 1 500 DT | 10 chaînes, alertes, export CSV |
| Enterprise | 4 000 DT | Tout + API + rapports PDF auto |

**Roadmap :**
- **An 1** → Tunisie : POC + 1 opérateur IPTV · Radio : 10 stations · OOH : 5 panneaux pilotes
- **An 2** → 15 abonnés annonceurs · 250–400K DT/an
- **An 3** → Extension Maghreb (Maroc, Algérie) · 1–2M DT/an

---

## 🛠️ Stack Technique Complète

| Couche | Technologie |
|---|---|
| **ML / Inférence TV** | Python · Scikit-learn · Pandas · Random Forest |
| **Dashboard TV** | Streamlit · Plotly |
| **Agent TV** | Google Gemini 2.0 Flash |
| **Scraping Radio** | Requests · Icecast/SHOUTcast probes · radio-browser.info |
| **Transcription Radio** | Groq Whisper-large-v3-turbo |
| **Profilage LLM** | Groq LLaMA-3.3-70B-versatile |
| **Computer Vision OOH** | YOLOv8 nano (Ultralytics) · OpenCV · ImageHash |
| **Backend OOH** | FastAPI · WebSocket |
| **Data** | CSV · JSON · RPD/IPTV · Flux web radio |
| **Conformité** | Loi n°2004-63 · Standards INTT · Données anonymisées |

---

## 🗂️ Structure du Dépôt

```
offlinepulse-tunisia/
│
├── mediapulse/
│   ├── dashboard/
│   │   └── app.py                  # Dashboard Streamlit TV
│   ├── agent/
│   │   └── agent_gemini.py         # Chatbot publicitaire Gemini
│   ├── models/                     # Modèles Random Forest entraînés
│   ├── data/                       # 5 000 foyers synthétiques
│   └── requirements.txt
│
├── radio-scraper/
│   ├── scraper_tunisia_v5.py       # Pipeline complet radio
│   ├── audience_log.csv            # Log généré (output)
│   └── requirements.txt
│
├── billboard/
│   ├── roi_drawer.py               # Outil interactif de définition ROI
│   ├── test.py                     # Pipeline CV principal
│   ├── cv_engine.py                # Moteur complet (avec simulation overlay)
│   ├── roi_config.txt              # Coordonnées ROI sauvegardées
│   ├── session_report.json         # Rapport de session (output)
│   ├── yolov8n.pt                  # Poids YOLOv8 nano (téléchargés auto)
│   └── requirements.txt
│
└── README.md                       # Ce fichier
```

---

## ⚡ Démarrage Rapide — Lancer les 3 Modules

```bash
# Cloner le projet
git clone https://github.com/TON_USERNAME/offlinepulse-tunisia.git
cd offlinepulse-tunisia

# === MODULE TV ===
cd mediapulse
pip install -r requirements.txt
echo "GEMINI_API_KEY=AIza..." > .env
streamlit run dashboard/app.py

# === MODULE RADIO (dans un autre terminal) ===
cd radio-scraper
pip install requests groq
export GROQ_API_KEY="gsk_..."
python scraper_tunisia_v5.py

# === MODULE OOH (dans un autre terminal) ===
cd billboard
pip install -r requirements.txt
python roi_drawer.py        # Définir la zone billboard
python test.py --source 0  # Lancer sur webcam
```

---

## 🔭 Roadmap Globale

- [ ] **Unification dashboard** — un seul écran agrégeant TV + Radio + OOH en temps réel
- [ ] **API unifiée** — endpoint `/api/stats` exposant les 3 modules pour intégrations tierces
- [ ] **Stockage time-series** — migration CSV → InfluxDB / TimescaleDB pour analyses temporelles
- [ ] **Détection transitions radio** — identifier automatiquement talk → musique → publicité
- [ ] **Partenariat opérateurs** — intégration données RPD réelles (Tunisie Telecom, Ooredoo)
- [ ] **Calibration LLM radio** — validation des prédictions contre vagues Sigma Conseil
- [ ] **Multi-pays** — extension Maroc, Algérie (architecture générique)
- [ ] **Billboard réseau** — gestion multi-panneaux avec centralisation des alertes fraude

---

## 📜 Conformité & Données

- Toutes les données TV sont **agrégées au niveau foyer**, jamais individualisées
- Les profils radio sont des **prédictions statistiques**, non des données personnelles collectées
- Le comptage OOH repose sur des **flux vidéo anonymisés**, sans reconnaissance faciale
- **Loi n°2004-63** (protection des données personnelles en Tunisie) · **Standards INTT**

---

## 📄 License

MIT — utilisable librement, attribution appréciée.

---

*OfflinePulse Tunisia — Hackathon 2026 · Mesure et visualisation temps réel de la consommation média offline*
