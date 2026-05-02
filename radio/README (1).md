# scraper_tunisia_v5

> **Mesure et profilage temps réel de l'audience des radios tunisiennes**
> Listener count Icecast/SHOUTcast + transcription audio Whisper + profilage démographique LLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Sortie console](#sortie-console)
- [Format CSV](#format-csv)
- [Stations couvertes](#stations-couvertes)
- [Comprendre les statuts](#comprendre-les-statuts)
- [Paramètres avancés](#paramètres-avancés)
- [Limitations honnêtes](#limitations-honnêtes)
- [Dépannage](#dépannage)
- [Coûts & quotas](#coûts--quotas)
- [Roadmap](#roadmap)

---

## Vue d'ensemble

`scraper_tunisia_v5.py` répond à un constat simple : en Tunisie, la mesure d'audience radio repose sur des panels Sigma Conseil livrés avec plusieurs jours de décalage. Ce script propose une approche complémentaire — **du temps réel via les flux web** des stations.

Pour chaque cycle (par défaut 60 s), il :

1. **Compte les auditeurs** connectés au stream web de chaque station (quand le serveur l'expose)
2. **Récupère** les métriques de popularité depuis radio-browser.info (clicks, trend 24h)
3. **Capture 15 s** du flux audio en direct
4. **Transcrit** avec Groq Whisper-large-v3-turbo (arabe ou français)
5. **Prédit le profil démographique** des auditeurs avec llama-3.3-70b en combinant :
   - Le profil éditorial statique de la station
   - Les mots-clés transcrits du contenu en cours
6. **Affiche** un tableau + détails IA dans la console
7. **Logge** une ligne CSV par station par cycle

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Parallèle (12 workers)                               │
│  ┌──────────────────────────┐    ┌──────────────────────────┐  │
│  │ probe_station × 10       │    │ fetch_radiobrowser_index │  │
│  │ ├─ resolve_stream()      │    │ (4 mirrors fallback)     │  │
│  │ ├─ liveness check        │    └──────────────────────────┘  │
│  │ ├─ probe_icecast_json    │              │                    │
│  │ ├─ probe_shoutcast_v2    │              ▼                    │
│  │ ├─ probe_shoutcast_v1    │    attach_radiobrowser            │
│  │ └─ probe_html_scrape     │    (clickcount + trend)           │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Séquentielle (anti rate-limit Groq)                  │
│  Pour chaque station avec status = ok ou locked :               │
│    ├─ capture_audio_clip(15s)      → fichier mp3 temporaire     │
│    ├─ transcribe_audio()           → Groq Whisper-large-v3-turbo│
│    ├─ extract_keywords()           → mots-clés AR/FR            │
│    └─ predict_audience()           → llama-3.3-70b → JSON       │
│                                       ↓                          │
│                       breakdown âge/genre/CSP/persona            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                console (table + détails) + CSV
```

---

## Installation

### Prérequis

- Python 3.10 ou plus récent
- Un compte Groq (gratuit) → [console.groq.com](https://console.groq.com/) pour récupérer une clé API

### Dépendances

```bash
pip install requests groq
```

C'est tout. Aucune dépendance lourde, pas de ffmpeg requis (Whisper accepte le mp3 directement).

---

## Configuration

### Variable d'environnement Groq

```bash
# Linux/macOS
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows PowerShell
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows CMD
set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Constantes éditables en haut du fichier

| Variable | Défaut | Description |
|---|---|---|
| `POLL_INTERVAL` | `60` | Secondes entre 2 cycles |
| `AUDIO_CAPTURE_SECONDS` | `15` | Durée du clip audio à transcrire |
| `TRANSCRIBE_EVERY_N` | `3` | Pipeline IA exécuté 1 cycle sur N |
| `WHISPER_MODEL` | `whisper-large-v3-turbo` | Modèle de transcription Groq |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Modèle de profilage Groq |
| `OUTPUT_CSV` | `audience_log.csv` | Fichier de log |

---

## Utilisation

### Lancement standard

```bash
python scraper_tunisia_v5.py
```

Cycle toutes les 60 s, pipeline IA toutes les 3 minutes, log dans `audience_log.csv`.

### Variantes utiles

```bash
# Cycle plus rapide (toutes les 30 s)
python scraper_tunisia_v5.py --watch 30

# Mode listener-count seul, sans Groq
python scraper_tunisia_v5.py --no-ai

# IA à chaque cycle (consomme plus de quota Groq)
python scraper_tunisia_v5.py --ai-every 1

# Ignorer radio-browser.info (un peu plus rapide)
python scraper_tunisia_v5.py --no-rb

# One-shot, sortie JSON pour intégration dashboard
python scraper_tunisia_v5.py --watch 0 --json

# Mode debug : trace chaque probe HTTP
python scraper_tunisia_v5.py --debug
```

### Tous les flags

| Flag | Description |
|---|---|
| `--watch SECONDS` | Intervalle entre cycles (défaut 60, mettre 0 pour one-shot) |
| `--no-ai` | Désactive complètement le pipeline IA Groq |
| `--no-rb` | Désactive la requête radio-browser.info |
| `--ai-every N` | Exécute le pipeline IA 1 cycle sur N (défaut 3) |
| `--json` | Émet du JSON brut au lieu du tableau formaté |
| `--debug` | Affiche tous les probes HTTP tentés (utile pour comprendre les `locked`) |

---

## Sortie console

À chaque cycle, deux sections :

### 1. Tableau de synthèse

```
Station                FM                  Status    Live   Peak   Clicks    Trend24h   Now playing
─────────────────────  ──────────────────  ────────  ─────  ─────  ────────  ─────────  ──────────────
Mosaïque FM            94.9 / 88.2-107.8   ok        4 230  12 800  187 432   +1 240     Saber Rebaï — Maalem
Shems FM               88.7-107.6 MHz      locked    —      —      94 217    +320       stream up, stats hidden
Express FM             103.6 / 104.0 MHz   locked    —      —      52 109    +180       stream up, stats hidden
...

Live listeners across 7 reachable stations: 18 432
Total radio-browser clicks: 1 234 567
```

### 2. Profils d'audience (cycles avec IA)

```
🎯  AUDIENCE PROFILES — predicted by llama-3.3-70b-versatile
────────────────────────────────────────────────────────────────────────

  ▶ Mosaïque FM                       4230 auditeurs            (ok)
     NOW PLAYING: Saber Rebaï — Maalem
     TRANSCRIPT : ...النشرة الاقتصادية لهذا الصباح، الدينار التونسي يواصل تراجعه أمام...
     KEYWORDS   : النشرة, الاقتصادية, الدينار, التونسي, تراجعه
     TOPIC      : Économie tunisienne (news)
     GENDER     : ♂ 58%   ♀ 42%
     AGE        : 15-24: 8% │ 25-34: 24% │ 35-44: 32% │ 45-54: 22% │ 55+: 14%
     SOCIO      : CSP+: 35% │ Moyen: 50% │ CSP-: 15%   (Cadres et indépendants)
     PERSONA    : Cadre tunisien de 35-44 ans suivant l'actualité économique du matin
```

---

## Format CSV

Une ligne par station par cycle dans `audience_log.csv` :

| Colonne | Type | Description |
|---|---|---|
| `timestamp` | ISO 8601 | Heure du cycle |
| `station` | str | Nom de la station |
| `status` | str | `ok` / `locked` / `hls` / `offline` |
| `listeners` | int | Auditeurs en direct (vide si `locked`) |
| `peak` | int | Pic de la session serveur |
| `clickcount` | int | Total clicks radio-browser.info |
| `clicktrend` | int | Évolution clicks sur 24 h |
| `now_playing` | str | Titre en cours (depuis le serveur stream) |
| `transcript` | str | 200 premiers chars de la transcription |
| `panel_topic` | str | Sujet inféré par le LLM |
| `tone` | str | `news` / `talk` / `music` / `comedy` / `religious` / `economic` / `cultural` |
| `men_pct`, `women_pct` | int | Répartition de genre prédite |
| `age_15_24` ... `age_55_plus` | int | Répartition par tranche d'âge (somme = 100) |
| `csp_plus`, `csp_middle`, `csp_minus` | int | Répartition socio-économique |
| `persona` | str | Phrase décrivant l'auditeur typique du moment |

Ce CSV est immédiatement consommable par Excel, Power BI, Tableau, pandas, etc.

---

## Stations couvertes

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

Pour ajouter une station, éditer la liste `STATIONS` en haut du script. Chaque entrée nécessite : `name`, `fm`, `stream`, `rb_name` (pour radio-browser), et un `profile` éditorial (utilisé par le LLM).

---

## Comprendre les statuts

| Statut | Signification | Action possible |
|---|---|---|
| `ok` | Listeners count récupéré du serveur | Rien — données fiables |
| `locked` | Stream actif mais stats endpoints bloqués par admin (cas tanitweb, infomaniak) | Lancer `--debug` pour voir tous les endpoints essayés. Pas de solution technique sans accord avec le CDN |
| `hls` | Stream HLS (`.m3u8`) — protocole stateless sans concept de "listeners" | Aucune — limite du protocole HLS |
| `offline` | Aucune réponse du serveur | Vérifier la connectivité, l'URL peut avoir changé |

Note : un statut `locked` n'est **pas** un bug — c'est une limite imposée par les opérateurs. Le pipeline IA continue de tourner sur ces stations en se basant uniquement sur le profil éditorial statique.

---

## Paramètres avancés

### Ajuster la fréquence du pipeline IA

Le pipeline IA (capture audio + Whisper + LLM) est ce qui consomme le plus de temps et de quota Groq. Trois leviers :

```python
POLL_INTERVAL = 60          # cycle complet (probe + radio-browser)
TRANSCRIBE_EVERY_N = 3      # IA toutes les 3 cycles
AUDIO_CAPTURE_SECONDS = 15  # durée du clip transcrit
```

Avec les valeurs par défaut : 1 transcription par station toutes les 3 minutes = 200 transcriptions/heure pour 10 stations.

### Changer la langue de transcription

Whisper auto-détecte, mais on guide via le profil éditorial de chaque station :

```python
"language": "Arabic dialect/French"   # → lang_hint = "ar"
"language": "French/English/Spanish"  # → lang_hint = "fr"
```

La détection se fait dans `enrich_with_ai()` : si le mot `French` apparaît dans le profil → `fr`, sinon → `ar`.

### Modifier le prompt du LLM

Le prompt système se trouve dans la constante `SYSTEM_PROMPT`. Le format de sortie JSON est strict — toute modification doit préserver la structure attendue par `render_audience_details()` et `log_csv()`.

---

## Limitations honnêtes

Trois choses que ce script **ne peut pas** faire, et il est important de les documenter clairement :

### 1. La FM hertzienne est invisible

Les auditeurs qui écoutent en voiture, sur un transistor, ou sur la chaîne hi-fi du salon ne sont **mesurables par aucun moyen technique**. Le broadcast FM n'a pas de retour. Seule la mesure panel (Sigma Conseil en Tunisie, Médiamétrie en France) couvre ces auditeurs. Ce script ne mesure que **les auditeurs du stream web**, qui représentent typiquement 5 à 15 % de l'audience totale.

### 2. Les profils démographiques sont des prédictions, pas des mesures

Le LLM infère un profil **probable** à partir du contenu diffusé et du profil éditorial. Ce n'est **pas** la démographie réelle des auditeurs — qui reste inconnue côté serveur Icecast. Pour valider, il faudrait croiser avec un panel ou un sondage.

### 3. Les stations en `locked` ne donnent pas de listeners

Tanitweb (radios publiques + Shems + Diwan) et infomaniak (Express FM) bloquent activement les endpoints stats. Aucun contournement technique fiable n'existe — le seul levier est un partenariat avec le CDN ou la station.

---

## Dépannage

### `GROQ_API_KEY non définie`

Le script tourne quand même, en mode listener-count seul. Pour activer l'IA :

```bash
export GROQ_API_KEY="gsk_..."
python scraper_tunisia_v5.py
```

### Beaucoup de stations en `locked`

C'est normal pour les CDN tanitweb et infomaniak. Lancer `--debug` pour confirmer :

```bash
python scraper_tunisia_v5.py --debug --watch 0
```

### Les chiffres listeners changent peu entre cycles

Les serveurs Icecast mettent à jour leur compte avec un délai (typiquement 30-60 s). Si le poll est trop rapide, on voit les mêmes chiffres. Augmenter `POLL_INTERVAL` à 90-120 s.

### Erreurs `transcription_error: 413`

Le clip dépasse 25 MB (limite Whisper). Réduire `AUDIO_CAPTURE_SECONDS` ou la station diffuse en très haute qualité (rare).

### `model_decommissioned` sur llama3-8b-8192

C'est attendu — le script utilise `llama-3.3-70b-versatile` qui est l'actuel. Si Groq déprécie aussi celui-ci, mettre à jour `LLM_MODEL` en consultant [console.groq.com/docs/models](https://console.groq.com/docs/models).

---

## Coûts & quotas

### Plan gratuit Groq

À titre indicatif (vérifier les limites actuelles sur le dashboard Groq) :

- **Whisper-large-v3-turbo** : ~7 200 secondes audio/jour gratuites
- **llama-3.3-70b-versatile** : ~14 400 requêtes/jour gratuites

### Estimation avec les défauts

- 10 stations × 15 s de transcription × 1 cycle sur 3 × cycles de 60 s
- ⇒ 10 × 15 × 20 = **3 000 secondes audio/heure** = ~50 minutes/heure
- ⇒ Sur 24 h = **1 200 minutes** = bien dans le quota gratuit

Pour rester dans la marge en cas d'usage continu, garder `TRANSCRIBE_EVERY_N >= 3`.

---

## Roadmap

Pistes d'évolution pour pousser le projet :

- [ ] Endpoint Flask exposant `/api/stats` pour brancher un dashboard temps réel
- [ ] Stockage InfluxDB / TimescaleDB plutôt que CSV pour les analyses temporelles
- [ ] Détection automatique de transitions (passage du talk → musique → publicité)
- [ ] Intégration OOH : capter les flux vidéo des panneaux publicitaires connectés
- [ ] Calibration des prédictions LLM contre les vagues Sigma Conseil (validation)
- [ ] Support multi-pays (Maroc, Algérie) — la logique de probe est générique

---

## License

MIT — utilisable librement, attribution appréciée.

---

## Crédits

- **Probe engine** : inspiré de `tn_radio_listeners.py` v2 (résolution playlists, fallback radio-browser.info)
- **Pipeline IA** : Groq Whisper + llama-3.3-70b
- **Données popularité** : [radio-browser.info](https://www.radio-browser.info/) (communautaire, gratuit, ouvert)
