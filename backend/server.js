import express from "express";
import { spawn } from "child_process";
import { existsSync, readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load .env from project root so GEMINI_API_KEY and GROQ_API_KEY are available
const envPath = path.join(__dirname, "..", ".env");
if (existsSync(envPath)) {
  readFileSync(envPath, "utf8").split("\n").forEach((line) => {
    const m = line.match(/^([A-Z_]+)=(.+)$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
  });
}

// Prefer the shared project venv; fall back to system python
const PYTHON_EXE = (() => {
  const venvPy = path.join(__dirname, "..", ".venv", "Scripts", "python.exe");
  return existsSync(venvPy) ? venvPy : "python";
})();

const app = express();
app.use(express.json());

// Allow requests from the Vite dev server on any port
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") { res.sendStatus(204); return; }
  next();
});

// ── TV chatbot system prompt ──────────────────────────────────────────────────

const SYSTEM_PROMPT = `Tu es un expert en planification publicitaire TV pour le marché tunisien.
Tu as accès à des données d'audience réelles de chaînes TV tunisiennes.

Heures disponibles : 0h–23h.
Tranches d'âge : 18-24, 25-34, 35-44, 45+.
Sexe : H (Homme), F (Femme).
Revenu : faible, moyen, eleve.
Jours : Lundi, Mardi, Mercredi, Jeudi, Vendredi, Samedi, Dimanche.

FORMULE DE SCORING :
1. SPA = 0.4×age_pct + 0.3×sex_pct + 0.3×rev_pct
2. SVP = audience_totale × SPA
3. Bonus jour : Lundi–Jeudi ×1.00–1.05, Vendredi ×1.20, Samedi ×1.30, Dimanche ×1.25, Férié +10%
4. Score final = SVP × bonus_jour → normalisé 0–100

RÈGLES : Demande âge/sexe/revenu si vague. Affiche TOP 5 avec jour. Verdict ✅/❌ pour les évaluations. Réponds TOUJOURS en français.

IMPORTANT: Keep all responses concise and short. Maximum 5 lines. No long tables. No detailed calculations. Just give the key recommendation directly.`;

// ── POST /api/chat — Groq (llama-3.3-70b) ────────────────────────────────────

app.post("/api/chat", async (req, res) => {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "GROQ_API_KEY not set in environment" });
    return;
  }

  const { message, history = [] } = req.body;
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...history.map((h) => ({
      role: h.role === "bot" ? "assistant" : "user",
      content: h.text,
    })),
    { role: "user", content: message },
  ];

  try {
    const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model: "llama-3.3-70b-versatile", messages, max_tokens: 512 }),
    });

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(`Groq API ${r.status}: ${errText.slice(0, 300)}`);
    }

    const data = await r.json();
    const content = data.choices?.[0]?.message?.content;
    if (!content) throw new Error("Groq returned no content");
    res.json({ response: content });
  } catch (err) {
    console.error("[chat]", err.message);
    res.status(500).json({ error: err.message ?? "Internal server error" });
  }
});

// ── POST /api/campaign-hit — Python script ────────────────────────────────────

// Override with SCRIPT_DIR env var if SearchedArtical.py lives elsewhere.
const SCRIPT_DIR =
  process.env.SCRIPT_DIR ??
  path.join(__dirname, "..", "frontend", "tdrtyf", "scripts");

const FRONTEND_ROOT =
  process.env.FRONTEND_ROOT ??
  path.join(__dirname, "..", "frontend", "tdrtyf");

app.post("/api/campaign-hit", (req, res) => {
  const { keyword, date, airTime } = req.body;
  if (!keyword || !date) {
    res.status(400).json({ error: "Missing keyword or date" });
    return;
  }

  const rawTime = (airTime ?? "").trim();
  const hhmm = rawTime.length >= 5 ? rawTime.slice(0, 5) : "00:00";
  const eventTime = `${date} ${hhmm}`;
  const scriptPath = path.join(SCRIPT_DIR, "SearchedArtical.py");

  console.log(`[campaign] keyword=${keyword} eventTime=${eventTime}`);

  const proc = spawn(
    PYTHON_EXE,
    [scriptPath, "--keyword", keyword, "--event-time", eventTime, "--geo", "TN"],
    { cwd: FRONTEND_ROOT },
  );

  let stdout = "";
  let stderr = "";
  proc.stdout.on("data", (d) => { stdout += d.toString(); });
  proc.stderr.on("data", (d) => { stderr += d.toString(); });

  proc.on("close", (code) => {
    console.log(`[campaign] exit code: ${code}`);
    if (stderr) console.log("[campaign] stderr:", stderr.slice(0, 500));
    if (code !== 0) {
      res.status(500).json({ error: `Python script failed: ${stderr.slice(0, 300)}` });
      return;
    }
    try {
      res.json(JSON.parse(stdout.trim()));
    } catch {
      res.status(500).json({ error: `Invalid JSON from script: ${stdout.slice(0, 300)}` });
    }
  });

  proc.on("error", (err) => {
    res.status(500).json({ error: `Failed to start Python: ${err.message}` });
  });
});

// ─────────────────────────────────────────────────────────────────────────────

const PORT = process.env.PORT ?? 3001;
app.listen(PORT, () =>
  console.log(`[TV Backend] Listening on http://localhost:${PORT}`),
);
