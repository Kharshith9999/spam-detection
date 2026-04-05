/**
 * SpamShield AI — Browser-side ML Inference
 *
 * Loads the exported TF-IDF + Logistic Regression model (model.json)
 * and runs inference entirely in the browser — no server required.
 */

// ── State ──────────────────────────────────────────────────────────────
let MODEL = null;   // { vocab, idf, coef, intercept, stats }

const STOP_WORDS = new Set([
  "a","about","above","after","again","against","all","am","an","and","any","are",
  "aren't","as","at","be","because","been","before","being","below","between","both",
  "but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't",
  "doing","don't","down","during","each","few","for","from","further","get","got","had",
  "hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her",
  "here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll",
  "i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me",
  "more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only",
  "or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she",
  "she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's",
  "the","their","theirs","them","themselves","then","there","there's","these","they",
  "they'd","they'll","they're","they've","this","those","through","to","too","under",
  "until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
  "weren't","what","what's","when","when's","where","where's","which","while","who",
  "who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll",
  "you're","you've","your","yours","yourself","yourselves"
]);

const SAMPLES = {
  spam: `Subject: URGENT — You Have Been Selected As Our Lucky Winner!

CONGRATULATIONS!! You have been SELECTED as a WINNER of $5,000,000 USD prize money!
Click HERE immediately to CLAIM your prize: http://claimprize-win.com/urgent

To release your funds we require a small processing fee of just $99. This is 100% LEGITIMATE.
Act NOW — this FREE offer EXPIRES in 24 HOURS!!
Reply with your full name, address, bank account number and social security number.

FREE FREE FREE!!! Limited Time Offer!!! CLICK NOW!!!
DO NOT IGNORE THIS EMAIL — FINAL NOTICE!!!`,

  ham: `Subject: Re: Project Update — Q2 Report

Hi Sarah,

Thanks for sending over the draft. I've reviewed sections 1 through 3 and left some inline comments in the shared document.

Overall, the structure looks solid. My main suggestion would be to expand the market analysis section with the latest figures from our analytics dashboard. Could you also double-check the revenue projections on page 7? The numbers seem slightly off compared to last quarter's actuals.

Let's plan to sync on Thursday at 2 PM to go through the final version before we submit to the board. Does that time work for you?

Looking forward to your thoughts.

Best regards,
Michael`
};

// ── DOM refs ────────────────────────────────────────────────────────────
const emailInput    = document.getElementById("email-input");
const charCount     = document.getElementById("char-count");
const btnAnalyze    = document.getElementById("btn-analyze");
const btnLoader     = document.getElementById("btn-loader");
const btnText       = btnAnalyze.querySelector(".btn-text");
const btnIcon       = btnAnalyze.querySelector(".btn-icon");
const btnClear      = document.getElementById("btn-clear");
const btnClearHist  = document.getElementById("btn-clear-history");
const resultCard    = document.getElementById("result-card");
const historyCard   = document.getElementById("history-card");
const historyList   = document.getElementById("history-list");
const modelStatus   = document.getElementById("model-status");
const loadingOverlay= document.getElementById("loading-overlay");

const history = [];

// ── Load model.json ─────────────────────────────────────────────────────
async function loadModel() {
  loadingOverlay.classList.add("active");
  try {
    const res  = await fetch("model.json");
    const data = await res.json();
    MODEL = data;

    // Populate stats bar
    const s = data.stats;
    document.getElementById("stat-accuracy").textContent = s.accuracy  + "%";
    document.getElementById("stat-f1").textContent       = s.f1_spam   + "%";
    document.getElementById("stat-total").textContent    = s.total_samples.toLocaleString();
    document.getElementById("stat-roc").textContent      = s.roc_auc   + "%";

    // Update status badge
    modelStatus.textContent  = "● Model Ready";
    modelStatus.className    = "badge badge-ready";
    btnAnalyze.disabled      = false;
    loadingOverlay.classList.remove("active");
  } catch (err) {
    modelStatus.textContent = "⚠ Model failed to load";
    modelStatus.className   = "badge badge-error";
    loadingOverlay.classList.remove("active");
    console.error("Model load error:", err);
  }
}
loadModel();

// ── TF-IDF + Logistic Regression in JS ─────────────────────────────────
function tokenize(text) {
  // Lowercase, keep only alphanumeric
  const words = text.toLowerCase().match(/[a-z0-9]+/g) || [];
  return words.filter(w => w.length > 1 && !STOP_WORDS.has(w));
}

function buildNgrams(tokens, n) {
  if (n === 1) return tokens;
  const ngrams = [...tokens];
  for (let i = 0; i <= tokens.length - n; i++) {
    ngrams.push(tokens.slice(i, i + n).join(" "));
  }
  return ngrams;
}

function predict(text) {
  const { vocab, idf, coef, intercept } = MODEL;
  const numFeatures = idf.length;

  // 1. Tokenize + generate 1-grams and 2-grams
  const tokens = tokenize(text);
  const terms  = buildNgrams(tokens, 2);  // includes 1-grams + 2-grams

  // 2. Compute raw term frequencies
  const tf = {};
  for (const t of terms) {
    if (t in vocab) {
      tf[t] = (tf[t] || 0) + 1;
    }
  }

  // 3. Apply sublinear TF scaling: 1 + log(count)
  // 4. Multiply by IDF
  // 5. Build sparse feature vector as Map (index → value)
  const features = {};
  for (const [term, count] of Object.entries(tf)) {
    const idx = vocab[term];
    const tfVal = 1 + Math.log(count);       // sublinear_tf=True
    features[idx] = tfVal * idf[idx];
  }

  // 6. L2 normalize
  let norm = 0;
  for (const v of Object.values(features)) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (const k in features) features[k] /= norm;
  }

  // 7. Dot product with LR coefficients + intercept
  let score = intercept;
  for (const [idx, val] of Object.entries(features)) {
    score += val * coef[parseInt(idx, 10)];
  }

  // 8. Sigmoid → spam probability
  const spamProb = 1 / (1 + Math.exp(-score));
  const hamProb  = 1 - spamProb;
  const isSpam   = spamProb >= 0.5;

  return {
    is_spam:    isSpam,
    label:      isSpam ? "spam" : "ham",
    spam_prob:  +(spamProb * 100).toFixed(2),
    ham_prob:   +(hamProb  * 100).toFixed(2),
    confidence: +(Math.max(spamProb, hamProb) * 100).toFixed(2),
    word_count: tokens.length,
  };
}

// ── UI: character counter ───────────────────────────────────────────────
emailInput.addEventListener("input", () => {
  const n = emailInput.value.length;
  charCount.textContent = n.toLocaleString() + " character" + (n !== 1 ? "s" : "");
});

// ── UI: sample buttons ──────────────────────────────────────────────────
document.getElementById("sample-spam").addEventListener("click", () => {
  emailInput.value = SAMPLES.spam;
  emailInput.dispatchEvent(new Event("input"));
  emailInput.focus();
});
document.getElementById("sample-ham").addEventListener("click", () => {
  emailInput.value = SAMPLES.ham;
  emailInput.dispatchEvent(new Event("input"));
  emailInput.focus();
});

// ── UI: clear ──────────────────────────────────────────────────────────
btnClear.addEventListener("click", () => {
  emailInput.value = "";
  emailInput.dispatchEvent(new Event("input"));
  resultCard.classList.add("hidden");
  emailInput.focus();
});
btnClearHist.addEventListener("click", () => {
  history.length = 0;
  historyList.innerHTML = "";
  historyCard.classList.add("hidden");
});

// ── UI: analyze ────────────────────────────────────────────────────────
btnAnalyze.addEventListener("click", analyze);
emailInput.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") analyze();
});

function analyze() {
  if (!MODEL) { alert("Model is still loading, please wait…"); return; }
  const text = emailInput.value.trim();
  if (!text) { shake(emailInput); return; }

  // brief loading flash
  setLoading(true);
  setTimeout(() => {
    const result = predict(text);
    showResult(result, text);
    addToHistory(result, text);
    setLoading(false);
  }, 120);
}

// ── Show result ─────────────────────────────────────────────────────────
function showResult(data, text) {
  const isSpam = data.is_spam;

  resultCard.classList.remove("hidden", "spam-result", "ham-result");
  resultCard.classList.add(isSpam ? "spam-result" : "ham-result");

  document.getElementById("result-icon").textContent  = isSpam ? "🚨" : "✅";
  const verdictEl = document.getElementById("result-verdict");
  verdictEl.textContent = isSpam ? "SPAM Detected!" : "HAM (Legitimate)";
  verdictEl.className   = "result-verdict " + (isSpam ? "spam-verdict" : "ham-verdict");
  document.getElementById("result-confidence").textContent =
    `Confidence: ${data.confidence.toFixed(1)}%`;

  // Verdict banner
  const banner = document.getElementById("verdict-banner");
  const bannerText = document.getElementById("verdict-banner-text");
  banner.className = "verdict-banner " + (isSpam ? "spam-banner" : "ham-banner");
  bannerText.textContent = isSpam
    ? "⚠️  Warning: This email shows strong spam signals. Do not click any links or provide personal information."
    : "✅  This email appears to be legitimate. It does not show common spam patterns.";

  // Bars
  ["spam-pct","ham-pct"].forEach((id,i) => {
    document.getElementById(id).textContent = [data.spam_prob, data.ham_prob][i].toFixed(1) + "%";
  });
  const spamBar = document.getElementById("spam-bar");
  const hamBar  = document.getElementById("ham-bar");
  spamBar.style.width = "0%"; hamBar.style.width = "0%";
  requestAnimationFrame(() => setTimeout(() => {
    spamBar.style.width = data.spam_prob + "%";
    hamBar.style.width  = data.ham_prob  + "%";
  }, 50));

  // Chips
  document.getElementById("detail-chips").innerHTML = `
    <div class="chip"><span>📊</span> ${isSpam ? "Spam" : "Ham"}: ${data.confidence.toFixed(1)}%</div>
    <div class="chip"><span>📝</span> ${data.word_count} words</div>
    <div class="chip"><span>🧠</span> TF-IDF + LR</div>
    <div class="chip"><span>${isSpam ? "⛔" : "🔒"}</span> ${isSpam ? "High Risk" : "Safe"}</div>
  `;

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── History ─────────────────────────────────────────────────────────────
function addToHistory(data, text) {
  history.unshift({ data, text });
  historyCard.classList.remove("hidden");
  historyList.innerHTML = history.slice(0, 8).map((entry, i) => {
    const preview = entry.text.replace(/\s+/g, " ").substring(0, 80) + "…";
    const label   = entry.data.is_spam ? "spam" : "ham";
    const conf    = entry.data.confidence.toFixed(1);
    return `<div class="history-item">
      <span class="history-badge ${label}">${label}</span>
      <span class="history-text" title="${esc(entry.text)}">${esc(preview)}</span>
      <span class="history-conf">${conf}%</span>
    </div>`;
  }).join("");
}

// ── Helpers ─────────────────────────────────────────────────────────────
function setLoading(on) {
  btnAnalyze.disabled      = on || !MODEL;
  btnLoader.classList.toggle("hidden", !on);
  btnIcon.style.visibility = on ? "hidden" : "visible";
  btnText.textContent      = on ? "Analyzing…" : "Check This Email";
}

function shake(el) {
  el.style.animation = "none"; el.offsetHeight;
  el.style.animation = "shake 0.4s ease";
  el.addEventListener("animationend", () => { el.style.animation = ""; }, { once: true });
}

function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// inject shake keyframe
const kf = document.createElement("style");
kf.textContent = `@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-8px)}40%{transform:translateX(8px)}60%{transform:translateX(-5px)}80%{transform:translateX(5px)}}`;
document.head.appendChild(kf);
