/* ─────────────────────────────────────────────────
   SpamShield AI — Frontend JavaScript
   ───────────────────────────────────────────────── */

const SAMPLES = {
  spam: `Subject: URGENT - You've Been Selected! Claim Your Prize NOW!

Congratulations!! You have been SELECTED as our LUCKY WINNER of $5,000,000.00 USD!
Click HERE immediately to claim your prize: http://totallylegit.win/claim

To release your funds, we require a small processing fee of $100. 
Act NOW - this offer EXPIRES in 24 hours!
Reply with your full name, address, bank account number, and social security number.

DO NOT IGNORE THIS EMAIL. This is your FINAL NOTICE.
FREE FREE FREE - Limited Time Offer!!!`,

  ham: `Subject: Re: Project Update – Q2 Report

Hi Sarah,

Thanks for sending over the draft. I've reviewed sections 1–3 and left some comments in the shared document.

Overall the structure looks good. My main suggestion would be to expand the market analysis section with the latest figures from our analytics dashboard. Could you also double-check the revenue projections on page 7?

Let's plan to sync on Thursday at 2 PM to go through the final version before submitting. Does that work for you?

Best regards,
Michael`
};

const history = [];

// ── DOM refs ──────────────────────────────────────────────
const emailInput     = document.getElementById("email-input");
const charCount      = document.getElementById("char-count");
const btnAnalyze     = document.getElementById("btn-analyze");
const btnLoader      = document.getElementById("btn-loader");
const btnText        = btnAnalyze.querySelector(".btn-text");
const btnIcon        = btnAnalyze.querySelector(".btn-icon");
const btnClear       = document.getElementById("btn-clear");
const btnClearHist   = document.getElementById("btn-clear-history");
const resultCard     = document.getElementById("result-card");
const historyCard    = document.getElementById("history-card");
const historyList    = document.getElementById("history-list");

// ── Fetch model stats on load ──────────────────────────────
async function loadStats() {
  try {
    const res  = await fetch("/api/stats");
    const data = await res.json();
    document.getElementById("stat-accuracy").textContent = data.accuracy + "%";
    document.getElementById("stat-f1").textContent       = data.f1_spam  + "%";
    document.getElementById("stat-total").textContent    = data.total_samples.toLocaleString();
    document.getElementById("stat-roc").textContent      = data.roc_auc  + "%";
    document.getElementById("model-name-footer").textContent = data.model;
  } catch (e) {
    console.warn("Could not load stats:", e);
  }
}
loadStats();

// ── Character counter ──────────────────────────────────────
emailInput.addEventListener("input", () => {
  const n = emailInput.value.length;
  charCount.textContent = n.toLocaleString() + " character" + (n !== 1 ? "s" : "");
});

// ── Sample buttons ─────────────────────────────────────────
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

// ── Clear ──────────────────────────────────────────────────
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

// ── Analyze ────────────────────────────────────────────────
btnAnalyze.addEventListener("click", analyze);
emailInput.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") analyze();
});

async function analyze() {
  const text = emailInput.value.trim();
  if (!text) {
    shake(emailInput);
    return;
  }

  setLoading(true);

  try {
    const res  = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.error) { alert("Error: " + data.error); return; }
    showResult(data, text);
    addToHistory(data, text);
  } catch (err) {
    alert("Could not connect to the server. Make sure the Flask API is running.");
    console.error(err);
  } finally {
    setLoading(false);
  }
}

// ── Show Result ────────────────────────────────────────────
function showResult(data, text) {
  const isSpam = data.is_spam;

  resultCard.classList.remove("hidden", "spam-result", "ham-result");
  resultCard.classList.add(isSpam ? "spam-result" : "ham-result");

  // Icon + verdict
  document.getElementById("result-icon").textContent    = isSpam ? "🚨" : "✅";
  const verdictEl = document.getElementById("result-verdict");
  verdictEl.textContent    = isSpam ? "SPAM Detected" : "HAM (Legitimate)";
  verdictEl.className      = "result-verdict " + (isSpam ? "spam-verdict" : "ham-verdict");
  document.getElementById("result-confidence").textContent =
    `Confidence: ${data.confidence.toFixed(1)}%`;

  // Bars — reset first, then animate
  document.getElementById("spam-pct").textContent = data.spam_prob.toFixed(1) + "%";
  document.getElementById("ham-pct").textContent  = data.ham_prob.toFixed(1)  + "%";
  const spamBar = document.getElementById("spam-bar");
  const hamBar  = document.getElementById("ham-bar");
  spamBar.style.width = "0%";
  hamBar.style.width  = "0%";
  requestAnimationFrame(() => {
    setTimeout(() => {
      spamBar.style.width = data.spam_prob + "%";
      hamBar.style.width  = data.ham_prob  + "%";
    }, 50);
  });

  // Detail chips
  const chips = document.getElementById("detail-chips");
  chips.innerHTML = `
    <div class="chip"><span>📊</span> Spam: ${data.spam_prob.toFixed(2)}%</div>
    <div class="chip"><span>📧</span> Ham: ${data.ham_prob.toFixed(2)}%</div>
    <div class="chip"><span>🧠</span> TF-IDF + LR</div>
    <div class="chip"><span>${isSpam ? "⚠️" : "🔒"}</span> ${isSpam ? "High Risk" : "Safe"}</div>
    <div class="chip"><span>📝</span> ${text.split(/\s+/).length} words</div>
  `;

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── History ────────────────────────────────────────────────
function addToHistory(data, text) {
  history.unshift({ data, text, time: new Date() });

  historyCard.classList.remove("hidden");
  historyList.innerHTML = history.slice(0, 8).map((entry, i) => {
    const preview = entry.text.replace(/\s+/g, " ").substring(0, 80) + "…";
    const conf    = entry.data.confidence.toFixed(1);
    const label   = entry.data.is_spam ? "spam" : "ham";
    return `
      <div class="history-item" id="hist-item-${i}">
        <span class="history-badge ${label}">${label}</span>
        <span class="history-text" title="${escapeHtml(entry.text)}">${escapeHtml(preview)}</span>
        <span class="history-conf">${conf}%</span>
      </div>`;
  }).join("");
}

// ── Helpers ────────────────────────────────────────────────
function setLoading(on) {
  btnAnalyze.disabled      = on;
  btnLoader.style.display  = on ? "block"  : "none";
  btnIcon.style.display    = on ? "none"   : "inline";
  btnText.textContent      = on ? "Analyzing…" : "Analyze Email";
}

function shake(el) {
  el.style.animation = "none";
  el.offsetHeight;   // reflow
  el.style.animation = "shake 0.4s ease";
  el.addEventListener("animationend", () => { el.style.animation = ""; }, { once: true });
}

function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ── Inject shake keyframes ─────────────────────────────────
const kf = document.createElement("style");
kf.textContent = `@keyframes shake {
  0%,100%{transform:translateX(0)}
  20%{transform:translateX(-8px)}
  40%{transform:translateX(8px)}
  60%{transform:translateX(-6px)}
  80%{transform:translateX(6px)}
}`;
document.head.appendChild(kf);
