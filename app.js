/**
 * SpamShield AI — app.js
 * Browser-side ML inference + Gmail OAuth integration
 * Client ID: 699683361957-vjfiu37l8kuv10is1ca6742r90b6reek.apps.googleusercontent.com
 */

const CLIENT_ID = "699683361957-vjfiu37l8kuv10is1ca6742r90b6reek.apps.googleusercontent.com";
const SCOPE     = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email";
const MAX_EMAILS = 30;

// ── State ──────────────────────────────────────────────────────────────
let MODEL       = null;
let tokenClient = null;
let accessToken = null;
let allEmailResults = [];

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

Overall, the structure looks solid. My main suggestion would be to expand the market analysis section with the latest figures from our analytics dashboard. Could you also double-check the revenue projections on page 7?

Let's plan to sync on Thursday at 2 PM to go through the final version before we submit. Does that time work for you?

Best regards,
Michael`
};

// ── DOM refs ────────────────────────────────────────────────────────────
const emailInput      = document.getElementById("email-input");
const charCount       = document.getElementById("char-count");
const btnAnalyze      = document.getElementById("btn-analyze");
const btnLoader       = document.getElementById("btn-loader");
const btnText         = btnAnalyze.querySelector(".btn-text");
const btnIcon         = btnAnalyze.querySelector(".btn-icon");
const btnClear        = document.getElementById("btn-clear");
const btnClearHist    = document.getElementById("btn-clear-history");
const resultCard      = document.getElementById("result-card");
const historyCard     = document.getElementById("history-card");
const historyList     = document.getElementById("history-list");
const modelStatus     = document.getElementById("model-status");
const loadingOverlay  = document.getElementById("loading-overlay");

// Gmail DOM refs
const btnConnectGmail   = document.getElementById("btn-connect-gmail");
const gmailConnectCard  = document.getElementById("gmail-connect-card");
const gmailScanningCard = document.getElementById("gmail-scanning-card");
const scanningProgress  = document.getElementById("scanning-progress");
const inboxCard         = document.getElementById("inbox-card");
const inboxList         = document.getElementById("inbox-list");
const inboxSummary      = document.getElementById("inbox-summary");
const gmailUserEl       = document.getElementById("gmail-user");
const gmailAvatar       = document.getElementById("gmail-avatar");
const gmailEmailEl      = document.getElementById("gmail-email");
const btnSignout        = document.getElementById("btn-signout");
const btnRescan         = document.getElementById("btn-rescan");
const inboxFilter       = document.getElementById("inbox-filter");

const history = [];

// ── Load model.json ─────────────────────────────────────────────────────
async function loadModel() {
  loadingOverlay.classList.add("active");
  try {
    const res  = await fetch("model.json");
    const data = await res.json();
    MODEL = data;
    const s = data.stats;
    document.getElementById("stat-accuracy").textContent = s.accuracy  + "%";
    document.getElementById("stat-f1").textContent       = s.f1_spam   + "%";
    document.getElementById("stat-total").textContent    = s.total_samples.toLocaleString();
    document.getElementById("stat-roc").textContent      = s.roc_auc   + "%";
    modelStatus.textContent = "● Model Ready";
    modelStatus.className   = "badge badge-ready";
    btnAnalyze.disabled     = false;
    loadingOverlay.classList.remove("active");
    initGmailAuth();
  } catch (err) {
    modelStatus.textContent = "⚠ Model failed to load";
    modelStatus.className   = "badge badge-error";
    loadingOverlay.classList.remove("active");
    console.error("Model load error:", err);
  }
}
loadModel();

// ── TF-IDF + Logistic Regression ────────────────────────────────────────
function tokenize(text) {
  const words = text.toLowerCase().match(/[a-z0-9]+/g) || [];
  return words.filter(w => w.length > 1 && !STOP_WORDS.has(w));
}
function buildNgrams(tokens) {
  const ngrams = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) ngrams.push(tokens[i] + " " + tokens[i+1]);
  return ngrams;
}
function predict(text) {
  if (!MODEL) return null;
  const { vocab, idf, coef, intercept } = MODEL;
  const tokens = tokenize(text);
  const terms  = buildNgrams(tokens);
  const tf = {};
  for (const t of terms) {
    if (t in vocab) tf[t] = (tf[t] || 0) + 1;
  }
  const features = {};
  for (const [term, count] of Object.entries(tf)) {
    const idx = vocab[term];
    features[idx] = (1 + Math.log(count)) * idf[idx];
  }
  let norm = 0;
  for (const v of Object.values(features)) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 0) for (const k in features) features[k] /= norm;
  let score = intercept;
  for (const [idx, val] of Object.entries(features)) score += val * coef[parseInt(idx,10)];
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

// ── Gmail OAuth ─────────────────────────────────────────────────────────
function initGmailAuth() {
  if (!window.google) {
    setTimeout(initGmailAuth, 500);
    return;
  }
  tokenClient = google.accounts.oauth2.initTokenClient({
    client_id: CLIENT_ID,
    scope:     SCOPE,
    callback:  onTokenReceived,
  });
}

function onTokenReceived(resp) {
  if (resp.error) {
    console.error("OAuth error:", resp.error);
    alert("Google sign-in failed: " + resp.error);
    return;
  }
  accessToken = resp.access_token;
  fetchUserInfo().then(() => scanInbox());
}

async function fetchUserInfo() {
  try {
    const res  = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
      headers: { Authorization: "Bearer " + accessToken }
    });
    const info = await res.json();
    gmailAvatar.src      = info.picture || "";
    gmailEmailEl.textContent = info.email || "";
    gmailUserEl.classList.remove("hidden");
  } catch(e) { console.warn("Could not fetch user info", e); }
}

btnConnectGmail.addEventListener("click", () => {
  if (!tokenClient) { alert("Auth not ready yet, please wait a moment."); return; }
  tokenClient.requestAccessToken({ prompt: "consent" });
});

btnSignout.addEventListener("click", () => {
  if (accessToken) google.accounts.oauth2.revoke(accessToken, () => {});
  accessToken = null;
  allEmailResults = [];
  gmailUserEl.classList.add("hidden");
  gmailConnectCard.classList.remove("hidden");
  gmailScanningCard.classList.add("hidden");
  inboxCard.classList.add("hidden");
});

btnRescan.addEventListener("click", () => {
  if (!accessToken) return;
  scanInbox();
});

// ── Gmail API ───────────────────────────────────────────────────────────
async function gmailGet(path) {
  const res = await fetch("https://gmail.googleapis.com/gmail/v1/users/me/" + path, {
    headers: { Authorization: "Bearer " + accessToken }
  });
  if (!res.ok) throw new Error("Gmail API error: " + res.status);
  return res.json();
}

function decodeBase64(str) {
  try {
    return decodeURIComponent(escape(atob(str.replace(/-/g,"+").replace(/_/g,"/"))));
  } catch { return ""; }
}

function extractBody(payload) {
  if (!payload) return "";
  // Direct body
  if (payload.body && payload.body.data) return decodeBase64(payload.body.data);
  // Multipart: prefer text/plain
  if (payload.parts) {
    for (const part of payload.parts) {
      if (part.mimeType === "text/plain" && part.body && part.body.data)
        return decodeBase64(part.body.data);
    }
    // Fall back to any part
    for (const part of payload.parts) {
      const txt = extractBody(part);
      if (txt) return txt;
    }
  }
  return "";
}

function getHeader(headers, name) {
  const h = headers.find(h => h.name.toLowerCase() === name.toLowerCase());
  return h ? h.value : "";
}

function formatDate(internalDate) {
  try {
    const d = new Date(parseInt(internalDate));
    return d.toLocaleDateString("en-US", { month:"short", day:"numeric" });
  } catch { return ""; }
}

async function scanInbox() {
  gmailConnectCard.classList.add("hidden");
  gmailScanningCard.classList.remove("hidden");
  inboxCard.classList.add("hidden");
  scanningProgress.textContent = "Fetching email list…";

  try {
    // 1. Get list of message IDs
    const listData = await gmailGet(`messages?maxResults=${MAX_EMAILS}&labelIds=INBOX`);
    const messages = listData.messages || [];

    if (!messages.length) {
      scanningProgress.textContent = "No emails found in inbox.";
      return;
    }

    // 2. Fetch each message
    allEmailResults = [];
    for (let i = 0; i < messages.length; i++) {
      scanningProgress.textContent = `Scanning email ${i+1} of ${messages.length}…`;
      try {
        const msg  = await gmailGet(`messages/${messages[i].id}?format=full`);
        const hdrs = msg.payload.headers;
        const from    = getHeader(hdrs, "From");
        const subject = getHeader(hdrs, "Subject") || "(no subject)";
        const date    = formatDate(msg.internalDate);
        const body    = extractBody(msg.payload);
        const content = `Subject: ${subject}\n${body}`.trim();
        const result  = predict(content) || { is_spam:false, label:"ham", spam_prob:0, ham_prob:100, confidence:100 };
        allEmailResults.push({ from, subject, date, body: body.substring(0,400), result });
      } catch(e) {
        console.warn("Skipping message", messages[i].id, e);
      }
    }

    // 3. Show results
    gmailScanningCard.classList.add("hidden");
    renderInbox("all");
    inboxCard.classList.remove("hidden");

  } catch (err) {
    gmailScanningCard.classList.add("hidden");
    gmailConnectCard.classList.remove("hidden");
    console.error("Gmail scan failed:", err);
    alert("Failed to read Gmail: " + err.message + "\n\nMake sure you granted permission.");
  }
}

// ── Render Inbox ─────────────────────────────────────────────────────────
function renderInbox(filter) {
  const emails  = filter === "spam" ? allEmailResults.filter(e => e.result.is_spam)
                : filter === "ham"  ? allEmailResults.filter(e => !e.result.is_spam)
                : allEmailResults;

  const spamCount = allEmailResults.filter(e => e.result.is_spam).length;
  const hamCount  = allEmailResults.length - spamCount;

  inboxSummary.innerHTML =
    `Scanned <strong>${allEmailResults.length}</strong> emails — ` +
    `<span class="spam-count">🚨 ${spamCount} spam</span> · ` +
    `<span class="ham-count">✅ ${hamCount} ham</span>`;

  inboxList.innerHTML = emails.map((e, i) => {
    const label = e.result.is_spam ? "spam" : "ham";
    const conf  = e.result.confidence.toFixed(0);
    return `
      <div class="email-row ${label}-row" onclick="toggleDetail(${i})" id="email-row-${i}">
        <span class="email-verdict-badge ${label}">${label === "spam" ? "🚨" : "✅"} ${label}</span>
        <div class="email-info">
          <div class="email-from">${esc(e.from)}</div>
          <div class="email-subject">${esc(e.subject)}</div>
        </div>
        <span class="email-conf">${conf}%</span>
        <span class="email-date">${esc(e.date)}</span>
      </div>
      <div class="email-detail" id="email-detail-${i}">
        <strong>Spam:</strong> ${e.result.spam_prob.toFixed(1)}%  |  <strong>Ham:</strong> ${e.result.ham_prob.toFixed(1)}%
        <br/><br/>${esc(e.body || "(no body)")}
      </div>`;
  }).join("");

  if (!emails.length) {
    inboxList.innerHTML = `<div style="text-align:center;padding:30px;color:var(--text-muted)">No emails match this filter.</div>`;
  }
}

window.toggleDetail = function(i) {
  const detail = document.getElementById("email-detail-" + i);
  if (detail) detail.classList.toggle("open");
};

inboxFilter.addEventListener("change", () => renderInbox(inboxFilter.value));

// ── Manual checker ──────────────────────────────────────────────────────
emailInput.addEventListener("input", () => {
  const n = emailInput.value.length;
  charCount.textContent = n.toLocaleString() + " character" + (n!==1?"s":"");
});
document.getElementById("sample-spam").addEventListener("click", () => {
  emailInput.value = SAMPLES.spam; emailInput.dispatchEvent(new Event("input")); emailInput.focus();
});
document.getElementById("sample-ham").addEventListener("click", () => {
  emailInput.value = SAMPLES.ham; emailInput.dispatchEvent(new Event("input")); emailInput.focus();
});
btnClear.addEventListener("click", () => {
  emailInput.value = ""; emailInput.dispatchEvent(new Event("input"));
  resultCard.classList.add("hidden"); emailInput.focus();
});
btnClearHist.addEventListener("click", () => {
  history.length = 0; historyList.innerHTML = ""; historyCard.classList.add("hidden");
});
btnAnalyze.addEventListener("click", analyze);
emailInput.addEventListener("keydown", e => { if(e.ctrlKey && e.key==="Enter") analyze(); });

function analyze() {
  if (!MODEL) { alert("Model is still loading…"); return; }
  const text = emailInput.value.trim();
  if (!text) { shake(emailInput); return; }
  setLoading(true);
  setTimeout(() => {
    const result = predict(text);
    showResult(result, text);
    addToHistory(result, text);
    setLoading(false);
  }, 120);
}

function showResult(data, text) {
  const isSpam = data.is_spam;
  resultCard.classList.remove("hidden","spam-result","ham-result");
  resultCard.classList.add(isSpam ? "spam-result" : "ham-result");
  document.getElementById("result-icon").textContent  = isSpam ? "🚨" : "✅";
  const verdictEl = document.getElementById("result-verdict");
  verdictEl.textContent = isSpam ? "SPAM Detected!" : "HAM (Legitimate)";
  verdictEl.className   = "result-verdict " + (isSpam ? "spam-verdict" : "ham-verdict");
  document.getElementById("result-confidence").textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
  const banner = document.getElementById("verdict-banner");
  document.getElementById("verdict-banner-text").textContent = isSpam
    ? "⚠️ Warning: This email shows strong spam signals. Do not click any links or share personal info."
    : "✅ This email appears to be legitimate and does not show common spam patterns.";
  banner.className = "verdict-banner " + (isSpam ? "spam-banner" : "ham-banner");
  document.getElementById("spam-pct").textContent = data.spam_prob.toFixed(1) + "%";
  document.getElementById("ham-pct").textContent  = data.ham_prob.toFixed(1)  + "%";
  const spamBar = document.getElementById("spam-bar");
  const hamBar  = document.getElementById("ham-bar");
  spamBar.style.width = "0%"; hamBar.style.width = "0%";
  requestAnimationFrame(() => setTimeout(() => {
    spamBar.style.width = data.spam_prob + "%";
    hamBar.style.width  = data.ham_prob  + "%";
  }, 50));
  document.getElementById("detail-chips").innerHTML = `
    <div class="chip"><span>${isSpam?"⛔":"🔒"}</span> ${isSpam?"High Risk":"Safe"}</div>
    <div class="chip"><span>📊</span> ${isSpam?"Spam":"Ham"}: ${data.confidence.toFixed(1)}%</div>
    <div class="chip"><span>📝</span> ${data.word_count} words</div>
    <div class="chip"><span>🧠</span> TF-IDF + LR</div>`;
  resultCard.scrollIntoView({ behavior:"smooth", block:"nearest" });
}

function addToHistory(data, text) {
  history.unshift({ data, text });
  historyCard.classList.remove("hidden");
  historyList.innerHTML = history.slice(0,8).map((e,i) => {
    const preview = e.text.replace(/\s+/g," ").substring(0,80)+"…";
    const label   = e.data.is_spam ? "spam" : "ham";
    return `<div class="history-item">
      <span class="history-badge ${label}">${label}</span>
      <span class="history-text">${esc(preview)}</span>
      <span class="history-conf">${e.data.confidence.toFixed(1)}%</span>
    </div>`;
  }).join("");
}

function setLoading(on) {
  btnAnalyze.disabled      = on || !MODEL;
  btnLoader.classList.toggle("hidden", !on);
  btnIcon.style.visibility = on ? "hidden" : "visible";
  btnText.textContent      = on ? "Analyzing…" : "Check This Email";
}
function shake(el) {
  el.style.animation = "none"; el.offsetHeight;
  el.style.animation = "shake 0.4s ease";
  el.addEventListener("animationend", () => { el.style.animation=""; }, { once:true });
}
function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
const kf = document.createElement("style");
kf.textContent=`@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-8px)}40%{transform:translateX(8px)}60%{transform:translateX(-5px)}80%{transform:translateX(5px)}}`;
document.head.appendChild(kf);
