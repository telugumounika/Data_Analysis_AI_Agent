// Frontend logic. Talks to the FastAPI backend with fetch().
// Same origin as the page (http://localhost:8000), so API_BASE is empty.
// If you open index.html from another server (e.g. Live Server on :5500), set it to "http://localhost:8000".
const API_BASE = "";

// ------------------------------------------------------------------ state
let datasetId = null;
let history = [];            // [{role, content}] sent to the backend so follow-up questions work
let trackedQuestions = [];   // analyses we re-run automatically when the data changes
let lastOverview = null;
let editPage = 0;
const PAGE_SIZE = 25;
const pending = new Map();   // "row|column" -> new text typed by the user (not saved yet)
const selectedRows = new Set();
let busy = false;
let warnedLLM = false;
let formCounter = 0;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- helpers
async function readError(res) {
  try {
    return (await res.json()).detail || "Request failed";
  } catch {
    return "Request failed";
  }
}

async function api(path, options = {}) {
  const res = await fetch(API_BASE + path, options);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

const postJSON = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

let toastTimer;
function toast(text, ms = 3800) {
  const t = $("toast");
  t.textContent = text;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}

function setMsg(text, kind) {
  const el = $("uploadMsg");
  el.textContent = text;
  el.className = "msg " + (kind || "");
}

function tableEl(table, title) {
  const wrap = document.createElement("div");
  if (title) {
    const t = document.createElement("div");
    t.className = "tabletitle";
    t.textContent = title.replaceAll("_", " ");
    wrap.appendChild(t);
  }
  const box = document.createElement("div");
  box.className = "tablewrap";
  const tbl = document.createElement("table");
  const head = tbl.createTHead().insertRow();
  table.columns.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    head.appendChild(th);
  });
  const body = tbl.createTBody();
  table.rows.forEach((r) => {
    const tr = body.insertRow();
    r.forEach((v) => (tr.insertCell().textContent = v === null ? "—" : v));
  });
  box.appendChild(tbl);
  wrap.appendChild(box);
  return wrap;
}

// ----------------------------------------------------------------- health
api("/api/health")
  .then((d) => {
    const el = $("status");
    if (!d.llm_enabled) {
      el.textContent = "Rule-based mode";
      el.title = "USE_LLM=false in backend/.env";
      return;
    }
    const l = d.llm || {};
    if (l.reachable && l.model_ok) {
      el.textContent = "LLM connected";
      el.className = "badge ok";
      el.title = `${d.llm_model} @ ${d.llm_url}`;
    } else if (l.reachable) {
      el.textContent = "LLM up, model not found";
      el.className = "badge bad";
      el.title = `LLM_MODEL is "${d.llm_model}" but the server has: ${l.models.join(", ")}. Fix LLM_MODEL in backend/.env`;
    } else {
      el.textContent = "LLM offline → rule mode";
      el.className = "badge bad";
      el.title = `Cannot reach ${d.llm_url}. Start the LM Studio server (Developer tab) or set USE_LLM=false.`;
    }
  })
  .catch(() => ($("status").textContent = "Backend offline"));

// ----------------------------------------------------------------- upload
$("uploadBtn").addEventListener("click", async () => {
  const file = $("fileInput").files[0];
  if (!file) return setMsg("Please choose a file first.", "err");

  const form = new FormData(); // FormData = how browsers send files
  form.append("file", file);
  $("uploadBtn").disabled = true;
  setMsg("Uploading…");
  try {
    const data = await api("/api/upload", { method: "POST", body: form });
    datasetId = data.dataset_id;
    history = [];
    trackedQuestions = [];
    pending.clear();
    selectedRows.clear();
    editPage = 0;
    showDashboard(data);
    await loadEditPage();
    setMsg(`Loaded ${data.filename}`, "ok");
  } catch (e) {
    setMsg(e.message, "err");
  } finally {
    $("uploadBtn").disabled = false;
  }
});

function renderOverview(d) {
  lastOverview = d;
  datasetId = d.dataset_id;
  $("datasetName").textContent = d.filename;
  $("statRows").textContent = d.rows.toLocaleString();
  $("statCols").textContent = d.columns;
  $("statMissing").textContent = d.missing_values.toLocaleString();
  $("statDup").textContent = d.duplicate_rows;
  $("statNum").textContent = d.numeric_columns.length;
  $("statCat").textContent = d.categorical_columns.length;

  $("tab-preview").replaceChildren(tableEl(d.preview, "First rows"));
  $("tab-stats").replaceChildren(d.statistics ? tableEl(d.statistics, "Numeric statistics") : "No numeric columns.");
  $("tab-columns").replaceChildren(
    tableEl(
      { columns: ["Column", "Type", "Missing", "Unique"], rows: d.column_details.map((c) => [c.name, c.type, c.missing, c.unique]) },
      "Columns"
    )
  );

  const vb = $("versionBadge");
  vb.textContent = d.version > 1 ? `v${d.version} · edited` : "v1 · original";
  vb.classList.remove("hidden");

  $("dlCsv").href = `${API_BASE}/api/dataset/${d.dataset_id}/download?format=csv`;
  $("dlXlsx").href = `${API_BASE}/api/dataset/${d.dataset_id}/download?format=xlsx`;
  $("changeLog").replaceChildren(
    ...[...d.change_log].reverse().map((c) => {
      const li = document.createElement("li");
      li.textContent = `v${c.version}: ${c.message}`;
      return li;
    })
  );
  updateEditButtons();
}

function showDashboard(d) {
  $("dashboard").classList.remove("hidden");
  renderOverview(d);
  $("insightsText").textContent = "";
  $("fillHost").replaceChildren();
  $("question").disabled = false;
  $("sendBtn").disabled = false;
  $("messages").replaceChildren();
  addBubble("ai", `Dataset loaded: ${d.rows} rows, ${d.columns} columns. What would you like to know?`);
  showSuggestions(d);
}

function showSuggestions(d) {
  const box = $("suggestions");
  const qs = ["Are there any missing values?", "Find unusual values or outliers.", "Summarize the important insights."];
  if (d.numeric_columns.length) qs.unshift(`What is the total ${d.numeric_columns[d.numeric_columns.length - 1]}?`);
  if (d.categorical_columns.length && d.numeric_columns.length)
    qs.push(`Create a chart showing ${d.numeric_columns[d.numeric_columns.length - 1]} by ${d.categorical_columns[0]}`);
  if (d.missing_values > 0) qs.push("Fill missing values");
  box.replaceChildren(
    ...qs.map((q) => {
      const b = document.createElement("button");
      b.textContent = q;
      b.type = "button";
      b.onclick = () => sendQuestion(q);
      return b;
    })
  );
  box.classList.remove("hidden");
}

// ------------------------------------------------------------------- tabs
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tabcontent").forEach((c) => c.classList.add("hidden"));
    $("tab-" + btn.dataset.tab).classList.remove("hidden");
  })
);

async function refreshInsights() {
  $("insightsText").textContent = "Analysing…";
  try {
    $("insightsText").textContent = (await postJSON("/api/insights", { dataset_id: datasetId })).insights;
  } catch (e) {
    $("insightsText").textContent = "Error: " + e.message;
  }
}
$("insightsBtn").addEventListener("click", refreshInsights);

// ------------------------------------------------------------ edit table
async function loadEditPage() {
  const d = await api(`/api/dataset/${datasetId}/data?page=${editPage}&page_size=${PAGE_SIZE}`);
  if (d.rows.length === 0 && editPage > 0) {
    editPage = Math.max(0, Math.ceil(d.total_rows / PAGE_SIZE) - 1);
    return loadEditPage();
  }
  renderEditTable(d);
}

function renderEditTable(d) {
  const tbl = document.createElement("table");
  tbl.className = "edittable";
  const head = tbl.createTHead().insertRow();
  ["", ...d.columns].forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    head.appendChild(th);
  });
  const body = tbl.createTBody();

  d.rows.forEach((row, i) => {
    const pos = d.start + i;
    const tr = body.insertRow();

    const sel = tr.insertCell();
    sel.className = "rowsel";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedRows.has(pos);
    cb.title = "Select row to delete";
    cb.onchange = () => {
      cb.checked ? selectedRows.add(pos) : selectedRows.delete(pos);
      updateEditButtons();
    };
    sel.append(cb, " " + (pos + 1));

    d.columns.forEach((col, j) => {
      const orig = row[j] === null ? "" : String(row[j]);
      const key = `${pos}|${col}`;
      const td = tr.insertCell();
      const input = document.createElement("input");
      input.className = "cell";
      input.placeholder = "∅";
      input.value = pending.has(key) ? pending.get(key) : orig;
      const paint = () => {
        td.classList.toggle("edited", pending.has(key));
        td.classList.toggle("missing", input.value === "");
      };
      input.addEventListener("input", () => {
        input.value === orig ? pending.delete(key) : pending.set(key, input.value);
        paint();
        updateEditButtons();
      });
      paint();
      td.appendChild(input);
    });
  });

  const box = document.createElement("div");
  box.className = "tablewrap";
  box.style.maxHeight = "380px";
  box.appendChild(tbl);
  $("editTable").replaceChildren(box);

  $("pageLabel").textContent = `Rows ${d.start + 1}–${d.start + d.rows.length} of ${d.total_rows}`;
  $("prevPage").disabled = editPage === 0;
  $("nextPage").disabled = d.start + d.rows.length >= d.total_rows;
  updateEditButtons();
}

function updateEditButtons() {
  const n = pending.size;
  $("saveBtn").textContent = n ? `💾 Save changes (${n})` : "💾 Save changes";
  $("saveBtn").disabled = busy || n === 0;
  $("discardBtn").disabled = busy || n === 0;
  $("deleteRowsBtn").textContent = selectedRows.size ? `🗑 Delete ${selectedRows.size} row(s)` : "🗑 Delete selected rows";
  $("deleteRowsBtn").disabled = busy || selectedRows.size === 0;
  $("undoBtn").disabled = busy || !(lastOverview && lastOverview.can_undo);
  $("fillBtn").disabled = busy;
  $("replaceBtn").disabled = busy;
}

$("prevPage").onclick = () => { editPage = Math.max(0, editPage - 1); loadEditPage(); };
$("nextPage").onclick = () => { editPage += 1; loadEditPage(); };
$("discardBtn").onclick = () => { pending.clear(); loadEditPage(); };

$("saveBtn").onclick = () => {
  const edits = [...pending].map(([k, value]) => {
    const i = k.indexOf("|");
    return { row: Number(k.slice(0, i)), column: k.slice(i + 1), value };
  });
  applyUpdate(() => postJSON(`/api/dataset/${datasetId}/edit`, { edits, delete_rows: [] }));
};

$("deleteRowsBtn").onclick = () => {
  if (pending.size) return toast("Save or discard your cell edits first.");
  if (!confirm(`Delete ${selectedRows.size} row(s)? You can undo this afterwards.`)) return;
  applyUpdate(() => postJSON(`/api/dataset/${datasetId}/edit`, { edits: [], delete_rows: [...selectedRows] }));
};

$("undoBtn").onclick = () => applyUpdate(() => postJSON(`/api/dataset/${datasetId}/undo`));

$("fillBtn").onclick = () => {
  if (!lastOverview.fill_options.columns.length) return toast("No missing values to fill 🎉");
  renderFillForm($("fillHost"), lastOverview.fill_options, {});
};

$("replaceBtn").onclick = () => $("replaceInput").click();
$("replaceInput").onchange = () => {
  const file = $("replaceInput").files[0];
  $("replaceInput").value = "";
  if (!file) return;
  if (!confirm(`Replace the current data with "${file.name}"? Your analysis will re-run. (Undo is available.)`)) return;
  const form = new FormData();
  form.append("file", file);
  applyUpdate(() => api(`/api/dataset/${datasetId}/replace`, { method: "POST", body: form }));
};

["dlCsv", "dlXlsx"].forEach((id) =>
  $(id).addEventListener("click", (e) => {
    if (pending.size) {
      e.preventDefault();
      toast("You have unsaved edits. Save them first so the download includes them.");
    }
  })
);

// -------------------------------------------------- updating the dataset
function setBusy(on) {
  busy = on;
  updateEditButtons();
}

// Runs one data-changing request, then refreshes everything. Returns true on success.
async function applyUpdate(request) {
  if (busy) return false;
  setBusy(true);
  try {
    const ov = await request();
    pending.clear();
    selectedRows.clear();
    await onDataUpdated(ov);
    return true;
  } catch (e) {
    toast("⚠️ " + e.message, 6000);
    return false;
  } finally {
    setBusy(false);
  }
}

// The "automatic analysis again" part.
async function onDataUpdated(ov) {
  renderOverview(ov);
  toast(`✅ ${ov.message}`);
  document.querySelectorAll(".bubble.result:not(.stale)").forEach((b) => b.classList.add("stale"));
  await loadEditPage();

  const rerun = trackedQuestions.slice(-3);
  const note = addBubble(
    "ai",
    `🔄 Data updated (v${ov.version}): ${ov.message}\nRefreshing dashboard and insights${rerun.length ? " and re-running your last analyses" : ""}…`,
    "refresh"
  );
  await refreshInsights();
  for (const q of rerun) {
    try {
      const data = await postJSON("/api/ask", { dataset_id: datasetId, question: q, history: [] });
      if (!data.needs_input) renderAnswer(data, `↻ ${q}`);
    } catch (e) {
      addBubble("ai", `↻ ${q}\nCouldn't re-run: ${e.message}`, "result");
    }
  }
  note.firstChild.textContent = `🔄 Data updated (v${ov.version}): ${ov.message}\nDashboard, insights${rerun.length ? " and your last analyses were" : " were"} refreshed.`;
}

// --------------------------------------------------------- fill-missing form
function renderFillForm(host, opts, preset) {
  const id = ++formCounter;
  const form = document.createElement("div");
  form.className = "fillform";

  const colLabel = document.createElement("label");
  colLabel.className = "field";
  colLabel.textContent = "1. Which column?";
  const colSel = document.createElement("select");
  const total = opts.columns.reduce((s, c) => s + c.missing, 0);
  colSel.add(new Option(`All columns with missing values (${total} cells)`, "__all__"));
  opts.columns.forEach((c) => colSel.add(new Option(`${c.name} — ${c.missing} missing (${c.dtype})`, c.name)));
  if (preset.suggested_column) colSel.value = preset.suggested_column;

  const methLabel = document.createElement("label");
  methLabel.className = "field";
  methLabel.textContent = "2. How should they be filled?";
  const methBox = document.createElement("div");
  const radios = {};
  opts.methods.forEach((m) => {
    const row = document.createElement("label");
    row.className = "method";
    const r = document.createElement("input");
    r.type = "radio";
    r.name = "fillmethod" + id;
    r.value = m.id;
    r.checked = preset.suggested_method === m.id;
    const txt = document.createElement("span");
    txt.append(m.label);
    const small = document.createElement("small");
    small.textContent = m.description;
    txt.appendChild(small);
    row.append(r, txt);
    methBox.appendChild(row);
    radios[m.id] = { row, r, meta: m };
  });

  const valLabel = document.createElement("label");
  valLabel.className = "field";
  valLabel.textContent = "3. Value to use";
  const valInput = document.createElement("input");
  valInput.type = "text";
  valInput.placeholder = "e.g. 0 or Unknown";

  const note = document.createElement("div");
  note.className = "hint";

  const apply = document.createElement("button");
  apply.textContent = "Apply";
  const cancel = document.createElement("button");
  cancel.textContent = "Cancel";
  cancel.className = "secondary";
  const actions = document.createElement("div");
  actions.className = "actions";
  actions.append(apply, cancel);

  const chosen = () => Object.values(radios).find((x) => x.r.checked)?.meta;
  const sync = () => {
    const col = opts.columns.find((c) => c.name === colSel.value);
    Object.values(radios).forEach(({ row, r, meta }) => {
      const bad = meta.numeric_only && col && !col.numeric;
      row.classList.toggle("disabled", bad);
      r.disabled = bad;
      if (bad && r.checked) r.checked = false;
    });
    const m = chosen();
    const showVal = m && m.needs_value;
    valLabel.classList.toggle("hidden", !showVal);
    valInput.classList.toggle("hidden", !showVal);
    note.textContent =
      colSel.value === "__all__" ? "For “all columns”, methods that don't fit a column (e.g. mean on text) skip that column." : "";
    apply.disabled = !m;
  };
  colSel.onchange = sync;
  Object.values(radios).forEach(({ r }) => (r.onchange = sync));
  sync();

  apply.onclick = async () => {
    const m = chosen();
    if (m.id === "drop_rows" && !confirm("This deletes whole rows. Continue? (You can undo.)")) return;
    const ok = await applyUpdate(() =>
      postJSON(`/api/dataset/${datasetId}/fill-missing`, {
        column: colSel.value,
        method: m.id,
        value: m.needs_value ? valInput.value : null,
      })
    );
    if (ok) form.classList.add("done");
  };
  cancel.onclick = () => form.remove();

  form.append(colLabel, colSel, methLabel, methBox, valLabel, valInput, note, actions);
  if (host.id === "fillHost") host.replaceChildren(form);   // Edit tab: one form at a time
  else host.appendChild(form);                               // chat: below the question
}

// ------------------------------------------------------------------- chat
function addBubble(role, text, extraClass) {
  const b = document.createElement("div");
  b.className = `bubble ${role} ${extraClass || ""}`;
  b.append(document.createTextNode(text));
  $("messages").appendChild(b);
  $("messages").scrollTop = $("messages").scrollHeight;
  return b;
}

function track(question) {
  trackedQuestions = trackedQuestions.filter((q) => q !== question);
  trackedQuestions.push(question);
  trackedQuestions = trackedQuestions.slice(-10);
}

async function sendQuestion(question) {
  if (!datasetId || !question.trim()) return;
  addBubble("user", question);
  const typing = addBubble("ai", "Thinking…", "typing");
  $("sendBtn").disabled = true;
  try {
    const data = await postJSON("/api/ask", { dataset_id: datasetId, question, history });
    typing.remove();
    renderAnswer(data);
    history.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
    if (!data.needs_input && (data.steps.length || data.charts.length)) track(question);
  } catch (e) {
    typing.remove();
    addBubble("ai", "Error: " + e.message);
  } finally {
    $("sendBtn").disabled = false;
  }
}

function renderAnswer(data, label) {
  if (data.warning && !warnedLLM) {
    warnedLLM = true;
    addBubble("ai", data.warning + " (Shown once. See the badge at the top right.)", "warn");
  }
  const bubble = addBubble("ai", (label ? label + "\n\n" : "") + data.answer, data.needs_input ? "" : "result");
  (data.tables || []).forEach((t) => bubble.appendChild(tableEl(t, t.title)));
  (data.charts || []).forEach((b64) => {
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + b64;
    img.alt = "chart";
    bubble.appendChild(img);
  });
  if (data.needs_input && data.needs_input.type === "fill_missing") {
    renderFillForm(bubble, data.needs_input, data.needs_input);
  }
  if (data.steps && data.steps.length) {
    const d = document.createElement("details");
    const s = document.createElement("summary");
    s.textContent = `How I got this (${data.mode} mode)`;
    d.appendChild(s);
    data.steps.forEach((st) => {
      const c = document.createElement("code");
      c.textContent = `${st.tool}(${JSON.stringify(st.args)})` + (st.error ? `\nError: ${st.error}` : "");
      d.appendChild(c);
    });
    bubble.appendChild(d);
  }
  $("messages").scrollTop = $("messages").scrollHeight;
}

$("chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("question").value;
  $("question").value = "";
  sendQuestion(q);
});
