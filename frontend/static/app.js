"use strict";
const API_BASE = "/api";
const $ = (id) => document.getElementById(id);
let busy = false;
let previousFocus;
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = String(text ?? "");
  if (className) el.className = className;
  return el;
}
function message(id, text, type = "info") {
  const el = $(id);
  el.hidden = false;
  el.style.display = "block";
  el.replaceChildren(node("div", text, `alert alert-${type}`));
}
function notice(text, type = "info") {
  message("notice", text, type);
}
function number(value, digits = 4) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : String(value ?? "");
}
function query(params) {
  return new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null),
  ).toString();
}
async function request(path, options = {}) {
  const response = await fetch(API_BASE + path, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Unexpected server response (${response.status})`);
  }
  if (!response.ok || data.success === false) {
    const detail =
      data.detail || data.message || `Request failed (${response.status})`;
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail),
    );
  }
  return data;
}
function post(path, data) {
  return request(path, {
    method: "POST",
    ...(data === undefined
      ? {}
      : {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        }),
  });
}
async function action(label, callback, output = "notice") {
  if (busy) return;
  busy = true;
  const buttons = [...document.querySelectorAll("button")].map((el) => [
    el,
    el.disabled,
  ]);
  buttons.forEach(([el]) => (el.disabled = true));
  document.body.setAttribute("aria-busy", "true");
  message(output, `${label}...`);
  try {
    await callback();
  } catch (error) {
    message(output, error.message, "danger");
  } finally {
    busy = false;
    buttons.forEach(([el, disabled]) => (el.disabled = disabled));
    document.body.removeAttribute("aria-busy");
  }
}
function options(id, values, selected) {
  const select = $(id);
  if (!select) return;
  const old = selected ?? select.value;
  select.replaceChildren(
    ...values.map((value) => new Option(String(value), String(value))),
  );
  if (values.map(String).includes(old)) select.value = old;
}
function table(id, columns, rows) {
  const el = $(id);
  const head = node("tr");
  columns.forEach((col) => head.append(node("th", col)));
  el.querySelector("thead").replaceChildren(head);
  const body = rows.map((row) => {
    const tr = node("tr");
    row.forEach((value) => {
      const td = node("td", value);
      td.title = String(value ?? "");
      tr.append(td);
    });
    return tr;
  });
  if (!body.length) {
    const tr = node("tr"),
      td = node("td", "No rows available.");
    td.colSpan = Math.max(columns.length, 1);
    tr.append(td);
    body.push(tr);
  }
  el.querySelector("tbody").replaceChildren(...body);
}
function listResult(id, title, items, type = "info") {
  const box = node("div", undefined, `alert alert-${type}`);
  box.append(node("strong", title));
  const ul = node("ul");
  items.forEach((item) => ul.append(node("li", item)));
  box.append(ul);
  $(id).style.display = "block";
  $(id).replaceChildren(box);
}
function metrics(id, title, data) {
  listResult(
    id,
    title,
    Object.entries(data || {})
      .filter(([, v]) => typeof v === "number")
      .map(([k, v]) => `${k}: ${number(v)}`),
  );
}
function updateStatus(text) {
  $("status").replaceChildren(node("span", text, "badge-status"));
}
function showModal(id) {
  previousFocus = document.activeElement;
  $(id).hidden = false;
  $(id).querySelector(".modal-content").focus();
}
function hideModal(id) {
  $(id).hidden = true;
  previousFocus?.focus();
}
document
  .querySelectorAll("[data-bs-dismiss]")
  .forEach((el) =>
    el.addEventListener("click", () => hideModal(el.closest(".modal").id)),
  );
document.addEventListener("keydown", (event) => {
  const modal = [...document.querySelectorAll(".modal")].find(
    (el) => !el.hidden,
  );
  if (!modal) return;
  if (event.key === "Escape") hideModal(modal.id);
  if (event.key === "Tab") {
    const controls = [
      ...modal.querySelectorAll("button,select,input,[tabindex='0']"),
    ].filter((el) => !el.disabled);
    const first = controls[0],
      last = controls.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
});
document.querySelectorAll(".nav-link").forEach((link) =>
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((el) => {
      el.classList.remove("active");
      el.removeAttribute("aria-current");
    });
    link.classList.add("active");
    link.setAttribute("aria-current", "page");
    document
      .querySelectorAll(".section")
      .forEach((el) =>
        el.classList.toggle("active", el.id === link.dataset.section),
      );
    const loaders = {
      explore: loadPreview,
      predict: loadPredictForm,
      evaluate: loadModels,
      feature: loadFeatureStatus,
    };
    loaders[link.dataset.section]?.();
  }),
);
const uploadArea = $("uploadArea"),
  fileInput = $("fileInput");
uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () =>
  uploadArea.classList.remove("dragover"),
);
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});
async function uploadFile(file) {
  if (file.size > 50 * 1024 * 1024) {
    message(
      "uploadResult",
      "This file exceeds the 50 MB upload limit.",
      "danger",
    );
    return;
  }
  await action(
    "Loading dataset",
    async () => {
      const form = new FormData();
      form.append("file", file);
      const r = await request("/upload", { method: "POST", body: form });
      listResult(
        "uploadResult",
        `${file.name} loaded`,
        [
          `Shape: ${r.shape}`,
          `Columns: ${(r.columns || []).slice(0, 8).join(", ")}`,
        ],
        "success",
      );
      updateStatus("Dataset ready");
      $("predictResult").replaceChildren();
      $("trainResult").replaceChildren();
      $("comparisonResult").replaceChildren();
      $("modelStatus").textContent = "No trained model";
      await loadColumns();
      await loadModels();
    },
    "uploadResult",
  );
  fileInput.value = "";
}
async function loadPreview() {
  try {
    const r = await request("/preview?rows=10");
    table("previewTable", r.columns, r.data);
  } catch (e) {
    notice(e.message, "warning");
  }
}
async function loadColumns() {
  try {
    const r = await request("/columns");
    ["targetColumn", "chartColumn"].forEach((id) => options(id, r.columns));
  } catch (e) {
    notice(e.message, "warning");
  }
}
async function cleanData(operation) {
  await action(
    "Preparing dataset",
    async () => {
      const r = await post(
        "/clean?" + query({ operation, target: $("targetColumn").value }),
      );
      const items = r.steps || [
        r.encoded
          ? `Encoded: ${r.encoded.join(", ")}`
          : r.removed !== undefined
            ? `Removed: ${r.removed}`
            : r.message || "Operation complete",
      ];
      listResult("cleanResult", "Data preparation complete", items, "success");
      await loadColumns();
      await loadModels();
      updateStatus("Dataset updated; retrain models");
    },
    "cleanResult",
  );
}
async function importance(path, title) {
  await action(
    "Calculating feature scores",
    async () => {
      const r = await request(path);
      const entries = Object.entries(r.importance || {}).slice(0, 15);
      listResult(
        "featureImportanceResult",
        title,
        entries.map(([name, value]) => `${name}: ${number(value, 6)}`),
      );
      const b = node("button", "Keep top features", "btn btn-primary");
      b.onclick = keepTopFeatures;
      $("featureImportanceResult").append(b);
    },
    "featureImportanceResult",
  );
}
function getRFImportance() {
  return importance(
    "/feature-importance-rf",
    "Random Forest feature importance",
  );
}
function getMutualInfo() {
  return importance("/mutual-info", "Mutual information scores");
}
async function applyManualSelection() {
  const features = $("manualFeatures")
    .value.split(",")
    .map((v) => v.trim())
    .filter(Boolean);
  if (!features.length) {
    notice("Enter feature names separated by commas.", "warning");
    return;
  }
  await action("Selecting features", async () => {
    const r = await post("/feature/select", { features });
    notice(`Selected ${r.count} features.`, "success");
    await loadFeatureStatus();
  });
}
async function applyPCA() {
  await action("Configuring PCA", async () => {
    const r = await post(
      "/feature/pca?" + query({ components: $("pcaComponents").value }),
    );
    notice(
      `PCA configured. Explained variance: ${number(r.total_variance * 100, 2)}%.`,
      "success",
    );
    await loadFeatureStatus();
  });
}
async function restoreFromPCA() {
  await action("Restoring features", async () => {
    await post("/feature/restore");
    notice("Original feature configuration restored.", "success");
    await loadFeatureStatus();
  });
}
async function keepTopFeatures() {
  const k = prompt("Number of top features to keep:", "10");
  if (k === null) return;
  if (!/^\d+$/.test(k) || Number(k) < 1) {
    notice("Enter a positive whole number.", "warning");
    return;
  }
  await action("Selecting top features", async () => {
    await post("/feature/keep_top?" + query({ k }));
    notice(`Selected top ${k} features.`, "success");
    await loadFeatureStatus();
  });
}
async function loadFeatureStatus() {
  try {
    const r = await request("/feature/status");
    message(
      "featureStatus",
      `Using ${r.selected} of ${r.total} features${r.pca ? " / PCA configured" : ""}`,
    );
  } catch (e) {
    message("featureStatus", e.message, "warning");
  }
}
const updateFeatureStatus = loadFeatureStatus;
function trainSettings() {
  return {
    target: $("targetColumn").value,
    model: $("modelType").value,
    test_size: Number($("testSize").value),
    use_smote: $("useSmote").checked,
    problem_type: $("problemType").value,
    n_estimators: Number($("nEstimators").value),
    max_depth: Number($("maxDepth").value),
    learning_rate: Number($("learningRate").value),
  };
}
async function trainModel() {
  if (!$("targetColumn").value) {
    notice("Upload data and select a target column first.", "warning");
    return;
  }
  await action(
    "Training model",
    async () => {
      const r = await post("/train", trainSettings());
      metrics("trainResult", `${r.model} / ${r.problem_type}`, r.metrics);
      updateStatus("Model ready");
      await loadModels();
      await loadPredictForm();
    },
    "trainResult",
  );
}
async function compareAllModels() {
  if (!$("targetColumn").value) {
    notice("Select a target column first.", "warning");
    return;
  }
  await action(
    "Comparing models with training cross-validation",
    async () => {
      const r = await post("/train/compare", trainSettings());
      const box = $("comparisonResult");
      box.replaceChildren(
        node("h3", "Model comparison / training cross-validation"),
      );
      const wrap = node("div", undefined, "table-responsive"),
        t = node("table");
      wrap.tabIndex = 0;
      wrap.setAttribute("role", "region");
      wrap.setAttribute("aria-label", "Model comparison table");
      t.id = "leaderboardTable";
      t.append(node("thead"), node("tbody"));
      wrap.append(t);
      box.append(wrap, node("p", `Best model: ${r.best_model}`));
      table(
        t.id,
        ["Rank", "Model", "Score"],
        r.leaderboard.map((m, i) => [i + 1, m.model, number(m.score)]),
      );
      await loadModels();
    },
    "comparisonResult",
  );
}
async function showStackingDialog() {
  try {
    const r = await request("/models");
    if (r.models.length < 2) {
      notice("Train at least two different models first.", "warning");
      return;
    }
    options("stackBase1", r.models);
    options("stackBase2", r.models, r.models[1]);
    showModal("stackingModal");
  } catch (e) {
    notice(e.message, "danger");
  }
}
async function trainStackedModel() {
  const a = $("stackBase1").value,
    b = $("stackBase2").value;
  if (a === b) {
    notice("Choose different base models.", "warning");
    return;
  }
  hideModal("stackingModal");
  await action("Training stacked model", async () => {
    const r = await post("/model/stack", {
      base_model_1: a,
      base_model_2: b,
      meta_model: $("stackMeta").value,
    });
    notice(
      `Stacked model trained with ${r.base_models.join(" + ")}.`,
      "success",
    );
    await loadModels();
  });
}
async function loadModels() {
  try {
    const r = await request("/models");
    options("modelSelector", r.models || [], r.current_model);
    $("modelStatus").textContent = r.current_model
      ? `Active model: ${r.current_model}`
      : "No trained model";
    if (!r.current_model) {
      $("metricsDisplay").style.display = "none";
      $("evalChartContainer").style.display = "none";
    }
  } catch (e) {
    notice(e.message, "warning");
  }
}
async function loadSelectedModel() {
  await action("Selecting model", async () => {
    const name = $("modelSelector").value;
    if (!name) throw new Error("Train a model first.");
    await post("/model/select", { model_name: name });
    await loadModels();
    await loadMetrics();
    notice(`Active model: ${name}`, "success");
  });
}
async function loadMetrics() {
  try {
    const r = await request("/evaluate");
    metrics("metricsDisplay", "Held-out model metrics", r.metrics);
  } catch (e) {
    message("metricsDisplay", e.message, "warning");
  }
}
async function chart(type, explore = false) {
  await action("Generating chart", async () => {
    const r = await post(
      "/chart/" +
        encodeURIComponent(type) +
        "?" +
        query({ column: $("chartColumn").value }),
    );
    const id = explore ? "chart" : "evalChart";
    $(id + "Image").src = "data:image/png;base64," + r.image;
    $(id + "Image").alt = `${type} chart`;
    $(id + "Container").style.display = "block";
    notice("Chart ready.", "success");
  });
}
function generateChart(type) {
  return chart(type);
}
function generateExploreChart() {
  return chart($("chartType").value, true);
}
async function getPermutationImportance() {
  await action(
    "Calculating permutation importance",
    async () => {
      const r = await request("/permutation-importance");
      listResult(
        "metricsDisplay",
        "Permutation importance",
        (r.importance || [])
          .slice(0, 15)
          .map(
            (v) =>
              `${v.feature}: ${number(v.importance)} (standard deviation ${number(v.std)})`,
          ),
      );
    },
    "metricsDisplay",
  );
}
async function getCrossValidation() {
  await action(
    "Running cross-validation",
    async () => {
      const r = await request("/cross-validation");
      listResult("metricsDisplay", `${r.cv || 5}-fold cross-validation`, [
        `Mean score: ${number(r.mean)}`,
        `Standard deviation: ${number(r.std)}`,
        `Fold scores: ${(r.scores || []).map((v) => number(v)).join(", ")}`,
      ]);
    },
    "metricsDisplay",
  );
}
async function showMisclassified() {
  await action("Finding misclassified samples", async () => {
    const r = await request("/misclassified");
    if (!r.samples?.length) {
      notice("No misclassified samples found.", "success");
      return;
    }
    const columns = Object.keys(r.samples[0]);
    table(
      "misclassifiedTable",
      columns,
      r.samples.map((row) =>
        columns.map((c) =>
          typeof row[c] === "number" ? number(row[c], 2) : row[c],
        ),
      ),
    );
    showModal("misclassifiedModal");
    notice(`${r.samples.length} misclassified samples shown.`);
  });
}
async function getFeatureImportance() {
  await action("Generating importance chart", async () => {
    const r = await request("/feature-importance-plot");
    $("evalChartImage").src = "data:image/png;base64," + r.image;
    $("evalChartImage").alt = "Feature importance chart";
    $("evalChartContainer").style.display = "block";
    notice("Chart ready.", "success");
  });
}
async function loadPredictForm() {
  $("predictResult").replaceChildren();
  try {
    const r = await request("/input-schema");
    const children = (r.schema || []).map((field, i) => {
      const wrap = node("div", undefined, "mb-3"),
        label = node("label", field.name, "form-label");
      const input = node(field.type === "categorical" ? "select" : "input");
      input.className =
        field.type === "categorical" ? "form-select" : "form-control";
      input.id = `predict-field-${i}`;
      label.htmlFor = input.id;
      input.dataset.feature = field.name;
      if (field.type === "categorical") {
        (field.options || []).forEach((v) =>
          input.add(new Option(String(v), String(v))),
        );
      } else {
        input.type = "number";
        input.step = "any";
        input.value = field.mean ?? 0;
      }
      wrap.append(label, input);
      return wrap;
    });
    $("predictForm").replaceChildren(
      ...(children.length
        ? children
        : [
            node(
              "p",
              "Train a model first to enable predictions.",
              "empty-state",
            ),
          ]),
    );
  } catch (e) {
    message("predictForm", e.message, "warning");
  }
}
async function makePrediction() {
  await action(
    "Making prediction",
    async () => {
      const inputs = Object.create(null);
      document
        .querySelectorAll("#predictForm [data-feature]")
        .forEach((el) => (inputs[el.dataset.feature] = el.value));
      if (!Object.keys(inputs).length) throw new Error("Train a model first.");
      const r = await post("/predict", inputs),
        v = r.result;
      const items = [`Prediction: ${v.prediction}`];
      if (v.confidence !== undefined)
        items.push(`Confidence: ${number(v.confidence * 100, 2)}%`);
      Object.entries(v.probabilities || {}).forEach(([k, p]) =>
        items.push(`${k}: ${number(p * 100, 2)}%`),
      );
      listResult("predictResult", "Prediction result", items, "success");
    },
    "predictResult",
  );
}
async function generateInsights() {
  await action("Generating insights", async () => {
    const r = await request("/insights");
    $("insightsList").replaceChildren(...r.insights.map((v) => node("li", v)));
    notice("Insights ready.", "success");
  });
}
async function showHealthScore() {
  await action("Checking dataset health", async () => {
    const r = await request("/health-score");
    listResult(
      "healthScoreContainer",
      `Dataset health: ${r.score}/100 / Grade ${r.grade}`,
      r.issues?.length
        ? r.issues
        : ["No issues detected by the available checks."],
      "info",
    );
    notice("Health check complete.", "success");
  });
}
async function exportReport() {
  await action("Preparing report", async () => {
    const response = await fetch(API_BASE + "/export/report");
    if (!response.ok) {
      let r = {};
      try {
        r = await response.json();
      } catch {}
      throw new Error(r.detail || "Report export failed.");
    }
    const url = URL.createObjectURL(await response.blob());
    const a = node("a");
    a.href = url;
    a.download = "datayab-report.html";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notice("Report downloaded.", "success");
  });
}
async function init() {
  try {
    const r = await request("/status");
    if (r.loaded) {
      updateStatus(`${r.rows} rows / ${r.cols} columns`);
      await loadColumns();
      await loadModels();
    } else updateStatus("No dataset loaded");
  } catch (e) {
    notice(`Cannot connect to DataYab: ${e.message}`, "danger");
  }
}
init();
