const state = {
  task: "ir",
  stage: "warmup",
  config: null,
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatScore(value) {
  return value == null ? "—" : Number(value).toFixed(4);
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value) {
  try {
    return new Date(value).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
  } catch (_) {
    return value;
  }
}

function showFlash(message, kind = "success") {
  const node = $("#flash");
  node.textContent = message;
  node.classList.toggle("error", kind === "error");
  node.classList.remove("hidden");
  window.clearTimeout(showFlash.timer);
  showFlash.timer = window.setTimeout(() => node.classList.add("hidden"), 6500);
}

function renderStageButtons() {
  const stages = state.config?.stages?.[state.task] || {};
  $("#stageButtons").innerHTML = Object.entries(stages).map(([stage, item]) => `
    <button class="stage-button ${stage === state.stage ? "active" : ""}" data-stage="${stage}" type="button">
      ${escapeHtml(item.label)}
    </button>
  `).join("");
  document.querySelectorAll(".stage-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.stage = button.dataset.stage;
      updateView();
    });
  });
}

function renderReferenceNotice() {
  const item = state.config?.stages?.[state.task]?.[state.stage];
  const node = $("#referenceNotice");
  if (!item) {
    node.classList.add("hidden");
    return;
  }
  if (item.reference_available) {
    node.textContent = `Reference sẵn sàng: ${item.reference_count.toLocaleString("vi-VN")} câu hỏi. Điểm sẽ dùng scorer ${state.task === "ir" ? "Recall/Precision BTC" : "METEOR/ROUGE-L"}.`;
    node.classList.remove("hidden");
    return;
  }
  node.textContent = `${item.label}: ${item.reference_error || "chưa có reference độc lập"} Submission vẫn được lưu để chấm sau.`;
  node.classList.remove("hidden");
}

function renderHeaders() {
  const qa = state.task === "qa";
  $("#leaderboardHead").innerHTML = `
    <tr>
      <th>#</th><th>Tên lần chạy</th><th>File</th><th>Ngày</th><th>Trạng thái</th>
      <th>${qa ? "METEOR" : "Recall"}</th><th>${qa ? "ROUGE-L" : "Precision"}</th><th></th>
    </tr>`;
}

function renderRows(items) {
  const body = $("#leaderboardBody");
  const empty = $("#emptyState");
  empty.classList.toggle("hidden", items.length > 0);
  body.innerHTML = items.map((item) => `
    <tr>
      <td class="rank">${item.rank ?? "—"}</td>
      <td><strong>${escapeHtml(item.label)}</strong><br><small>${escapeHtml(item.sha256.slice(0, 12))}</small></td>
      <td>${escapeHtml(item.filename)}<br><small>${formatBytes(item.size_bytes)}</small></td>
      <td>${escapeHtml(formatDate(item.uploaded_at))}</td>
      <td><span class="status ${escapeHtml(item.status)}">${item.status === "scored" ? "Đã chấm" : "Chờ reference"}</span></td>
      <td class="score">${formatScore(item.metrics?.primary)}</td>
      <td class="score">${formatScore(item.secondary)}</td>
      <td><button class="delete-button" title="Xóa lần nộp" data-delete="${escapeHtml(item.id)}">×</button></td>
    </tr>`).join("");
  body.querySelectorAll("[data-delete]").forEach((button) => {
    button.addEventListener("click", () => deleteSubmission(button.dataset.delete));
  });
}

function updateSummary(items) {
  const scored = items.filter((item) => item.status === "scored");
  const best = scored.length ? Math.max(...scored.map((item) => item.metrics.primary)) : null;
  const stageLabel = state.config?.stages?.[state.task]?.[state.stage]?.label || state.stage;
  const taskLabel = state.config?.tasks?.[state.task] || state.task;
  $("#taskTitle").textContent = taskLabel;
  $("#summaryTitle").textContent = `${taskLabel} · ${stageLabel}`;
  $("#currentStageLabel").textContent = stageLabel;
  $("#submissionCount").textContent = items.length;
  $("#scoredCount").textContent = scored.length;
  $("#bestScore").textContent = formatScore(best);
}

async function loadSubmissions() {
  const response = await fetch(`/api/submissions?task=${encodeURIComponent(state.task)}&stage=${encodeURIComponent(state.stage)}`);
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || "Không tải được bảng xếp hạng");
  renderHeaders();
  renderRows(payload.submissions);
  updateSummary(payload.submissions);
}

async function updateView() {
  document.querySelectorAll(".task-tab").forEach((button) => button.classList.toggle("active", button.dataset.task === state.task));
  renderStageButtons();
  renderReferenceNotice();
  try {
    await loadSubmissions();
  } catch (error) {
    showFlash(error.message, "error");
  }
}

async function deleteSubmission(id) {
  if (!window.confirm("Xóa lần submission này khỏi bảng local?")) return;
  const response = await fetch(`/api/submissions/${encodeURIComponent(id)}`, { method: "DELETE" });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    showFlash(payload.error || "Không xóa được submission", "error");
    return;
  }
  showFlash("Đã xóa submission.");
  await updateView();
}

async function uploadSubmission(event) {
  event.preventDefault();
  const file = $("#submissionFile").files[0];
  if (!file) return;
  const button = $("#uploadButton");
  button.disabled = true;
  button.textContent = "Đang chấm...";
  const form = new FormData();
  form.append("task", state.task);
  form.append("stage", state.stage);
  form.append("label", $("#label").value);
  form.append("file", file, file.name);
  try {
    const response = await fetch("/api/submissions", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Upload thất bại");
    const result = payload.submission;
    showFlash(result.status === "scored" ? `Đã chấm ${result.label}: ${formatScore(result.primary)}` : "Đã lưu submission; stage chưa có reference để chấm.");
    $("#uploadForm").reset();
    $("#fileName").textContent = "Chọn file để tải lên";
    await updateView();
  } catch (error) {
    showFlash(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Chấm submission";
  }
}

async function boot() {
  try {
    const response = await fetch("/api/config");
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Không tải được cấu hình");
    state.config = payload;
    $("#health").innerHTML = '<span class="dot"></span> Backend đang chạy';
    await updateView();
  } catch (error) {
    $("#health").innerHTML = '<span class="dot" style="background:#bd4a4a"></span> Không kết nối được';
    showFlash(`Không kết nối được backend: ${error.message}`, "error");
  }
}

document.querySelectorAll(".task-tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.task = button.dataset.task;
    state.stage = "warmup";
    updateView();
  });
});
$("#uploadForm").addEventListener("submit", uploadSubmission);
$("#submissionFile").addEventListener("change", (event) => {
  $("#fileName").textContent = event.target.files[0]?.name || "Chọn file để tải lên";
});
$("#refreshButton").addEventListener("click", updateView);
boot();
