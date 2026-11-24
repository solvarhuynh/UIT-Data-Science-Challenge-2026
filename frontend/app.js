const state = {
  task: "ir",
  stage: "warmup",
  config: null,
};

const $ = (selector) => document.querySelector(selector);

function taskLabel(task = state.task) {
  return task === "qa" ? "QA" : "IR";
}

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
  showFlash.timer = window.setTimeout(() => node.classList.add("hidden"), 5500);
}

function currentStageConfig() {
  return state.config?.stages?.[state.task]?.[state.stage];
}

function renderTask() {
  const qa = state.task === "qa";
  $("#taskTitle").textContent = taskLabel();
  $("#tableTask").textContent = taskLabel();
  $("#uploadHint").innerHTML = qa
    ? 'QA: mỗi ID cần một câu trả lời trong trường <code>answer</code>.'
    : 'IR: mỗi ID cần tối đa 5 document trong trường <code>answer</code>.';
  $("#leaderboardHead").innerHTML = `
    <tr>
      <th>#</th><th>File submission</th><th>Ngày</th><th>Trạng thái</th>
      <th>${qa ? "METEOR" : "RECALL"}</th><th>${qa ? "ROUGE-L" : "PRECISION"}</th><th></th>
    </tr>`;

  const reference = currentStageConfig();
  const status = $("#referenceStatus");
  if (reference?.reference_available) {
    status.textContent = `${reference.reference_count.toLocaleString("vi-VN")} câu có gold`;
    status.classList.remove("missing");
  } else {
    status.textContent = "Chưa có gold để chấm";
    status.classList.add("missing");
  }
}

function renderRows(items) {
  const body = $("#leaderboardBody");
  const empty = $("#emptyState");
  empty.classList.toggle("hidden", items.length > 0);
  body.innerHTML = items.map((item) => `
    <tr>
      <td class="rank">${item.rank ?? "—"}</td>
      <td><strong>${escapeHtml(item.filename)}</strong><br><small>${formatBytes(item.size_bytes)} · ${escapeHtml(item.sha256.slice(0, 10))}</small></td>
      <td>${escapeHtml(formatDate(item.uploaded_at))}</td>
      <td><span class="status ${escapeHtml(item.status)}">${item.status === "scored" ? "Đã chấm" : "Chờ gold"}</span></td>
      <td class="score">${formatScore(item.metrics?.primary)}</td>
      <td class="score">${formatScore(item.secondary)}</td>
      <td><button class="delete-button" title="Xóa submission" data-delete="${escapeHtml(item.id)}">×</button></td>
    </tr>`).join("");
  body.querySelectorAll("[data-delete]").forEach((button) => {
    button.addEventListener("click", () => deleteSubmission(button.dataset.delete));
  });
}

async function loadSubmissions() {
  const response = await fetch(`/api/submissions?task=${encodeURIComponent(state.task)}&stage=${encodeURIComponent(state.stage)}`);
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || "Không tải được bảng điểm");
  renderRows(payload.submissions);
}

async function updateView() {
  document.querySelectorAll(".task-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.task === state.task);
  });
  renderTask();
  try {
    await loadSubmissions();
  } catch (error) {
    showFlash(error.message, "error");
  }
}

async function deleteSubmission(id) {
  if (!window.confirm("Xóa submission này?")) return;
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
  form.append("label", file.name);
  form.append("file", file, file.name);
  try {
    const response = await fetch("/api/submissions", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Tải lên thất bại");
    const result = payload.submission;
    showFlash(result.status === "scored" ? `Đã chấm ${taskLabel()}: ${formatScore(result.primary)}` : "Đã lưu; stage này chưa có gold.");
    $("#uploadForm").reset();
    $("#fileName").textContent = "Chọn file submission";
    await updateView();
  } catch (error) {
    showFlash(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Tải lên và chấm";
  }
}

async function boot() {
  try {
    const response = await fetch("/api/config");
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Không tải được cấu hình");
    state.config = payload;
    state.stage = payload.default_stage || "warmup";
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
    updateView();
  });
});
$("#uploadForm").addEventListener("submit", uploadSubmission);
$("#submissionFile").addEventListener("change", (event) => {
  $("#fileName").textContent = event.target.files[0]?.name || "Chọn file submission";
});
$("#refreshButton").addEventListener("click", updateView);
boot();
