/* ---------------------------------------------------------------------------
   Financial QA Agent — Frontend
   Chat + Trace panel with SSE streaming
--------------------------------------------------------------------------- */

const STREAM_URL = "/api/ask/stream";

// DOM references — chat
const chatBox = document.getElementById("chatBox");
const form = document.getElementById("askForm");
const input = document.getElementById("questionInput");
const btn = document.getElementById("submitBtn");

// DOM references — trace tabs
const traceTabs = document.querySelectorAll(".trace-tab");
const tabPanes = document.querySelectorAll(".tab-pane");

// Map SSE tool names → tab pane IDs
const TOOL_TAB_MAP = {
  market_data: "tab-market-data",
  fundamental_data: "tab-market-data",
  news_search: "tab-news-search",
  local_knowledge: "tab-knowledge",
  web_knowledge: "tab-knowledge",
};

// Sub-section labels for tabs with multiple tools
const TOOL_SECTION_LABEL = {
  market_data: "Price Data",
  fundamental_data: "Fundamental Data",
  local_knowledge: "Local Knowledge",
  web_knowledge: "Online Knowledge",
};

/* ---------------------------------------------------------------------------
   Chat helpers
--------------------------------------------------------------------------- */

/* Configure marked for safe rendering */
marked.setOptions({
  breaks: true, // Convert \n to <br> within paragraphs
  gfm: true, // GitHub-flavoured markdown (tables, strikethrough)
});

function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `msg ${type}`;

  if (type === "agent") {
    // Render markdown → HTML, then sanitize
    const rawHtml = marked.parse(text);
    div.innerHTML = DOMPurify.sanitize(rawHtml, { ADD_ATTR: ["target"] });
    // Open all links in new tab
    div.querySelectorAll("a").forEach((a) => {
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
    });
  } else {
    div.textContent = text;
  }

  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/* ---------------------------------------------------------------------------
   Tab switching
--------------------------------------------------------------------------- */

function activateTab(tabName) {
  traceTabs.forEach((t) => t.classList.remove("active"));
  tabPanes.forEach((p) => p.classList.remove("active"));
  const btn = document.querySelector(`.trace-tab[data-tab="${tabName}"]`);
  const pane = document.getElementById(`tab-${tabName}`);
  if (btn) btn.classList.add("active");
  if (pane) pane.classList.add("active");
}

traceTabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

/* ---------------------------------------------------------------------------
   Trace reset / render helpers
--------------------------------------------------------------------------- */

function resetTrace() {
  document.getElementById("tab-agent-loop").innerHTML = "";
  document.getElementById("tab-market-data").innerHTML = "";
  document.getElementById("tab-news-search").innerHTML = "";
  document.getElementById("tab-knowledge").innerHTML = "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}

function formatData(obj) {
  if (typeof obj === "string") return escapeHtml(obj);
  return escapeHtml(JSON.stringify(obj, null, 2));
}

/* Agent Loop tab — progressive timeline */
function appendAgentLoopEntry(data) {
  const container = document.getElementById("tab-agent-loop");
  const stage = data.stage || "unknown";
  const status = data.status || "started";
  const detail = data.detail || "";

  // If "completed" and a "started" entry exists, update it in place
  if (status === "completed") {
    const existing = document.getElementById(`stage-${stage}`);
    if (existing) {
      existing.className = "trace-entry completed";
      existing.querySelector(".stage-icon").textContent = "\u2713";
      existing.querySelector(".stage-detail").textContent = detail;
      return;
    }
  }

  const entry = document.createElement("div");
  entry.className = `trace-entry ${status}`;
  entry.id = `stage-${stage}`;
  const icon = status === "started" ? "\u23F3" : "\u2713";
  entry.innerHTML =
    `<span class="stage-icon">${icon}</span>` +
    `<span class="stage-name">${escapeHtml(stage)}</span>` +
    `<span class="stage-detail">${escapeHtml(detail)}</span>`;

  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
}

/* Tool tabs — input / output blocks */

/** Get or create a sub-section container for a tool inside its tab. */
function getToolContainer(tabId, toolName) {
  const label = TOOL_SECTION_LABEL[toolName];
  if (!label) return document.getElementById(tabId);

  // Check if a sub-section already exists for this tool
  const existingId = `subsection-${toolName}`;
  let subsection = document.getElementById(existingId);
  if (subsection) return subsection;

  // Create sub-section with header
  const tab = document.getElementById(tabId);
  subsection = document.createElement("div");
  subsection.id = existingId;
  subsection.className = "knowledge-subsection";
  subsection.innerHTML = `<div class="subsection-header">${escapeHtml(label)}</div>`;
  tab.appendChild(subsection);
  return subsection;
}

function appendToolInput(data) {
  const tabId = TOOL_TAB_MAP[data.tool];
  if (!tabId) return;
  const container = getToolContainer(tabId, data.tool);

  const section = document.createElement("div");
  section.className = "tool-section";
  section.innerHTML =
    `<div class="tool-section-label">Input</div>` +
    `<div class="tool-data">${formatData(data.input)}</div>`;
  container.appendChild(section);
}

function appendToolOutput(data) {
  const tabId = TOOL_TAB_MAP[data.tool];
  if (!tabId) return;
  const container = getToolContainer(tabId, data.tool);

  const section = document.createElement("div");
  section.className = "tool-section";
  section.innerHTML =
    `<div class="tool-section-label">Output</div>` +
    `<div class="tool-data">${formatData(data.output)}</div>`;
  container.appendChild(section);
}

/* ---------------------------------------------------------------------------
   SSE stream consumption
--------------------------------------------------------------------------- */

/**
 * Parse an SSE text buffer into discrete events.
 * Returns { parsed: [{type, data}], remainder: string }.
 */
function parseSSEBuffer(buffer) {
  const parsed = [];
  const blocks = buffer.split("\n\n");
  const remainder = blocks.pop(); // may be incomplete

  for (const block of blocks) {
    if (!block.trim()) continue;
    let eventType = "trace";
    let dataStr = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) eventType = line.slice(7);
      else if (line.startsWith("data: ")) dataStr = line.slice(6);
    }
    if (dataStr) {
      try {
        parsed.push({ type: eventType, data: JSON.parse(dataStr) });
      } catch (_) {
        /* skip malformed */
      }
    }
  }
  return { parsed, remainder };
}

/** Route a single SSE event to the right handler. */
function handleTraceEvent(event) {
  switch (event.type) {
    case "trace":
      appendAgentLoopEntry(event.data);
      break;
    case "tool_input":
      appendToolInput(event.data);
      break;
    case "tool_output":
      appendToolOutput(event.data);
      break;
    case "answer":
      addMessage(event.data.answer, "agent");
      break;
    case "error":
      addMessage(event.data.message || "Something went wrong.", "error");
      break;
    case "done":
      // stream finished — handled by caller
      break;
  }
}

/** Stream a question via SSE and progressively update trace + chat. */
async function streamQuestion(question) {
  resetTrace();
  activateTab("agent-loop");

  const response = await fetch(STREAM_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    addMessage(`Server error: ${response.status}`, "error");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const result = parseSSEBuffer(buffer);
    buffer = result.remainder;

    for (const event of result.parsed) {
      handleTraceEvent(event);
    }
  }
}

/* ---------------------------------------------------------------------------
   Form submit
--------------------------------------------------------------------------- */

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  btn.disabled = true;

  try {
    await streamQuestion(question);
  } catch (err) {
    addMessage("Failed to connect to the server.", "error");
  } finally {
    btn.disabled = false;
    input.focus();
  }
});
