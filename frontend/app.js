const API_URL = "/api/ask";
const chatBox = document.getElementById("chatBox");
const form = document.getElementById("askForm");
const input = document.getElementById("questionInput");
const btn = document.getElementById("submitBtn");

function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  btn.disabled = true;

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const json = await res.json();

    if (json.status === "ok" && json.data) {
      addMessage(json.data.answer, "agent");
    } else {
      addMessage(json.message || "Something went wrong.", "error");
    }
  } catch (err) {
    addMessage("Failed to connect to the server.", "error");
  } finally {
    btn.disabled = false;
    input.focus();
  }
});
