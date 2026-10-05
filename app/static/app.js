const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const conversation = document.querySelector("#conversation");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const clear = document.querySelector("#clear");
const statusLine = document.querySelector("#status");
const characterCount = document.querySelector("#character-count");
const guide = document.querySelector("#guide");
let history = [];
let activeRequest = null;
let lastFailure = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function icon(name, className = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  node.setAttribute("class", `icon ${className}`.trim());
  node.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#icon-${name}`);
  node.append(use);
  return node;
}

// Render basic answer formatting as DOM text, never as model-provided HTML.
function appendInline(parent, text) {
  text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      parent.append(element("strong", "", part.slice(2, -2)));
    } else if (part.startsWith("`") && part.endsWith("`")) {
      parent.append(element("code", "", part.slice(1, -1)));
    } else {
      parent.append(document.createTextNode(part));
    }
  });
}

function renderAnswer(text) {
  const content = element("div", "answer-content");
  let block = null;
  let list = null;
  text.split(/\r?\n/).forEach((line) => {
    if (!line.trim()) { block = null; list = null; return; }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    const item = line.match(/^\s*(?:[-*]\s+|\d+[.)]\s+)(.+)$/);
    if (heading) {
      const title = element("h3");
      appendInline(title, heading[1]);
      content.append(title);
      block = null;
      list = null;
    } else if (item) {
      const tag = /^\s*\d/.test(line) ? "ol" : "ul";
      if (!list || list.localName !== tag) {
        list = element(tag);
        if (tag === "ol") list.start = Number(line.match(/^\s*(\d+)/)[1]);
        content.append(list);
      }
      const li = element("li");
      appendInline(li, item[1]);
      list.append(li);
      block = null;
    } else {
      list = null;
      if (!block) { block = element("p"); content.append(block); }
      else block.append(document.createTextNode("\n"));
      appendInline(block, line);
    }
  });
  return content;
}

function renderSources(sources) {
  const details = element("details", "sources");
  const summary = element("summary");
  summary.append(icon("book"), element("span", "", `${sources.length} source${sources.length > 1 ? "s" : ""} consultée${sources.length > 1 ? "s" : ""}`), icon("chevron", "chevron"));
  details.append(summary);
  sources.forEach((source) => {
    const card = element("section", "source-card");
    const heading = element("div", "source-heading");
    heading.append(element("span", "source-number", `Source ${source.rank}`), element("h3", "source-title", source.title || "Extrait du corpus juridique"));
    card.append(heading);
    const metadata = [source.article_number && `Article ${source.article_number}`, source.category].filter(Boolean);
    if (metadata.length) card.append(element("p", "source-meta", [...new Set(metadata)].join(" · ")));
    if (source.excerpt) card.append(element("blockquote", "source-excerpt", source.excerpt));
    if (source.doc_id) card.append(element("p", "source-reference", `Référence du document : ${source.doc_id}`));
    details.append(card);
  });
  return details;
}

function showConversation() {
  welcome.hidden = true;
  messages.hidden = false;
}

function scrollToBottom() {
  conversation.scrollTop = conversation.scrollHeight;
}

function addMessage(role, text, response) {
  showConversation();
  const box = element("article", `message ${role}`);
  box.setAttribute("aria-label", role === "user" ? "Votre question" : "Réponse de l’assistant");
  if (role === "user") {
    box.textContent = text;
  } else {
    const header = element("div", "assistant-header");
    header.append(icon("book"), element("span", "", "Assistant juridique"));
    box.append(header, renderAnswer(text));
    if (response?.sources?.length) box.append(renderSources(response.sources));
    const copy = element("button", "copy-button");
    copy.type = "button";
    copy.setAttribute("aria-label", "Copier la réponse");
    const label = element("span", "", "Copier");
    copy.append(icon("copy"), label);
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        label.textContent = "Copié";
        statusLine.textContent = "Réponse copiée.";
        setTimeout(() => { label.textContent = "Copier"; }, 2000);
      } catch {
        statusLine.textContent = "La copie n’est pas disponible. Sélectionnez le texte pour le copier.";
      }
    });
    box.append(copy);
  }
  messages.append(box);
  scrollToBottom();
  return box;
}

function updateComposer() {
  input.style.height = "auto";
  input.style.height = `${Math.min(Math.max(input.scrollHeight, 48), 160)}px`;
  characterCount.hidden = input.value.length < 1600;
  characterCount.textContent = `${input.value.length.toLocaleString("fr-FR")} / 2 000`;
  send.disabled = !activeRequest && !input.value.trim();
}

function setBusy(value) {
  input.disabled = value;
  input.required = !value;
  send.classList.toggle("is-busy", value);
  send.setAttribute("aria-label", value ? "Arrêter l’attente" : "Envoyer la question");
  send.title = value ? "Arrêter l’attente" : "Envoyer la question";
  statusLine.classList.toggle("is-busy", value);
  statusLine.textContent = value ? "L’assistant prépare sa réponse…" : "";
  messages.querySelectorAll(".retry-button").forEach((button) => { button.disabled = value; });
  updateComposer();
}

function removeFailure() {
  lastFailure?.remove();
  lastFailure = null;
}

function showFailure(message, detail, cancelled = false) {
  const box = element("div", cancelled ? "message message-notice" : "message message-error");
  box.setAttribute("role", cancelled ? "status" : "alert");
  box.append(element("p", "", detail));
  const retry = element("button", "retry-button", "Réessayer");
  retry.type = "button";
  retry.addEventListener("click", () => submitQuestion(message, true));
  box.append(retry);
  messages.append(box);
  lastFailure = box;
  scrollToBottom();
}

async function submitQuestion(message, retry = false) {
  if (activeRequest || !message || message.length > 2000) return;
  removeFailure();
  if (!retry) addMessage("user", message);
  input.value = "";
  const request = new AbortController();
  activeRequest = request;
  setBusy(true);
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; request.abort(); }, 180000);
  try {
    const reply = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
      signal: request.signal,
    });
    const data = await reply.json().catch(() => null);
    if (!reply.ok) {
      const detail = typeof data?.detail === "string" ? data.detail : "La demande n’a pas abouti. Réessayez dans un instant.";
      throw new Error(detail);
    }
    if (typeof data?.answer !== "string" || !data.answer.trim()) throw new Error("La réponse est vide. Réessayez votre question.");
    if (activeRequest !== request) return;
    addMessage("assistant", data.answer, data);
    history = [...history, { role: "user", content: message }, { role: "assistant", content: data.answer.slice(0, 2000) }].slice(-40);
  } catch (error) {
    if (activeRequest !== request) return;
    if (request.signal.aborted) {
      showFailure(message, timedOut ? "Le délai de réponse est dépassé. Vous pouvez réessayer." : "L’attente a été arrêtée. Vous pouvez réessayer ou poser une autre question.", !timedOut);
    } else {
      showFailure(message, error instanceof TypeError ? "Impossible de joindre l’assistant local. Vérifiez que le service est démarré, puis réessayez." : error.message);
    }
  } finally {
    clearTimeout(timer);
    if (activeRequest === request) {
      activeRequest = null;
      setBusy(false);
      input.focus({ preventScroll: true });
    }
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (activeRequest) { activeRequest.abort(); return; }
  submitQuestion(input.value.trim());
});

input.addEventListener("input", updateComposer);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (input.value.trim() && !activeRequest) form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    updateComposer();
    input.focus();
  });
});

clear.addEventListener("click", () => {
  activeRequest?.abort();
  activeRequest = null;
  history = [];
  removeFailure();
  messages.replaceChildren();
  messages.hidden = true;
  welcome.hidden = false;
  input.value = "";
  setBusy(false);
  conversation.scrollTop = 0;
  input.focus({ preventScroll: true });
});

document.querySelector("#guide-open").addEventListener("click", () => guide.showModal());
document.querySelector("#guide-close").addEventListener("click", () => guide.close());
guide.addEventListener("click", (event) => {
  if (event.target !== guide) return;
  const bounds = guide.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) guide.close();
});

updateComposer();
