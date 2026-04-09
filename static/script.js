const form = document.querySelector("[data-detector-form]");
const tabs = document.querySelectorAll("[data-mode]");
const panels = document.querySelectorAll("[data-panel]");
const urlInput = document.querySelector("#url");
const textInput = document.querySelector("#text");
const counter = document.querySelector("[data-counter]");
const message = document.querySelector("[data-form-message]");
const submitButton = document.querySelector("[data-submit-button]");
const buttonLabel = document.querySelector("[data-button-label]");

function setMode(mode) {
  tabs.forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });

  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === mode);
  });

  if (mode === "url") {
    textInput.value = "";
    urlInput.focus();
  } else {
    urlInput.value = "";
    textInput.focus();
  }

  updateState();
}

function updateState() {
  const textLength = textInput.value.trim().length;
  const hasUrl = urlInput.value.trim().length > 0;
  const hasText = textLength > 0;

  counter.textContent = `${textLength} character${textLength === 1 ? "" : "s"}`;

  if (hasUrl && hasText) {
    message.textContent = "Use either a link or article text, not both.";
    submitButton.disabled = true;
    return;
  }

  if (hasText && textLength < 120) {
    message.textContent = "Add more article text for a stronger prediction.";
    submitButton.disabled = true;
    return;
  }

  if (!hasUrl && !hasText) {
    message.textContent = "Paste a link or switch to article text.";
    submitButton.disabled = true;
    return;
  }

  message.textContent = hasUrl ? "Ready to extract article text." : "Ready to predict from pasted text.";
  submitButton.disabled = false;
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

urlInput.addEventListener("input", updateState);
textInput.addEventListener("input", updateState);

form.addEventListener("submit", () => {
  submitButton.classList.add("loading");
  submitButton.disabled = true;
  buttonLabel.textContent = "Checking...";
});

if (textInput.value.trim()) {
  setMode("text");
} else {
  setMode("url");
}
