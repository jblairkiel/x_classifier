// Labeling page keyboard shortcuts: h / a / u / s.
// Calls htmx directly (instead of clicking the button) so a key pressed right after a card swap
// can't race htmx's processing of the new form and fall back to a full-page submit.
let labelInFlight = false;

document.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey || ev.repeat) return;
  const tag = (ev.target && ev.target.tagName) || "";
  if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
  const form = document.getElementById("label-form");
  if (!form) return;
  const btn = form.querySelector(`button[data-key="${ev.key.toLowerCase()}"]`);
  if (!btn || btn.disabled) return;
  ev.preventDefault();
  if (labelInFlight) return;
  if (window.htmx) {
    labelInFlight = true;
    window.htmx
      .ajax("POST", form.getAttribute("action"), {
        source: form,
        values: { label: btn.value },
        target: "#label-card",
        swap: "innerHTML",
      })
      .finally(() => (labelInFlight = false));
  } else {
    btn.click();
  }
});

// Prevent double submits while an htmx request is in flight; re-enable if it fails.
document.addEventListener("htmx:beforeRequest", (ev) => {
  const elt = ev.detail.elt;
  if (elt && elt.querySelectorAll) elt.querySelectorAll("button").forEach((b) => (b.disabled = true));
});
document.addEventListener("htmx:afterRequest", (ev) => {
  const elt = ev.detail.elt;
  if (!ev.detail.successful && elt && elt.querySelectorAll) {
    elt.querySelectorAll("button").forEach((b) => (b.disabled = false));
  }
});
