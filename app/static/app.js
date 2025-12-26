const textEl = document.getElementById("text");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");

const resultEl = document.getElementById("result");
const predEl = document.getElementById("pred");
const confEl = document.getElementById("conf");
const barsEl = document.getElementById("bars");
const errorEl = document.getElementById("error");

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
  resultEl.classList.add("hidden");
}

function clearError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}

function renderBars(ranked) {
  barsEl.innerHTML = "";
  ranked.forEach(item => {
    const pct = (item.prob * 100).toFixed(1);

    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `
      <div>${item.label}</div>
      <div class="track"><div class="fill" style="width:${pct}%"></div></div>
      <div>${pct}%</div>
    `;
    barsEl.appendChild(row);
  });
}

predictBtn.addEventListener("click", async () => {
  clearError();
  const text = textEl.value.trim();
  if (!text) return showError("Write something first.");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });

    const data = await res.json();
    if (data.error) return showError(data.error);

    predEl.textContent = data.prediction;
    confEl.textContent = (data.confidence * 100).toFixed(1) + "%";
    renderBars(data.ranked);

    resultEl.classList.remove("hidden");
  } catch (e) {
    showError("Server error. Is the backend running?");
  }
});

clearBtn.addEventListener("click", () => {
  textEl.value = "";
  resultEl.classList.add("hidden");
  clearError();
});