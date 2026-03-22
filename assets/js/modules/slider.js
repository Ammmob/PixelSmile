import { assetManifest } from "../data/asset-manifest.js";

export function createSliderModule(root, config) {
  const expressionSets = config.groups.flatMap((group) =>
    chunkExpressions(group.names, 3).map((names, i) => ({
      id: `${group.id}-${i + 1}`,
      label: `${group.label.split(" ")[0]} ${i + 1 === 1 ? "I" : "II"}`,
      names
    }))
  );

  let activeSetIndex = 0;
  let requestId = 0;
  let activeEntries = [];
  let intensities = [];
  let editedBoxes = [];
  const assetCache = new Map();
  const listCache = new Map();

  const wrap = document.createElement("div");
  wrap.className = "editor-layout";

  const setTabs = document.createElement("div");
  setTabs.className = "expr-set-tabs";

  const labelRow = document.createElement("div");
  labelRow.className = "expr-triplet-labels";

  const originalRow = document.createElement("div");
  originalRow.className = "expr-triplet-row";

  const editedRow = document.createElement("div");
  editedRow.className = "expr-triplet-row";

  const sliderRow = document.createElement("div");
  sliderRow.className = "expr-triplet-sliders";

  wrap.append(setTabs, labelRow, originalRow, editedRow, sliderRow);
  root.append(wrap);

  function chunkExpressions(arr, size) {
    const out = [];
    for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
  }

  function formatIntensity(v) {
    return Number(v).toFixed(2);
  }

  function expressionDir(expression) {
    return `./assets/img/slider/${expression.toLowerCase()}/`;
  }

  function normalizeHref(href, dir) {
    if (!href) return "";
    if (/^https?:\/\//i.test(href)) return "";
    if (href.startsWith("/")) return `.${href}`;
    return `${dir}${href.replace(/^\.\//, "")}`;
  }

  function fileNumber(path) {
    const m = path.match(/(?:^|\/)(\d{2})\.(avif|png|jpg|jpeg|webp)$/i);
    return m ? Number(m[1]) : Number.POSITIVE_INFINITY;
  }

  function splitOriginalAndEdited(files) {
    const sorted = [...files].sort((a, b) => fileNumber(a) - fileNumber(b) || a.localeCompare(b));
    return {
      original: sorted.find((p) => fileNumber(p) === 0) || "",
      edited: sorted.filter((p) => fileNumber(p) >= 1 && Number.isFinite(fileNumber(p)))
    };
  }

  function loadListing(url) {
    if (listCache.has(url)) return Promise.resolve(listCache.get(url));
    return fetch(url)
      .then((res) => (res.ok ? res.text() : ""))
      .then((html) => {
        if (!html) return [];
        const files = [...new DOMParser().parseFromString(html, "text/html").querySelectorAll("a")]
          .map((a) => a.getAttribute("href") || "")
          .filter((href) => /\.(avif|png|jpg|jpeg|webp)$/i.test(href))
          .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }))
          .map((href) => normalizeHref(href, url));
        listCache.set(url, files);
        return files;
      })
      .catch(() => {
        listCache.set(url, []);
        return [];
      });
  }

  function loadExpressionAssets(expression) {
    const key = expression.toLowerCase();
    if (assetCache.has(key)) return Promise.resolve(assetCache.get(key));
    const dir = expressionDir(expression);
    const staticFiles = assetManifest.slider[key];
    const filesPromise = Array.isArray(staticFiles)
      ? Promise.resolve(staticFiles.map((name) => `${dir}${name}`))
      : loadListing(dir);
    return filesPromise.then((files) => {
      const split = splitOriginalAndEdited(files);
      const entry = { expression, ...split, dir };
      assetCache.set(key, entry);
      return entry;
    });
  }

  function setBoxContent(box, imagePath, emptyText) {
    if (!imagePath) {
      box.innerHTML = `<div>${emptyText}</div>`;
      return;
    }
    const img = new Image();
    img.src = imagePath;
    img.alt = emptyText;
    img.onload = () => {
      box.innerHTML = "";
      box.append(img);
    };
    img.onerror = () => {
      box.innerHTML = `<div>${emptyText}<br /><code>${imagePath}</code></div>`;
    };
  }

  function editedPath(entry, colIdx) {
    if (!entry.edited.length) return "";
    const intensity = intensities[colIdx] ?? 0;
    const idx = Math.round(intensity * (entry.edited.length - 1));
    return entry.edited[Math.max(0, Math.min(entry.edited.length - 1, idx))];
  }

  function createColumnSlider(colIdx) {
    const card = document.createElement("div");
    card.className = "slider-card";

    const top = document.createElement("div");
    top.className = "slider-top";
    const label = document.createElement("strong");
    label.textContent = "Intensity";
    const value = document.createElement("span");
    value.className = "slider-value";
    value.textContent = formatIntensity(intensities[colIdx] ?? 0);
    top.append(label, value);

    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "100";
    input.step = "1";
    input.value = String(Math.round((intensities[colIdx] ?? 0) * 100));
    input.addEventListener("input", () => {
      intensities[colIdx] = Number(input.value) / 100;
      value.textContent = formatIntensity(intensities[colIdx]);
      updateEditedCol(colIdx);
    });

    card.append(top, input);
    return card;
  }

  function updateEditedCol(colIdx) {
    const entry = activeEntries[colIdx];
    const box = editedBoxes[colIdx];
    if (!entry || !box) return;
    setBoxContent(
      box,
      editedPath(entry, colIdx),
      `<strong>${entry.expression}</strong><br />Missing 01+ images<br /><code>${entry.dir}</code>`
    );
  }

  function renderRows() {
    labelRow.innerHTML = "";
    originalRow.innerHTML = "";
    editedRow.innerHTML = "";
    sliderRow.innerHTML = "";
    editedBoxes = [];

    activeEntries.forEach((entry, i) => {
      const label = document.createElement("div");
      label.className = "expr-label";
      label.textContent = entry.expression;
      labelRow.append(label);

      const originalBox = document.createElement("div");
      originalBox.className = "preview-box";
      setBoxContent(
        originalBox,
        entry.original,
        `<strong>${entry.expression}</strong><br />Missing 00.*<br /><code>${entry.dir}</code>`
      );
      originalRow.append(originalBox);

      const editedBox = document.createElement("div");
      editedBox.className = "preview-box";
      setBoxContent(
        editedBox,
        editedPath(entry, i),
        `<strong>${entry.expression}</strong><br />Missing 01+ images<br /><code>${entry.dir}</code>`
      );
      editedBoxes.push(editedBox);
      editedRow.append(editedBox);

      sliderRow.append(createColumnSlider(i));
    });
  }

  function renderSetTabs() {
    setTabs.innerHTML = "";
    expressionSets.forEach((set, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `set-tab${idx === activeSetIndex ? " is-active" : ""}`;
      btn.textContent = set.label;
      btn.addEventListener("click", () => {
        activeSetIndex = idx;
        renderSetTabs();
        renderSetContent();
      });
      setTabs.append(btn);
    });
  }

  function renderSetContent() {
    const rid = ++requestId;
    const names = expressionSets[activeSetIndex].names;
    Promise.all(names.map((name) => loadExpressionAssets(name))).then((entries) => {
      if (rid !== requestId) return;
      activeEntries = entries;
      intensities = entries.map(() => 0);
      renderRows();
    });
  }

  renderSetTabs();
  renderSetContent();
}
