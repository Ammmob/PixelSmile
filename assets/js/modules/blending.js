import { assetManifest } from "../data/asset-manifest.js";

function parseDirectoryListing(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return [...doc.querySelectorAll("a")]
    .map((a) => a.getAttribute("href") || "")
    .filter(Boolean);
}

function cleanEntry(entry) {
  return entry.replace(/\/+$/, "");
}

function toPath(base, entry) {
  if (/^https?:\/\//i.test(entry) || entry.startsWith("/")) return entry;
  return `${base}${entry}`;
}

function isImageFile(name) {
  return /\.(png|jpg|jpeg|webp|gif)$/i.test(name);
}

function stemFromPath(path) {
  const last = path.split("/").pop() || "";
  return last.replace(/\.[^.]+$/, "");
}

function labelize(expr) {
  if (!expr) return "";
  return expr
    .split(/[_-]+/)
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
}

function renderMessage(root, html) {
  root.innerHTML = `<div class="blend-empty">${html}</div>`;
}

function createCard() {
  const card = document.createElement("div");
  card.className = "blend-card";

  const t = document.createElement("div");
  t.className = "blend-title";

  const imgBox = document.createElement("div");
  imgBox.className = "blend-img-box";

  const img = new Image();
  imgBox.append(img);

  card.append(t, imgBox);
  return { card, titleEl: t, imgBox, imgEl: img };
}

function updateCard(cardRef, title, path) {
  if (!cardRef) return;
  cardRef.titleEl.textContent = title;
  cardRef.imgEl.alt = title;
  cardRef.imgEl.onerror = () => {
    cardRef.imgBox.innerHTML = `<div class="blend-empty">Failed to load<br /><code>${path}</code></div>`;
  };
  cardRef.imgEl.onload = () => {
    if (!cardRef.imgBox.contains(cardRef.imgEl)) {
      cardRef.imgBox.innerHTML = "";
      cardRef.imgBox.append(cardRef.imgEl);
    }
  };
  if (!cardRef.imgBox.contains(cardRef.imgEl)) {
    cardRef.imgBox.innerHTML = "";
    cardRef.imgBox.append(cardRef.imgEl);
  }
  cardRef.imgEl.src = path;
}

function loadListing(url) {
  const staticIds = assetManifest.blending.ids;
  const staticFilesById = assetManifest.blending.filesById;
  const idMatch = url.match(/\/blending\/([^/]+)\/?$/);

  if (idMatch) {
    return Promise.resolve(staticFilesById[idMatch[1]] || []);
  }

  if (/\/blending\/?$/.test(url)) {
    return Promise.resolve(staticIds);
  }

  return fetch(url)
    .then((res) => (res.ok ? res.text() : ""))
    .then((html) => (html ? parseDirectoryListing(html) : []))
    .catch(() => []);
}

function parseAssets(files) {
  const stems = files.map((p) => stemFromPath(p));
  const singles = new Set(stems.filter((s) => !s.includes("_")));

  const pairs = [];
  const pairMap = new Map();
  for (const s of stems) {
    if (!s.includes("_")) continue;
    const parts = s.split("_");
    if (parts.length !== 2) continue;
    const [a, b] = parts;
    if (!singles.has(a) || !singles.has(b)) continue;

    const aPath = files.find((p) => stemFromPath(p) === a) || "";
    const bPath = files.find((p) => stemFromPath(p) === b) || "";
    const blendPath = files.find((p) => stemFromPath(p) === s) || "";
    if (!aPath || !bPath || !blendPath) continue;

    const key = `${a}__${b}`;
    const item = { a, b, aPath, bPath, blendPath };
    pairs.push(item);
    pairMap.set(key, item);
  }

  const neighbors = new Map();
  function addNeighbor(a, b) {
    if (!neighbors.has(a)) neighbors.set(a, new Set());
    neighbors.get(a).add(b);
  }
  pairs.forEach((p) => {
    addNeighbor(p.a, p.b);
    addNeighbor(p.b, p.a);
  });

  const expressions = [...neighbors.keys()].sort((x, y) =>
    x.localeCompare(y, undefined, { sensitivity: "base" })
  );

  return { pairs, pairMap, neighbors, expressions };
}

export function createBlendingModule(root) {
  const base = "./assets/img/blending/";
  const wrapSelect = (selectEl) => {
    const shell = document.createElement("div");
    shell.className = "blend-select-wrap";
    shell.append(selectEl);
    return shell;
  };

  const controls = document.createElement("div");
  controls.className = "blend-controls";

  const idLabel = document.createElement("label");
  idLabel.textContent = "Subject ID:";
  idLabel.className = "blend-label";
  const idSelect = document.createElement("select");
  idSelect.className = "blend-select";

  const idGroup = document.createElement("div");
  idGroup.className = "blend-id-group is-hidden";
  idGroup.append(idLabel, wrapSelect(idSelect));

  const aLabel = document.createElement("label");
  aLabel.textContent = "Expression A:";
  aLabel.className = "blend-label";
  const aSelect = document.createElement("select");
  aSelect.className = "blend-select";
  const aField = document.createElement("div");
  aField.className = "blend-field blend-field-a";
  aField.append(aLabel, wrapSelect(aSelect));

  const bLabel = document.createElement("label");
  bLabel.textContent = "Expression B:";
  bLabel.className = "blend-label";
  const bSelect = document.createElement("select");
  bSelect.className = "blend-select";
  const bField = document.createElement("div");
  bField.className = "blend-field blend-field-b";
  bField.append(bLabel, wrapSelect(bSelect));

  const emptyField = document.createElement("div");
  emptyField.className = "blend-field blend-field-empty";

  const stage = document.createElement("div");
  stage.className = "blend-stage";

  controls.append(idGroup, aField, bField, emptyField);
  root.append(controls, stage);

  let currentAssets = null;
  let row = null;
  let cardA = null;
  let cardB = null;
  let cardBlend = null;

  function ensureTripletRow() {
    if (row) return;
    row = document.createElement("div");
    row.className = "blend-row";
    cardA = createCard();
    cardB = createCard();
    cardBlend = createCard();
    const plus = document.createElement("div");
    plus.className = "blend-op";
    plus.textContent = "+";
    const equals = document.createElement("div");
    equals.className = "blend-op";
    equals.textContent = "=";
    row.append(cardA.card, plus, cardB.card, equals, cardBlend.card);
    stage.innerHTML = "";
    stage.append(row);
  }

  function renderTriplet(item) {
    ensureTripletRow();
    updateCard(cardA, labelize(item.a), item.aPath);
    updateCard(cardB, labelize(item.b), item.bPath);
    updateCard(cardBlend, "Blended", item.blendPath);
  }

  function orientPair(pair, a, b) {
    if (!pair) return null;
    if (pair.a === a && pair.b === b) return pair;
    if (pair.a === b && pair.b === a) {
      return {
        a,
        b,
        aPath: pair.bPath,
        bPath: pair.aPath,
        blendPath: pair.blendPath
      };
    }
    return pair;
  }

  function renderSelectedPair(a, b) {
    const raw = pickPair(a, b);
    const pair = orientPair(raw, a, b);
    if (!pair) {
      renderMessage(stage, "No valid blend pair for the selected expressions.");
      return;
    }
    renderTriplet(pair);
  }

  function pickPair(a, b) {
    if (!currentAssets) return null;
    return (
      currentAssets.pairMap.get(`${a}__${b}`) ||
      currentAssets.pairMap.get(`${b}__${a}`) ||
      null
    );
  }

  function updateBOptionsAndRender() {
    if (!currentAssets) return;
    const a = aSelect.value;
    const bs = [...(currentAssets.neighbors.get(a) || [])].sort((x, y) =>
      x.localeCompare(y, undefined, { sensitivity: "base" })
    );
    bSelect.innerHTML = "";
    bs.forEach((b) => {
      const opt = document.createElement("option");
      opt.value = b;
      opt.textContent = labelize(b);
      bSelect.append(opt);
    });

    const selectedB = bSelect.value;
    renderSelectedPair(a, selectedB);
  }

  function loadIdAssets(id) {
    const dir = `${base}${id}/`;
    loadListing(dir).then((entries) => {
      const files = entries
        .map(cleanEntry)
        .filter((name) => isImageFile(name))
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }))
        .map((name) => toPath(dir, name));

      if (!files.length) {
        currentAssets = null;
        aSelect.innerHTML = "";
        bSelect.innerHTML = "";
        renderMessage(stage, `No images found in <code>${dir}</code>`);
        return;
      }

      const parsed = parseAssets(files);
      if (!parsed.pairs.length) {
        currentAssets = null;
        aSelect.innerHTML = "";
        bSelect.innerHTML = "";
        renderMessage(
          stage,
          `No valid blend pairs in <code>${dir}</code>.<br />Need: <code>A.*</code>, <code>B.*</code>, <code>A_B.*</code>`
        );
        return;
      }

      currentAssets = parsed;
      aSelect.innerHTML = "";
      parsed.expressions.forEach((a) => {
        const opt = document.createElement("option");
        opt.value = a;
        opt.textContent = labelize(a);
        aSelect.append(opt);
      });
      updateBOptionsAndRender();
    });
  }

  aSelect.addEventListener("change", updateBOptionsAndRender);
  bSelect.addEventListener("change", () => {
    renderSelectedPair(aSelect.value, bSelect.value);
  });

  loadListing(base).then((entries) => {
    const ids = entries
      .map(cleanEntry)
      .filter((name) => name && !name.startsWith(".") && !name.startsWith(".."));

    if (!ids.length) {
      renderMessage(
        stage,
        `No blending IDs found.<br />Put data under <code>assets/img/blending/&lt;id&gt;/</code>`
      );
      return;
    }

    idSelect.innerHTML = "";
    ids.forEach((id) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      idSelect.append(opt);
    });

    idSelect.addEventListener("change", () => {
      loadIdAssets(idSelect.value);
    });

    loadIdAssets(ids[0]);
  });
}
