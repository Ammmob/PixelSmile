import { customStopwords, datasetStyle } from "../data/dataset.js";

const jsonCache = new Map();
const BAR_FONT = {
  tick: 13,
  value: 13,
  label: 13
};
const CLOUD_FONT_FAMILIES = {
  byCloud: {
    human_appearance_sentence: "\"Comic Sans MS\", \"Trebuchet MS\", sans-serif",
    human_action_sentence: "\"Comic Sans MS\", \"Trebuchet MS\", sans-serif",
    human_background_sentence: "\"Comic Sans MS\", \"Trebuchet MS\", sans-serif",
    anime_appearance_sentence: "\"Comic Sans MS\", \"Trebuchet MS\", sans-serif",
    anime_action_sentence: "\"Comic Sans MS\", \"Trebuchet MS\", sans-serif",
    anime_background_sentence: "\"Comic Sans MS\", \"Trebuchet MS\", sans-serif"
  },
  fallback: "\"Trebuchet MS\", \"Segoe UI\", sans-serif"
};

function fetchJson(path) {
  if (jsonCache.has(path)) return Promise.resolve(jsonCache.get(path));
  return fetch(path)
    .then((res) => (res.ok ? res.json() : null))
    .then((data) => {
      jsonCache.set(path, data);
      return data;
    })
    .catch(() => null);
}

function titleFromKey(key) {
  return key
    .split(/[_-]+/)
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
}

function cloudFontFamily(cloudKey) {
  return CLOUD_FONT_FAMILIES.byCloud[cloudKey] || CLOUD_FONT_FAMILIES.fallback;
}

function makeStopwordSet(field) {
  const global = customStopwords.global || [];
  const local = customStopwords[field] || [];
  return new Set([...global, ...local].map((w) => String(w).toLowerCase()));
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function hexToRgb(hex) {
  const cleaned = String(hex).replace("#", "").trim();
  const full = cleaned.length === 3
    ? cleaned.split("").map((ch) => ch + ch).join("")
    : cleaned;
  if (full.length !== 6) return null;
  return {
    r: Number.parseInt(full.slice(0, 2), 16),
    g: Number.parseInt(full.slice(2, 4), 16),
    b: Number.parseInt(full.slice(4, 6), 16)
  };
}

function rgbToHex({ r, g, b }) {
  return `#${clampByte(r).toString(16).padStart(2, "0")}${clampByte(g).toString(16).padStart(2, "0")}${clampByte(b).toString(16).padStart(2, "0")}`;
}

function mixHexColors(baseHex, targetHex, amount) {
  const base = hexToRgb(baseHex);
  const target = hexToRgb(targetHex);
  if (!base || !target) return baseHex;
  const t = Math.max(0, Math.min(1, amount));
  return rgbToHex({
    r: base.r + (target.r - base.r) * t,
    g: base.g + (target.g - base.g) * t,
    b: base.b + (target.b - base.b) * t
  });
}

function drawBarChart(container, rows, theme, title) {
  container.innerHTML = "";
  if (!rows.length) {
    container.innerHTML = `<div class="blend-empty">No data for ${title}</div>`;
    return;
  }

  const sorted = [...rows].sort((a, b) => b.value - a.value);
  const total = sorted.reduce((acc, r) => acc + r.value, 0) || 1;
  const rowsPct = sorted.map((r) => ({ ...r, pct: (r.value / total) * 100 }));
  const maxVal = rowsPct[0].pct || 1;
  const colors = datasetStyle.chartColors[theme] || datasetStyle.chartColors.human;
  const width = 560;
  const height = 320;
  const margin = { top: 18, right: 12, bottom: 72, left: 52 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const barW = innerW / sorted.length;

  const frame = document.createElement("div");
  frame.className = "dataset-viz-frame";
  const tooltip = document.createElement("div");
  tooltip.className = "dataset-tooltip";

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "dataset-bar-svg");

  for (let i = 0; i <= 4; i += 1) {
    const yVal = maxVal * (1 - i / 4);
    const y = margin.top + (innerH * i) / 4;
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(margin.left));
    line.setAttribute("x2", String(width - margin.right));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", colors.grid);
    line.setAttribute("stroke-width", "1");
    svg.append(line);

    const tick = document.createElementNS(svgNS, "text");
    tick.setAttribute("x", String(margin.left - 8));
    tick.setAttribute("y", String(y + 4));
    tick.setAttribute("text-anchor", "end");
    tick.setAttribute("font-size", String(BAR_FONT.tick));
    tick.setAttribute("fill", colors.axis);
    tick.textContent = `${yVal.toFixed(1)}%`;
    svg.append(tick);
  }

  const axisX = document.createElementNS(svgNS, "line");
  axisX.setAttribute("x1", String(margin.left));
  axisX.setAttribute("x2", String(width - margin.right));
  axisX.setAttribute("y1", String(margin.top + innerH));
  axisX.setAttribute("y2", String(margin.top + innerH));
  axisX.setAttribute("stroke", colors.axis);
  axisX.setAttribute("stroke-width", "1.2");
  svg.append(axisX);

  const axisY = document.createElementNS(svgNS, "line");
  axisY.setAttribute("x1", String(margin.left));
  axisY.setAttribute("x2", String(margin.left));
  axisY.setAttribute("y1", String(margin.top));
  axisY.setAttribute("y2", String(margin.top + innerH));
  axisY.setAttribute("stroke", colors.axis);
  axisY.setAttribute("stroke-width", "1.2");
  svg.append(axisY);

  rowsPct.forEach((row, idx) => {
    const h = Math.max(2, (row.pct / maxVal) * innerH);
    const x = margin.left + idx * barW + 6;
    const y = margin.top + innerH - h;
    const bw = Math.max(8, barW - 12);
    const progress = sorted.length <= 1 ? 0 : idx / (sorted.length - 1);
    const barTop = mixHexColors(colors.barTop || colors.bar, "#ffffff", 0.06 + 0.28 * progress);
    const barBottom = mixHexColors(colors.barBottom || colors.bar, "#ffffff", 0.03 + 0.38 * progress);
    const barHover = mixHexColors(colors.barHover || colors.bar, "#ffffff", 0.08 + 0.20 * progress);

    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(bw));
    rect.setAttribute("height", String(h));
    rect.setAttribute("rx", "5");
    const gradId = `bar-grad-${theme}-${idx}`;
    const defs = document.createElementNS(svgNS, "defs");
    const grad = document.createElementNS(svgNS, "linearGradient");
    grad.setAttribute("id", gradId);
    grad.setAttribute("x1", "0");
    grad.setAttribute("y1", "0");
    grad.setAttribute("x2", "0");
    grad.setAttribute("y2", "1");
    const stopTop = document.createElementNS(svgNS, "stop");
    stopTop.setAttribute("offset", "0%");
    stopTop.setAttribute("stop-color", barTop);
    const stopBottom = document.createElementNS(svgNS, "stop");
    stopBottom.setAttribute("offset", "100%");
    stopBottom.setAttribute("stop-color", barBottom);
    grad.append(stopTop, stopBottom);
    defs.append(grad);
    svg.append(defs);

    rect.setAttribute("fill", `url(#${gradId})`);
    rect.style.transition = "transform 140ms ease, fill 140ms ease";
    rect.style.transformOrigin = `${x + bw / 2}px ${margin.top + innerH}px`;

    rect.addEventListener("mousemove", (e) => {
      rect.setAttribute("fill", barHover);
      rect.style.transform = "scale(1.04)";
      tooltip.classList.add("is-visible");
      tooltip.textContent = `${titleFromKey(row.key)}: ${row.pct.toFixed(1)}%`;
      const box = frame.getBoundingClientRect();
      tooltip.style.left = `${e.clientX - box.left + 10}px`;
      tooltip.style.top = `${e.clientY - box.top - 26}px`;
    });
    rect.addEventListener("mouseleave", () => {
      rect.setAttribute("fill", `url(#${gradId})`);
      rect.style.transform = "scale(1)";
      tooltip.classList.remove("is-visible");
    });

    svg.append(rect);

    const valueText = document.createElementNS(svgNS, "text");
    valueText.setAttribute("x", String(x + bw / 2));
    valueText.setAttribute("y", String(y - 5));
    valueText.setAttribute("text-anchor", "middle");
    valueText.setAttribute("font-size", String(BAR_FONT.value));
    valueText.setAttribute("fill", colors.axis);
    valueText.textContent = `${row.pct.toFixed(1)}%`;
    svg.append(valueText);

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", String(x + bw / 2));
    label.setAttribute("y", String(margin.top + innerH + 14));
    label.setAttribute("text-anchor", "end");
    label.setAttribute("font-size", String(BAR_FONT.label));
    label.setAttribute("fill", colors.axis);
    label.setAttribute("transform", `rotate(-28 ${x + bw / 2} ${margin.top + innerH + 14})`);
    label.textContent = titleFromKey(row.key);
    svg.append(label);
  });

  frame.append(svg, tooltip);
  container.append(frame);
}

function rectangleIntersects(a, b) {
  return !(
    a.x + a.w < b.x ||
    b.x + b.w < a.x ||
    a.y + a.h < b.y ||
    b.y + b.h < a.y
  );
}

function candidateRect(mode, attempt, width, height, rectW, rectH, step) {
  const cx = width / 2;
  const cy = height / 2;

  if (mode === "wave_fill") {
    const laneCount = 10;
    const lane = attempt % laneCount;
    const laneW = width / laneCount;
    const x = lane * laneW + ((attempt * 3.7) % Math.max(8, laneW - rectW - 4));
    const phase = attempt * 0.23 + lane * 0.8;
    const yBase = ((attempt * 11.4) % Math.max(12, height - rectH - 8));
    const y = yBase + Math.sin(phase) * 12;
    return { x, y, w: rectW, h: rectH };
  }

  if (mode === "vertical_fill") {
    const cols = 8;
    const c = attempt % cols;
    const colW = width / cols;
    const x = c * colW + ((attempt * 1.9) % Math.max(6, colW - rectW - 3));
    const y = ((Math.floor(attempt / cols) * 10.8) + (c % 2) * 8) % Math.max(10, height - rectH - 8);
    return { x, y, w: rectW, h: rectH };
  }

  if (mode === "horizontal_fill") {
    const rows = 6;
    const r = attempt % rows;
    const rowH = height / rows;
    const y = r * rowH + ((attempt * 1.8) % Math.max(6, rowH - rectH - 3));
    const x = ((Math.floor(attempt / rows) * 13.2) + (r % 2) * 11) % Math.max(10, width - rectW - 8);
    return { x, y, w: rectW, h: rectH };
  }

  const theta = attempt * 0.35;
  const radius = attempt * step;
  const x = cx + radius * Math.cos(theta) - rectW / 2;
  const y = cy + radius * Math.sin(theta) - rectH / 2;
  return { x, y, w: rectW, h: rectH };
}

function drawWordCloud(container, words, field, theme, cloudKey) {
  container.innerHTML = "";
  if (!words.length) {
    container.innerHTML = `<div class="blend-empty">No words for ${titleFromKey(field)}</div>`;
    return;
  }

  const stopSet = makeStopwordSet(field);
  const filtered = words
    .filter((w) => !stopSet.has(String(w.word || "").toLowerCase()))
    .sort((a, b) => b.count - a.count);

  const topK = Number(datasetStyle.topKWords) || 0;
  const limited = topK > 0 ? filtered.slice(0, topK) : filtered;

  if (!limited.length) {
    container.innerHTML = `<div class="blend-empty">No words after stopword filtering.</div>`;
    return;
  }

  const frame = document.createElement("div");
  frame.className = "dataset-viz-frame";
  const tooltip = document.createElement("div");
  tooltip.className = "dataset-tooltip";

  const canvas = document.createElement("canvas");
  canvas.className = "dataset-cloud-canvas";
  const width = 560;
  const height = 320;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    container.innerHTML = `<div class="blend-empty">Canvas is not supported.</div>`;
    return;
  }

  const palette = datasetStyle.wordColors[theme] || datasetStyle.wordColors.human;
  const maxCount = limited[0].count || 1;
  const minCount = limited[limited.length - 1].count || 1;
  const placed = [];
  const activeLayout = datasetStyle.cloudLayouts?.flow || {
    fontMin: 9,
    fontMax: 28,
    rotateRatio: 0.05,
    spiralStep: 0.24,
    radiusLimit: 290
  };
  const cloudFamily = cloudFontFamily(cloudKey);

  limited.forEach((item, idx) => {
    const t = maxCount === minCount ? 0.5 : (item.count - minCount) / (maxCount - minCount);
    let fontSize = Math.round(activeLayout.fontMin + t * (activeLayout.fontMax - activeLayout.fontMin));
    const text = String(item.word);
    const angle = Math.random() < activeLayout.rotateRatio ? -Math.PI / 2 : 0;
    const fontFamily = cloudFamily;
    let placedRect = null;

    while (!placedRect && fontSize >= activeLayout.fontMin - 1) {
      ctx.save();
      ctx.font = `${fontSize}px ${fontFamily}`;
      const textW = ctx.measureText(text).width;
      const rectW = angle === 0 ? textW : fontSize;
      const rectH = angle === 0 ? fontSize : textW;
      ctx.restore();

      for (let attempt = 0; attempt < 2400; attempt += 1) {
        const candidate = candidateRect(
          activeLayout.mode || "spiral",
          attempt,
          width,
          height,
          rectW,
          rectH,
          activeLayout.spiralStep
        );

        const inside =
          candidate.x >= 2 &&
          candidate.y >= 2 &&
          candidate.x + candidate.w <= width - 2 &&
          candidate.y + candidate.h <= height - 2;
        const collides = placed.some((p) => rectangleIntersects(candidate, p.rect));
        if (inside && !collides) {
          placedRect = candidate;
          break;
        }
      }
      if (!placedRect) fontSize -= 1;
    }

    if (!placedRect) return;

    const color = palette[idx % palette.length];
    placed.push({
      word: text,
      count: item.count,
      color,
      fontSize,
      fontFamily,
      angle,
      rect: placedRect,
      x: placedRect.x + placedRect.w / 2,
      y: placedRect.y + placedRect.h / 2
    });
  });

  function repaint(highlightIndex = -1) {
    ctx.clearRect(0, 0, width, height);
    placed.forEach((p, idx) => {
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.angle);
      ctx.font = `${p.fontSize}px ${p.fontFamily}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = idx === highlightIndex ? "#1f2937" : p.color;
      ctx.fillText(p.word, 0, 0);
      ctx.restore();
    });
  }

  repaint();

  canvas.addEventListener("mousemove", (e) => {
    const box = canvas.getBoundingClientRect();
    const x = ((e.clientX - box.left) / box.width) * width;
    const y = ((e.clientY - box.top) / box.height) * height;
    const idx = placed.findIndex((p) => x >= p.rect.x && x <= p.rect.x + p.rect.w && y >= p.rect.y && y <= p.rect.y + p.rect.h);
    if (idx >= 0) {
      repaint(idx);
      tooltip.classList.add("is-visible");
      tooltip.textContent = `${placed[idx].word}: ${placed[idx].count}`;
      tooltip.style.left = `${e.clientX - box.left + 10}px`;
      tooltip.style.top = `${e.clientY - box.top - 26}px`;
      canvas.style.cursor = "pointer";
    } else {
      repaint(-1);
      tooltip.classList.remove("is-visible");
      canvas.style.cursor = "default";
    }
  });

  canvas.addEventListener("mouseleave", () => {
    repaint(-1);
    tooltip.classList.remove("is-visible");
    canvas.style.cursor = "default";
  });

  frame.append(canvas, tooltip);
  container.append(frame);
}

function buildPane(titleText, tabsClass, tabClass) {
  const pane = document.createElement("div");
  pane.className = "dataset-pane";

  const head = document.createElement("div");
  head.className = "dataset-pane-head";
  const title = document.createElement("h3");
  title.className = "dataset-pane-title";
  title.textContent = titleText;
  head.append(title);

  const tabs = document.createElement("div");
  tabs.className = tabsClass;
  tabs.setAttribute("role", "tablist");

  const panel = document.createElement("div");
  panel.className = "dataset-pane-panel";

  pane.append(head, tabs, panel);
  return { pane, tabs, panel, tabClass };
}

function renderSwitchTabs(pane, items, onSelect) {
  const { tabs, tabClass } = pane;
  tabs.innerHTML = "";
  let activeId = items[0]?.id || "";

  function repaintTab() {
    tabs.querySelectorAll("button").forEach((btn) => {
      const isActive = btn.dataset.id === activeId;
      btn.classList.toggle("is-active", isActive);
      btn.setAttribute("aria-selected", String(isActive));
    });
    if (activeId) onSelect(activeId);
  }

  items.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = tabClass;
    btn.dataset.id = item.id;
    btn.setAttribute("role", "tab");
    btn.textContent = item.title;
    btn.addEventListener("click", () => {
      activeId = item.id;
      repaintTab();
    });
    tabs.append(btn);
  });
  repaintTab();
}

function toRows(statsObj) {
  return Object.entries(statsObj || {}).map(([key, value]) => ({ key, value: Number(value) || 0 }));
}

function renderTabContent(container, tab) {
  container.innerHTML = "";

  const dual = document.createElement("div");
  dual.className = "dataset-dual";

  const statsPane = buildPane("Distribution Statistics", "dataset-stat-tabs", "dataset-stat-tab");
  const wcPane = buildPane("Semantic Word Clouds", "dataset-wc-tabs", "dataset-wc-tab");
  dual.append(statsPane.pane, wcPane.pane);
  container.append(dual);

  Promise.all([
    fetchJson(tab.statsPath),
    ...tab.wordclouds.map((w) => fetchJson(w.path))
  ]).then(([statsJson, ...wcJsons]) => {
    const wcMap = new Map();
    tab.wordclouds.forEach((cfg, idx) => wcMap.set(cfg.id, wcJsons[idx] || []));

    renderSwitchTabs(statsPane, tab.stats, (statId) => {
      const data = toRows((statsJson || {})[statId]);
      drawBarChart(statsPane.panel, data, tab.theme, statId);
    });

    renderSwitchTabs(wcPane, tab.wordclouds, (wcId) => {
      drawWordCloud(wcPane.panel, wcMap.get(wcId) || [], wcId, tab.theme, `${tab.id}_${wcId}`);
    });
  });
}

export function createDatasetModule(root, tabs) {
  if (!root || !tabs || !tabs.length) return;

  const tabRow = document.createElement("div");
  tabRow.className = "dataset-tabs";
  tabRow.setAttribute("role", "tablist");

  const panel = document.createElement("div");
  panel.className = "dataset-panel";
  root.append(tabRow, panel);

  let activeId = tabs[0].id;

  function repaint() {
    tabRow.querySelectorAll("button").forEach((btn) => {
      const isActive = btn.dataset.id === activeId;
      btn.classList.toggle("is-active", isActive);
      btn.setAttribute("aria-selected", String(isActive));
    });
    const activeTab = tabs.find((t) => t.id === activeId) || tabs[0];
    renderTabContent(panel, activeTab);
  }

  tabs.forEach((tab) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dataset-tab";
    btn.dataset.id = tab.id;
    btn.setAttribute("role", "tab");
    btn.textContent = tab.title;
    btn.addEventListener("click", () => {
      activeId = tab.id;
      repaint();
    });
    tabRow.append(btn);
  });

  repaint();
}
