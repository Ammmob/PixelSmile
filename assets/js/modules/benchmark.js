function safeMetric(value) {
  return typeof value === "number" ? value : null;
}

function metricExtent(methods, metricKey) {
  const values = methods.map((m) => safeMetric(m.metric[metricKey])).filter((v) => v !== null);
  return { min: Math.min(...values), max: Math.max(...values) };
}

function axisRange(metric, extent) {
  const span = extent.max - extent.min;
  const pad = span > 0 ? span * 0.18 : Math.max(Math.abs(extent.max) * 0.2, 0.2);
  let min = extent.min - pad;
  let max = extent.max + pad;

  if (metric.key === "mSCR") {
    // Revert to previous mSCR behavior.
    if (extent.max <= 0) max = 0;
    if (extent.max > 0) max = Math.min(max, 1.0);
    if (max <= min) min = max - 0.1;
    return { min, max };
  }

  if (metric.key === "Acc6") {
    // User rule: Acc-6 top should be at most 0.9.
    max = Math.min(0.9, max);
    min = Math.max(0, min);
  } else if (metric.key === "CLS6" || metric.key === "CLS12") {
    // CLS can be negative; keep signed range for visibility.
    max = Math.min(1.0, max);
  } else {
    // Other bounded metrics: top should be at most 1.0.
    max = Math.min(1.0, max);
    min = Math.max(0, min);
  }
  if (max <= min) min = max - 0.1;
  return { min, max };
}

function normalizedHeight(value, extent, invert) {
  if (value === null) return 0;
  if (extent.max === extent.min) return 100;
  const raw = ((value - extent.min) / (extent.max - extent.min)) * 100;
  const mapped = invert ? 100 - raw : raw;
  return Math.max(0, Math.min(100, mapped));
}

function shortName(name) {
  return name
    .split(/[\s()-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((x) => x[0].toUpperCase())
    .join("");
}

function fallbackNode(name) {
  const node = document.createElement("span");
  node.className = "method-fallback";
  node.textContent = shortName(name);
  return node;
}

function buildTicks(extent, invert, count = 5) {
  const ticks = [];
  for (let i = 0; i < count; i += 1) {
    const t = i / (count - 1);
    const value = invert
      ? extent.min + (extent.max - extent.min) * t
      : extent.max - (extent.max - extent.min) * t;
    ticks.push(value);
  }
  return ticks;
}

function fmtTick(v) {
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(3);
}

export function createBenchmarkModule(root, tabs) {
  let activeTab = 0;

  const tabRow = document.createElement("div");
  tabRow.className = "bench-tabs";

  const legend = document.createElement("div");
  legend.className = "bench-legend";

  const chartStage = document.createElement("div");
  chartStage.className = "benchmark-stage";

  root.append(tabRow, legend, chartStage);

  function renderTabs() {
    tabRow.innerHTML = "";
    tabs.forEach((tab, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `bench-tab${idx === activeTab ? " is-active" : ""}`;
      btn.textContent = tab.label;
      btn.addEventListener("click", () => {
        activeTab = idx;
        render();
      });
      tabRow.append(btn);
    });
  }

  function renderLegend() {
    const current = tabs[activeTab];
    legend.innerHTML = "";

    current.methods.forEach((method) => {
      const item = document.createElement("div");
      item.className = "bench-legend-item";

      if (current.showLogos || method.logo) {
        const logo = document.createElement("img");
        logo.className = "bench-legend-logo";
        logo.alt = `${method.name} logo`;
        logo.src = method.logo || "";
        logo.onerror = () => logo.replaceWith(fallbackNode(method.name));
        item.append(logo);
      } else {
        const initials = document.createElement("span");
        initials.className = "bench-legend-initials";
        initials.textContent = method.initials || shortName(method.name);
        item.append(initials);
      }

      const name = document.createElement("span");
      name.className = "bench-legend-name";
      name.textContent = method.name;
      item.append(name);

      legend.append(item);
    });
  }

  function createMetricCard(current, metric) {
    const displayMethods = current.methods;

    const extent = metricExtent(current.methods, metric.key);
    const axis = axisRange(metric, extent);

    const card = document.createElement("div");
    card.className = "benchmark-card";

    const cardTitle = document.createElement("div");
    cardTitle.className = "benchmark-card-title";
    cardTitle.textContent = metric.label;

    const chartWithAxis = document.createElement("div");
    chartWithAxis.className = "chart-with-axis";

    const yAxis = document.createElement("div");
    yAxis.className = "y-axis";
    buildTicks(axis, metric.invert).forEach((tick) => {
      const node = document.createElement("div");
      node.className = "y-tick";
      node.textContent = fmtTick(tick);
      yAxis.append(node);
    });

    const plotWrap = document.createElement("div");
    plotWrap.className = "plot-wrap";

    const yGrid = document.createElement("div");
    yGrid.className = "y-grid";
    buildTicks(axis, metric.invert).forEach(() => {
      const line = document.createElement("div");
      line.className = "y-grid-line";
      yGrid.append(line);
    });

    const vbars = document.createElement("div");
    vbars.className = "vbar-grid";
    vbars.style.gridTemplateColumns = `repeat(${displayMethods.length}, minmax(0, 1fr))`;

    displayMethods.forEach((method) => {
      const value = safeMetric(method.metric[metric.key]);
      const height = normalizedHeight(value, axis, metric.invert);

      const item = document.createElement("div");
      item.className = "vbar-item";
      item.tabIndex = 0;
      item.setAttribute("aria-label", `${method.name}: ${value === null ? "N/A" : value.toFixed(4)}`);

      const tooltip = document.createElement("div");
      tooltip.className = "vbar-tooltip";
      tooltip.classList.add("vbar-tooltip-main");
      tooltip.innerHTML = `
        <div class="vbar-tooltip-name">${method.name}</div>
        <div class="vbar-tooltip-score">${value === null ? "N/A" : value.toFixed(4)}</div>
      `;
      tooltip.style.bottom = `calc(${Math.max(height, 12)}% + 40px)`;

      const col = document.createElement("div");
      col.className = "vbar-col";

      const score = document.createElement("div");
      score.className = "vbar-score";
      score.textContent = value === null ? "N/A" : value.toFixed(4);
      score.style.bottom = `calc(${Math.max(height, 12)}% + 6px)`;
      col.append(score);

      const fill = document.createElement("div");
      fill.className = `vbar-fill${method.ours ? " is-ours" : ""}`;
      fill.style.height = `${Math.max(height, 12)}%`;

      if (current.showLogos || (method.ours && method.logo)) {
        const logo = document.createElement("img");
        logo.className = "vbar-logo";
        logo.alt = `${method.name} logo`;
        logo.src = method.logo || "";
        logo.onerror = () => logo.replaceWith(fallbackNode(method.name));
        fill.append(logo);
      } else {
        const initials = document.createElement("span");
        initials.className = "vbar-initials";
        initials.textContent = method.initials || shortName(method.name);
        fill.append(initials);
      }

      col.append(fill);
      col.append(tooltip);
      item.append(col);

      vbars.append(item);
    });

    plotWrap.append(yGrid, vbars);
    chartWithAxis.append(yAxis, plotWrap);
    card.append(cardTitle, chartWithAxis);
    return card;
  }

  function renderBars() {
    const current = tabs[activeTab];
    chartStage.innerHTML = "";
    chartStage.className = "benchmark-stage is-grid-2x2";
    current.metrics.forEach((metric) => {
      chartStage.append(createMetricCard(current, metric));
    });
  }

  function render() {
    renderTabs();
    renderLegend();
    renderBars();
  }

  render();
}
