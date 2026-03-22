import { expressionConfig } from "./data/expressions.js";
import { benchmarkTabs } from "./data/benchmark.js";
import { datasetTabs } from "./data/dataset.js";
import { createSliderModule } from "./modules/slider.js";
import { createBenchmarkModule } from "./modules/benchmark.js";
import { createBlendingModule } from "./modules/blending.js";
import { createDatasetModule } from "./modules/dataset.js";

function init() {
  const datasetRoot = document.querySelector("#dataset-root");
  const sliderRoot = document.querySelector("#slider-root");
  const benchmarkRoot = document.querySelector("#benchmark-root");
  const blendingRoot = document.querySelector("#blending-root");

  if (datasetRoot) {
    createDatasetModule(datasetRoot, datasetTabs);
  }

  if (sliderRoot) {
    createSliderModule(sliderRoot, expressionConfig);
  }

  if (benchmarkRoot) {
    createBenchmarkModule(benchmarkRoot, benchmarkTabs);
  }

  if (blendingRoot) {
    createBlendingModule(blendingRoot);
  }
}

init();
