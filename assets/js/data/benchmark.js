export const benchmarkTabs = [
  {
    id: "general",
    label: "General Editing",
    showLogos: true,
    metrics: [
      { key: "mSCR", label: "mSCR (lower is better)", invert: true },
      { key: "Acc6", label: "Acc-6 (higher is better)", invert: false },
      { key: "Acc12", label: "Acc-12 (higher is better)", invert: false },
      { key: "IDSim", label: "ID Sim (higher is better)", invert: false }
    ],
    methods: [
      { name: "Seedream", metric: { mSCR: 0.3725, Acc6: 0.5294, Acc12: 0.3737, IDSim: 0.7221 }, logo: "./assets/img/logos/Seedream.png" },
      { name: "Nano Banana Pro", metric: { mSCR: 0.1754, Acc6: 0.8431, Acc12: 0.62, IDSim: 0.7107 }, logo: "./assets/img/logos/Nano Banana Pro.png" },
      { name: "GPT-Image", metric: { mSCR: 0.1107, Acc6: 0.8039, Acc12: 0.63, IDSim: 0.5056 }, logo: "./assets/img/logos/GPT-Image.png" },
      { name: "FLUX-Klein", metric: { mSCR: 0.285, Acc6: 0.451, Acc12: 0.331, IDSim: 0.4146 }, logo: "./assets/img/logos/FLUX-Klein.png" },
      { name: "LongCat", metric: { mSCR: 0.1754, Acc6: 0.6275, Acc12: 0.41, IDSim: 0.6036 }, logo: "./assets/img/logos/LongCat.png" },
      { name: "Qwen-Edit", metric: { mSCR: 0.2625, Acc6: 0.451, Acc12: 0.29, IDSim: 0.6938 }, logo: "./assets/img/logos/Qwen-Edit.png" },
      { name: "Ours", metric: { mSCR: 0.055, Acc6: 0.8627, Acc12: 0.6, IDSim: 0.6522 }, logo: "./assets/img/logos/PixelSmile.png", ours: true }
    ]
  },
  {
    id: "linear",
    label: "Linear Control",
    showLogos: false,
    metrics: [
      { key: "CLS6", label: "CLS-6 (higher is better)", invert: false },
      { key: "CLS12", label: "CLS-12 (higher is better)", invert: false },
      { key: "HES", label: "HES (higher is better)", invert: false },
      { key: "IDSim", label: "ID Sim (higher is better)", invert: false }
    ],
    methods: [
      { name: "SAEdit", initials: "SAE", metric: { CLS6: -0.0183, CLS12: 0.0007, HES: null, IDSim: null } },
      { name: "ConceptSlider", initials: "CS", metric: { CLS6: 0.3161, CLS12: null, HES: 0.3656, IDSim: 0.625 } },
      { name: "AttributeControl", initials: "AC", metric: { CLS6: 0.2856, CLS12: null, HES: 0.2712, IDSim: 0.3609 } },
      { name: "K-Slider", initials: "KS", metric: { CLS6: -0.0459, CLS12: -0.0634, HES: 0.3272, IDSim: 0.7974 } },
      { name: "SliderEdit", initials: "SE", metric: { CLS6: 0.5599, CLS12: 0.5217, HES: 0.3441, IDSim: 0.7414 } },
      { name: "Ours", logo: "./assets/img/logos/PixelSmile.png", metric: { CLS6: 0.8078, CLS12: 0.7305, HES: 0.4723, IDSim: 0.6522 }, ours: true }
    ]
  }
];
