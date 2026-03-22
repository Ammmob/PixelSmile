export const datasetTabs = [
  {
    id: "human",
    title: "Human Data",
    theme: "human",
    statsPath: "./assets/data/dataset/human/categorical_stats.json",
    stats: [
      { id: "age_group", title: "Age Distribution" },
      { id: "skin_tone", title: "Skin Tone Distribution" }
    ],
    wordclouds: [
      {
        id: "appearance_sentence",
        title: "Appearance",
        path: "./assets/data/dataset/human/top_words_appearance_sentence.json"
      },
      {
        id: "action_sentence",
        title: "Action",
        path: "./assets/data/dataset/human/top_words_action_sentence.json"
      },
      {
        id: "background_sentence",
        title: "Background",
        path: "./assets/data/dataset/human/top_words_background_sentence.json"
      }
    ]
  },
  {
    id: "anime",
    title: "Anime Data",
    theme: "anime",
    statsPath: "./assets/data/dataset/anime/categorical_stats.json",
    stats: [
      { id: "age_group", title: "Age Distribution" },
      { id: "anime_style", title: "Style Distribution" }
    ],
    wordclouds: [
      {
        id: "appearance_sentence",
        title: "Appearance",
        path: "./assets/data/dataset/anime/top_words_appearance_sentence.json"
      },
      {
        id: "action_sentence",
        title: "Action",
        path: "./assets/data/dataset/anime/top_words_action_sentence.json"
      },
      {
        id: "background_sentence",
        title: "Background",
        path: "./assets/data/dataset/anime/top_words_background_sentence.json"
      }
    ]
  }
];

export const datasetStyle = {
  topKWords: 0,
  chartColors: {
    human: {
      bar: "#c47a6d",
      barHover: "#a96458",
      barTop: "#d7998f",
      barBottom: "#a96458",
      axis: "#7f5a57",
      grid: "#ead7d4"
    },
    anime: {
      bar: "#2f6d82",
      barHover: "#25586a",
      barTop: "#5d93a6",
      barBottom: "#25586a",
      axis: "#456779",
      grid: "#d7e4ea"
    }
  },
  wordColors: {
    human: ["#8d5d57", "#a96e64", "#bf8075", "#d19489", "#9e746d"],
    anime: ["#245667", "#2f6d82", "#4b8096", "#6c9db0", "#5e7e9a"]
  },
  cloudLayouts: {
    flower: { mode: "spiral", fontMin: 10, fontMax: 36, rotateRatio: 0.1, spiralStep: 0.27, radiusLimit: 290 },
    flow: { mode: "wave_fill", fontMin: 10, fontMax: 34, rotateRatio: 0.06, spiralStep: 0.24, radiusLimit: 300 },
    vertical: { mode: "vertical_fill", fontMin: 10, fontMax: 33, rotateRatio: 0.14, spiralStep: 0.22, radiusLimit: 300 },
    horizontal: { mode: "horizontal_fill", fontMin: 10, fontMax: 33, rotateRatio: 0.06, spiralStep: 0.22, radiusLimit: 300 }
  }
};

export const customStopwords = {
  global: [
    "visible",
    "one",
    "while",
    "has",
    "over",
    "around",
    "near",
    "under",
    "small",
    "large",
    "plain",
    "textured",
    "both",
  ],
  appearance_sentence: [
    "wears",
    "wearing",
    "holds",
    "holding",
    "framing",
    "falling",
    "styled",
    "face",
    "shoulder",
    "shoulders",
    "chest",
    "neck",
    "child",
    "light-colored",
    "top"
  ],
  action_sentence: [
    "camera",
    "directly",
    "forward",
    "slightly",
    "gently",
    "broadly",
    "calm",
    "relaxed",
    "composed",
    "steady",
    "gentle",
    "posture",
    "expression",
    "gaze",
    "against",
    "off-camera",
    "toward",
    "right",
    "left"
  ],
  background_sentence: [
    "background",
    "setting",
    "suggesting",
    "shows",
    "includes",
    "features",
    "creating",
    "consists",
    "subject",
    "scene",
    "objects",
    "textures",
    "indistinct",
    "environmental",
    "partially",
    "through",
    "part"
  ]
};
