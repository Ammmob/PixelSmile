const sliderExpressions = [
  "angry",
  "anxious",
  "confident",
  "confused",
  "contempt",
  "disgust",
  "fear",
  "happy",
  "sad",
  "shy",
  "sleepy",
  "surprised"
];

export const assetManifest = {
  slider: Object.fromEntries(
    sliderExpressions.map((name) => [
      name,
      Array.from({ length: 28 }, (_, i) => `${String(i).padStart(2, "0")}.avif`)
    ])
  ),
  blending: {
    ids: ["000001"],
    filesById: {
      "000001": [
        "angry.jpg",
        "disgust.jpg",
        "disgust_fear.jpg",
        "fear.jpg",
        "happy.jpg",
        "happy_disgust.jpg",
        "happy_fear.jpg",
        "happy_sad.jpg",
        "happy_surprised.jpg",
        "sad.jpg",
        "sad_angry.jpg",
        "sad_disgust.jpg",
        "sad_fear.jpg",
        "sad_surprised.jpg",
        "surprised.jpg"
      ]
    }
  }
};
