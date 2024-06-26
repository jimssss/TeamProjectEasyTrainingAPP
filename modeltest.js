import { ImageClassifier, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

const imageInput = document.getElementById('imageInput');
const classifyButton = document.getElementById('classifyButton');
const result = document.getElementById('result');
let imageClassifier;

const createImageClassifier = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  imageClassifier = await ImageClassifier.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'exported_model_test/model.tflite'
    },
    maxResults: 1,
    runningMode: 'IMAGE'
  });
};

createImageClassifier();

async function classifyImage(image) {
  if (!imageClassifier) {
    return;
  }
  const classificationResult = await imageClassifier.classify(image);
  const classifications = classificationResult.classifications;
  result.innerText =
    "Classification: " +
    classifications[0].categories[0].categoryName +
    "\nConfidence: " +
    Math.round(parseFloat(classifications[0].categories[0].score) * 100) +
    "%";
}

imageInput.addEventListener('change', event => {
  const file = event.target.files[0];
  if (file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => classifyImage(img);
  }
});

classifyButton.addEventListener('click', () => {
  const file = imageInput.files[0];
  if (file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => classifyImage(img);
  }
});
