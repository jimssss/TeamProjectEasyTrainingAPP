import { ImageClassifier, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

const dropArea = document.getElementById('dropArea');
const dropAreaText = document.getElementById('dropAreaText');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
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

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

// Handle selected files
imageInput.addEventListener('change', handleFiles, false);

dropArea.addEventListener('click', () => imageInput.click());

function preventDefaults (e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files instanceof FileList) {
        ([...files]).forEach(previewFile);
    } else if (files.target) {
        ([...files.target.files]).forEach(previewFile);
    }
}

function previewFile(file) {
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function() {
        imagePreview.src = reader.result;
        imagePreview.style.display = 'block';
        dropAreaText.style.display = 'none';  // 隐藏文本
    }
}

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

classifyButton.addEventListener('click', () => {
    if (imagePreview.src) {
        classifyImage(imagePreview);
    } else {
        result.textContent = "Please select an image first.";
    }
});