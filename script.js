
const dropArea = document.getElementById('dropArea');
const dropAreaText = document.getElementById('dropAreaText');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const userinfoDiv=document.getElementById('user-info');
const result = document.getElementById('result');

const downloadButton = document.getElementById('downloadButton');
const deleteButton = document.getElementById('deleteButton');

const APIURL ="";
const APIURL2 = 'https://my-loginapp-qo754edula-de.a.run.app';
const taskSelect = document.getElementById('taskSelect');
let tfliteModel;
let ModelLabel;
let useremail="";
// set path from user-info api
// const token= localStorage.getItem("access_token");
// if (!token) {
//     window.location.href = 'https://my-loginapp-qo754edula-de.a.run.app';
// }
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJxd2VAcXdlLmNvbSIsImV4cCI6MTcyMTk3MTIzNH0.83GEcsbehdxKqHr7NHVLARkWus8AxAajnted-Y0pWJE";


async function getuser() {
  try {
        const response = await fetch(`${APIURL2}/users/me`, {
        method: 'GET',
        headers: {
            'Authorization': `Bearer ${token}`,
        }
        }
        );
        if (!response.ok) {
            throw new Error('Failed to fetch user email');
        }
        const userData = await response.json();
        useremail = userData.email; //set useremail
        if (userData.line_name !== null && userData.line_name !== undefined) {
            userinfoDiv.innerText = "使用者: " + userData.line_name;
        } else {
            userinfoDiv.innerText = "使用者: " + userData.email;
            console.log(userData.email);
        }
        
    } 
    catch (error) {
        console.error(error);
        userinfoDiv.innerText = '無法獲取用戶訊息';
    }
}

// 下載模型
async function downloadModel() {
  const taskName = taskSelect.value;
  if (!taskName) {
      alert('Please select a task');
      return;
  }

  try {
      const response = await fetch(`${APIURL}/download_model/${taskName}`, {
          headers: {
              'Authorization': `Bearer ${token}`
          }
      });
      const model_data = await response.json();
      const url = model_data.model_file_url;
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = model_data.model_file_name;
      document.body.appendChild(a);
      a.click();
  } catch (error) {
      console.error('Error downloading model:', error);
      alert('Error downloading model');
  }
}

// 刪除模型
async function deleteModel() {
  const taskName = taskSelect.value;
  if (!taskName) {
      alert('Please select a task');
      return;
  }

  if (confirm(`Are you sure you want to delete the model for task "${taskName}"?`)) {
      try {
          const response = await fetch(`${APIURL}/delete_model/${taskName}`, {
              method: 'DELETE',
              headers: {
                  'Authorization': `Bearer ${token}`
              }
          });
          if (response.ok) {
              alert('Model deleted successfully');
              taskSelect.remove(taskSelect.selectedIndex);
              taskSelect.value = '';
          } else {
              alert('Error deleting model');
          }
      } catch (error) {
          console.error('Error deleting model:', error);
          alert('Error deleting model');
      }
  }
}

async function loadUserTasks() {
  try {
    const response = await fetch(`${APIURL}/user_tasks/`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });
    if (!response.ok) {
      throw new Error('無法獲取用戶任務');
    }
    const taskData = await response.json();
    if (taskData.user_tasks.length===0) {
      throw new Error('沒有已完成的任務');
    }
    populateTaskSelect(taskData.user_tasks);
  } catch (error) {
    console.error('載入任務失敗:', error);
  }
}

// 新增：填充任務選擇下拉選單
function populateTaskSelect(tasks) {
  taskSelect.innerHTML = '<option value="">請選擇一個任務</option>';
  tasks.forEach(task => {
    const option = document.createElement('option');
    option.value = task;
    option.textContent = task;
    taskSelect.appendChild(option);
  });
}

// load TensorFlow Lite model for the selected task
async function loadTFLiteModel(taskName) {
    try {
      const response = await fetch(`${APIURL}/download_model/${taskName}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) {
        throw new Error('無法獲取模型信息');
      }
      const modelInfo = await response.json();
      // loading the TensorFlow Lite model for url
      tfliteModel = await tflite.loadTFLiteModel(modelInfo.model_file_url);
      // ladoing the labels for the model
      ModelLabel = modelInfo.model_label;
      
      
      console.log(`已加載任務 ${taskName} 的模型`);
    } catch (error) {
      console.error('Error loading model', error);
    }
  }
  
  // update the tfliteModel with the selected task
  taskSelect.addEventListener('change', (event) => {
    const selectedTask = event.target.value;
    if (selectedTask) {
      loadTFLiteModel(selectedTask);
    }
  });

  downloadButton.addEventListener('click', downloadModel);
  deleteButton.addEventListener('click', deleteModel);

  
 



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
    imagePreview.onload = function() {
        // 確保圖像已設置並加載完成後調用 classifyImage 函數
        classifyImage(imagePreview);
    }
}

async function classifyImage(image) {
   if (!tfliteModel) {
    throw new Error("Model not loaded");
  }
  // show start stage
  result.innerText = "Classifying...";
  let tensor;
  try {
    // preprocess the image scale=1./127.5 offset=-1 ,-1~1
    const tensor = tf.tidy(() => {
      const imageTensor = tf.browser.fromPixels(image);
      const resized = tf.image.resizeNearestNeighbor(imageTensor, [224, 224]);
      const offset = tf.scalar(127.5);
      const normalizedImageData = resized.sub(offset).div(offset);
      return normalizedImageData.cast('float32').expandDims(0);
    });

    const predictions = await tfliteModel.predict(tensor);
    // get probabilities
    const probabilities = await predictions.data();
    console.log(probabilities)
    
    // get the classification results
    console.log(ModelLabel)
    const results = ModelLabel.map((label, index) => ({
      label,
      probability: probabilities[index]
    }));

    // sort by probability
    results.sort((a, b) => b.probability - a.probability);

    // use requestAnimationFrame to update the UI
    requestAnimationFrame(() => {
      // create the result string
      // const resultString = results.map(item => 
      //   `${item.label}: ${(item.probability * 100).toFixed(2)}%`
      // ).join('\n');
      const topResult = results[0]; // get the top result
      const resultString = `${topResult.label}: ${(topResult.probability * 100).toFixed(2)}%`;
      result.innerText = `Classification Result:\n${resultString}`;
    });
    await getGradCAM(image);
    return results; // return the results for further processing

  } catch (error) {
    console.error('Classification error:', error);
    result.innerText = "An error occurred during classification.";
  } finally {
    // 確保釋放 tensor 的內存
    if (tensor) tensor.dispose();
  }
}

async function getGradCAM(image) {
  const taskName = taskSelect.value;
  if (!taskName) {
    console.error('No task selected');
    return;
  }

  // Convert base64 to blob
  const base64Response = await fetch(image.src);
  const blob = await base64Response.blob();

  // Create a File object
  const file = new File([blob], "image.jpg", { type: "image/jpeg" });

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${APIURL}/predict/${taskName}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      },
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch Grad-CAM image: ${response.status} ${response.statusText}`);
    }

    const gradcamBlob = await response.blob();
    const gradcamUrl = URL.createObjectURL(gradcamBlob);
    
    const gradcamImage = document.getElementById('gradcamImage');
    gradcamImage.src = gradcamUrl;
    gradcamImage.style.display = 'block';
  } catch (error) {
    console.error('Error fetching Grad-CAM:', error);
  }
}


document.addEventListener('DOMContentLoaded', async () => {
  await getuser();
  await loadUserTasks();
});