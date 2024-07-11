import { ImageClassifier, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

const dropArea = document.getElementById('dropArea');
const dropAreaText = document.getElementById('dropAreaText');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const userinfoDiv=document.getElementById('user-info');
const result = document.getElementById('result');

const downloadButton = document.getElementById('downloadButton');
const deleteButton = document.getElementById('deleteButton');

const APIURL ="";
const APIURL2 = 'https://my-loginapp2-qo754edula-uc.a.run.app';
const taskSelect = document.getElementById('taskSelect');
let imageClassifier;
let useremail="";
// set path from user-info api
// const token= localStorage.getItem("token");
// if (!token) {
//     window.location.href = 'login.html';
// }
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhc2RAYXNkLmNvbSIsImV4cCI6MTcyMDk3MTg5OX0.b7iIlihbwHDBcMj7cXRzY-eUuqrwUofOwKgflRWWq3Y";


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
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `${taskName}_model.pth`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
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





// useremail="qwe@qwe.com";

// 修改：根據選擇的任務創建圖像分類器
async function createImageClassifier(taskName) {
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
      
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
      );
      imageClassifier = await ImageClassifier.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelInfo.model_file_url
        },
        maxResults: 1,
        runningMode: 'IMAGE'
      });
      
      console.log(`已加載任務 ${taskName} 的模型`);
    } catch (error) {
      console.error('創建圖像分類器失敗:', error);
    }
  }
  
  // 修改：當任務選擇改變時更新分類器
  taskSelect.addEventListener('change', (event) => {
    const selectedTask = event.target.value;
    if (selectedTask) {
      createImageClassifier(selectedTask);
    }
  });
  downloadButton.addEventListener('click', downloadModel);
  deleteButton.addEventListener('click', deleteModel);

  getuser();
  loadUserTasks();



// const createImageClassifier = async () => {
//   const vision = await FilesetResolver.forVisionTasks(
//     "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
//   );  
//   imageClassifier = await ImageClassifier.createFromOptions(vision, {
//     baseOptions: {
//       modelAssetPath: `exported_model_test/${useremail}/model.tflite`
//     },
//     maxResults: 1,
//     runningMode: 'IMAGE'
//   });
// };



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
        classifyImage(imagePreview);
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
