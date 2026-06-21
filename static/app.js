const btnStartRecord = document.getElementById("btnStartRecord");
const btnStopRecord = document.getElementById("btnStopRecord");
const btnTranscribe = document.getElementById("btnTranscribe");
const btnTranscribeAndLatex = document.getElementById("btnTranscribeAndLatex");
const btnToLatex = document.getElementById("btnToLatex");
const btnClear = document.getElementById("btnClear");
const btnCopyLatex = document.getElementById("btnCopyLatex");
const btnInsertDemo = document.getElementById("btnInsertDemo");

const audioFile = document.getElementById("audioFile");
const audioPreview = document.getElementById("audioPreview");
const recognizedText = document.getElementById("recognizedText");
const latexOutput = document.getElementById("latexOutput");
const renderBox = document.getElementById("renderBox");
const asrStatus = document.getElementById("asrStatus");
const latexStatus = document.getElementById("latexStatus");
const copyStatus = document.getElementById("copyStatus");
const demoSelect = document.getElementById("demoSelect");
const demoPreview = document.getElementById("demoPreview");

let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;

const demoExamples = [
  {
    title: "Простая дробь",
    text: "дробь икс плюс один деленного на игрек",
  },
  {
    title: "Логарифм",
    text: "логарифм по основанию три от икс",
  },
  {
    title: "Корень",
    text: "корень из икс в степени три",
  },
  {
    title: "Производная",
    text: "производная по икс от косеканс икс",
  },
  {
    title: "Предел",
    text: "предел при икс стремится к нулю от синус икс деленное на икс",
  },
  {
    title: "Сумма",
    text: "сумма от эн равно один до десять от икс в степени эн",
  },
  {
    title: "Логарифм натуральный",
    text: "натуральный логарифм ка",
  },
  {
    title: "Отношение",
    text: "икс в квадрате минус три икс плюс два равно нулю",
  }
];

function setStatus(el, text, isError = false) {
  el.textContent = text || "";
  el.classList.toggle("error", Boolean(isError));
}

function getCurrentAudioBlob() {
  if (recordedBlob) {
    return recordedBlob;
  }
  const file = audioFile.files && audioFile.files[0];
  return file || null;
}

function updateAudioPreviewFromBlob(blob) {
  if (!blob) {
    audioPreview.removeAttribute("src");
    audioPreview.load();
    return;
  }
  const url = URL.createObjectURL(blob);
  audioPreview.src = url;
}

function initDemoExamples() {
  demoExamples.forEach((item, idx) => {
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = item.title;
    demoSelect.appendChild(option);
  });

  demoSelect.addEventListener("change", () => {
    const idx = demoSelect.value;
    if (idx === "") {
      demoPreview.textContent = "";
      return;
    }
    demoPreview.textContent = demoExamples[Number(idx)].text;
  });
}

function insertSelectedDemo() {
  const idx = demoSelect.value;
  if (idx === "") {
    setStatus(latexStatus, "Сначала выбери демонстрационный пример.", true);
    return;
  }

  const example = demoExamples[Number(idx)];
  recognizedText.value = example.text;
  setStatus(latexStatus, `Подставлен пример: ${example.title}`);
}

async function startRecording() {
  setStatus(asrStatus, "");
  setStatus(latexStatus, "");
  setStatus(copyStatus, "");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    recordedChunks = [];
    recordedBlob = null;

    const options = {};
    if (MediaRecorder.isTypeSupported("audio/webm")) {
      options.mimeType = "audio/webm";
    }

    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      updateAudioPreviewFromBlob(recordedBlob);

      for (const track of stream.getTracks()) {
        track.stop();
      }
    };

    mediaRecorder.start();

    btnStartRecord.disabled = true;
    btnStopRecord.disabled = false;
    setStatus(asrStatus, "Идёт запись...");
  } catch (err) {
    setStatus(asrStatus, `Ошибка доступа к микрофону: ${err.message}`, true);
  }
}

function stopRecording() {
  if (!mediaRecorder) {
    return;
  }

  mediaRecorder.stop();
  btnStartRecord.disabled = false;
  btnStopRecord.disabled = true;
  setStatus(asrStatus, "Запись завершена.");
}

audioFile.addEventListener("change", () => {
  recordedBlob = null;
  const file = audioFile.files && audioFile.files[0];
  if (file) {
    updateAudioPreviewFromBlob(file);
    setStatus(asrStatus, `Выбран файл: ${file.name}`);
  }
});

btnStartRecord.addEventListener("click", startRecording);
btnStopRecord.addEventListener("click", stopRecording);
btnInsertDemo.addEventListener("click", insertSelectedDemo);

async function transcribeOnly() {
  const blob = getCurrentAudioBlob();
  if (!blob) {
    setStatus(asrStatus, "Сначала запиши или загрузи аудио.", true);
    return;
  }

  setStatus(asrStatus, "Локальное распознавание...");
  const form = new FormData();
  form.append("audio", blob, blob.name || "recording.webm");

  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      body: form,
    });

    const data = await res.json();

    if (!data.ok) {
      throw new Error(data.error || "Не удалось распознать аудио.");
    }

    recognizedText.value = data.text || "";
    setStatus(asrStatus, `Готово. ${data.asr_backend}, ${data.seconds} c`);
  } catch (err) {
    setStatus(asrStatus, err.message, true);
  }
}

async function transcribeAndLatex() {
  const blob = getCurrentAudioBlob();
  if (!blob) {
    setStatus(asrStatus, "Сначала запиши или загрузи аудио.", true);
    return;
  }

  setStatus(asrStatus, "Локальное распознавание и генерация...");
  setStatus(latexStatus, "");
  setStatus(copyStatus, "");

  const form = new FormData();
  form.append("audio", blob, blob.name || "recording.webm");

  try {
    const res = await fetch("/api/transcribe-and-latex", {
      method: "POST",
      body: form,
    });

    const data = await res.json();

    if (!data.ok) {
      throw new Error(data.error || "Ошибка комбинированного запроса.");
    }

    recognizedText.value = data.text || "";
    latexOutput.value = data.latex || "";
    renderLatex(data.latex || "");

    setStatus(asrStatus, `Готово. ${data.asr_backend}, ${data.seconds} c`);
    setStatus(latexStatus, `Модель: ${data.model_name}`);
  } catch (err) {
    setStatus(asrStatus, err.message, true);
  }
}

async function textToLatex() {
  const text = recognizedText.value.trim();
  if (!text) {
    setStatus(latexStatus, "Нет текста для преобразования.", true);
    return;
  }

  setStatus(latexStatus, "Генерация LaTeX...");
  setStatus(copyStatus, "");

  try {
    const res = await fetch("/api/latex", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!data.ok) {
      throw new Error(data.error || "Не удалось получить LaTeX.");
    }

    latexOutput.value = data.latex || "";
    renderLatex(data.latex || "");
    setStatus(latexStatus, `Готово. ${data.seconds} c | ${data.model_name}`);
  } catch (err) {
    setStatus(latexStatus, err.message, true);
  }
}

async function copyLatex() {
  const text = latexOutput.value.trim();
  if (!text) {
    setStatus(copyStatus, "Нет LaTeX для копирования.", true);
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    setStatus(copyStatus, "LaTeX скопирован.");
  } catch (err) {
    setStatus(copyStatus, `Ошибка копирования: ${err.message}`, true);
  }
}

function renderLatex(latex) {
  renderBox.innerHTML = "";

  if (!latex) {
    return;
  }

  try {
    katex.render(latex, renderBox, {
      throwOnError: false,
      displayMode: true,
    });
  } catch (err) {
    renderBox.textContent = `Ошибка рендера: ${err.message}`;
  }
}

btnTranscribe.addEventListener("click", transcribeOnly);
btnTranscribeAndLatex.addEventListener("click", transcribeAndLatex);
btnToLatex.addEventListener("click", textToLatex);
btnCopyLatex.addEventListener("click", copyLatex);

btnClear.addEventListener("click", () => {
  recognizedText.value = "";
  latexOutput.value = "";
  renderBox.innerHTML = "";
  setStatus(asrStatus, "");
  setStatus(latexStatus, "");
  setStatus(copyStatus, "");
  demoSelect.value = "";
  demoPreview.textContent = "";
});

initDemoExamples();