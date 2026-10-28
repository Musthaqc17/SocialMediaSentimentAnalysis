
const API_URL = "http://127.0.0.1:5000";

// Text sentiment analysis
const sentimentForm = document.getElementById("sentimentForm");
const postText = document.getElementById("postText");
const modelSelect = document.getElementById("modelSelect");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");

const result = document.getElementById("result");
const sentimentValue = document.getElementById("sentimentValue");
const modelValue = document.getElementById("modelValue");
const textValue = document.getElementById("textValue");

// Image sentiment analysis
const imageForm = document.getElementById("imageForm");
const imageFile = document.getElementById("imageFile");
const imageModelSelect = document.getElementById("imageModelSelect");
const imageAnalyzeButton = document.getElementById("imageAnalyzeButton");

const imageResult = document.getElementById("imageResult");
const extractedTextValue = document.getElementById("extractedTextValue");
const imageSentimentValue = document.getElementById("imageSentimentValue");
const imageModelValue = document.getElementById("imageModelValue");

function showError(message) {
    statusMessage.textContent = message;
    statusMessage.style.color = "#d93025";
}

function showSuccess(message) {
    statusMessage.textContent = message;
    statusMessage.style.color = "#188038";
}

// Analyze text
sentimentForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = postText.value.trim();
    const model = modelSelect.value;

    if (!text) {
        showError("Please enter some text.");
        return;
    }

    analyzeButton.disabled = true;
    result.classList.add("hidden");
    imageResult.classList.add("hidden");
    showSuccess("Analyzing text...");

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text, model })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Text analysis failed.");
        }

        sentimentValue.textContent =
            data.sentiment ?? data.prediction ?? "Unavailable";

        modelValue.textContent = model;
        textValue.textContent = text;

        result.classList.remove("hidden");
        showSuccess("Text analysis completed.");

    } catch (error) {
        showError(error.message || "Could not connect to the backend.");
    } finally {
        analyzeButton.disabled = false;
    }
});

// Analyze an uploaded image
imageForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageFile.files[0];

    if (!file) {
        showError("Please choose an image.");
        return;
    }

    const allowedTypes = ["image/png", "image/jpeg"];

    if (!allowedTypes.includes(file.type)) {
        showError("Please upload a PNG or JPG image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("Image size must be 10 MB or less.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("model", imageModelSelect.value);

    imageAnalyzeButton.disabled = true;
    imageResult.classList.add("hidden");
    result.classList.add("hidden");
    showSuccess("Extracting text and analyzing image...");

    try {
        const response = await fetch(`${API_URL}/predict-image`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Image analysis failed.");
        }

        const prediction = data.sentiment_result || {};

        extractedTextValue.textContent = data.extracted_text || "";
        imageSentimentValue.textContent =
            prediction.sentiment ??
            prediction.prediction ??
            "Check the prediction response";

        imageModelValue.textContent = imageModelSelect.value;

        imageResult.classList.remove("hidden");
        showSuccess("Image analysis completed.");

    } catch (error) {
        showError(error.message || "Could not connect to the backend.");
    } finally {
        imageAnalyzeButton.disabled = false;
    }
});

