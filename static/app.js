document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const predictBtn = document.getElementById("predictBtn");
    const resultText = document.getElementById("result");
    const previewImg = document.getElementById("preview");

    // Show image preview
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (!file) return;
        const imageURL = URL.createObjectURL(file);
        previewImg.src = imageURL;
        previewImg.style.display = "block";
        resultText.innerHTML = ""; // Clear text on new selection
    });

    // Predict button
    predictBtn.addEventListener("click", async () => {
        const file = fileInput.files[0];
        if (!file) {
            resultText.innerText = "Please select an image first.";
            return;
        }

        resultText.innerText = "Predicting...";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                resultText.innerText = "Error: " + data.error;
                return;
            }

            // Using innerHTML and <br> for the stacked look
            const confidencePercent = (data.confidence * 100).toFixed(2);
            resultText.innerHTML = `Prediction: ${data.prediction}<br>Confidence: ${confidencePercent}%`;

        } catch (error) {
            resultText.innerText = "Error occurred during prediction.";
            console.error(error);
        }
    });
});