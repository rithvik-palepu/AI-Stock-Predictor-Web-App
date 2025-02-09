document.addEventListener("DOMContentLoaded", function () {
    const submitBtn = document.getElementById("submit-btn");
    const tickerInput = document.getElementById("stock-input");
    const loadingElement = document.getElementById("loading");
    const imageElement = document.getElementById("stock-plot");

    // Ensure the spinner is hidden when the page loads
    loadingElement.style.display = "none";
    imageElement.style.display = "none";

    submitBtn.addEventListener("click", function () {
        const ticker = tickerInput.value.trim().toUpperCase();
        if (!ticker) {
            alert("Please enter a stock ticker.");
            return;
        }

        console.log("Sending request with ticker:", ticker);

        // Show loading spinner and hide image
        loadingElement.style.display = "flex";
        imageElement.style.display = "none";

        fetch("https://ai-stock-predictor-web-app.onrender.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ ticker: ticker }),
        })
            .then((response) => {
                console.log("Response received:", response);
                if (!response.ok) {
                    throw new Error("Error fetching prediction plot.");
                }
                return response.blob();
            })
            .then((imageBlob) => {
                console.log("Image Blob:", imageBlob);
                const imageUrl = URL.createObjectURL(imageBlob);

                // Hide the loading spinner and show the image
                loadingElement.style.display = "none";
                imageElement.src = imageUrl;
                imageElement.style.display = "block";
                console.log("Image successfully loaded:", imageUrl);
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("Failed to fetch prediction plot.");
                loadingElement.style.display = "none"; // Hide spinner on error
            });
    });
});
