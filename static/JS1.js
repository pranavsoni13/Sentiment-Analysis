async function analyzeTweet() {
    const tweet = document.getElementById("tweetInput").value;
    const resultDiv = document.getElementById("result");
    resultDiv.textContent = "Analyzing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ tweet: tweet })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.prediction) {
            resultDiv.textContent = `Prediction: ${data.prediction}`;
        } else {
            resultDiv.textContent = "Error: Unexpected response format.";
        }
    } catch (error) {
        console.error("Fetch error:", error);
        resultDiv.textContent = "An error occurred while analyzing the tweet.";
    }
}