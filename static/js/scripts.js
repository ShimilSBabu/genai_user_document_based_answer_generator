document.getElementById('predict-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    // Collect form data
    const api_key = document.getElementById('api_key').value;
    const topic = document.getElementById('topic').value;
//    const num_questions = parseInt(document.getElementById('num_questions').value, 10);

//    const features = [
//        parseFloat(document.getElementById('feature1').value),
//        parseFloat(document.getElementById('feature2').value),
////        parseFloat(document.getElementById('feature3').value),
////        parseFloat(document.getElementById('feature4').value),
//    ];

    // Send POST request to Flask server
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, api_key })
    });

    // Parse and display the result
    const result = await response.json();
//    document.getElementById('result').textContent = result.prediction;
    document.getElementById('result').innerHTML = result.prediction;
});
