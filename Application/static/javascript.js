const preview = document.querySelector('#preview');
const image = document.querySelector('#image');
preview.style.display = 'none';

image.addEventListener('change', () => {
    preview.src = '';
    preview.style.display = 'none';
    predictionDiv.innerHTML = '';
    const file = image.files[0];
    const reader = new FileReader();
    reader.addEventListener('load', () => {
        preview.src = reader.result;
        preview.style.display = 'block';
    });
    reader.readAsDataURL(file);
});

const form = document.querySelector('form');
const predictBtn = document.querySelector('#predict-btn');
const predictionDiv = document.querySelector('#prediction');
predictBtn.addEventListener('click', (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    predictionDiv.textContent = '';
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const prediction = data.signature;
        const predictionMsg = `Prediction: ${prediction}`;
        const predictionText = document.createTextNode(predictionMsg);
        predictionDiv.appendChild(predictionText);
    })
    .catch(error => console.error(error));
});
