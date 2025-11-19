// Main application logic

let network = null;
let currentDataset = null;
let trainingData = null;
let testData = null;
let isTraining = false;
let animationId = null;
let decisionBoundaryProgress = 0;

const elements = {
    datasetSelect: document.getElementById('datasetSelect'),
    moonsControlsGroup: document.getElementById('moonsControlsGroup'),
    moonNoise: document.getElementById('moonNoise'),
    moonNoiseValue: document.getElementById('moonNoiseValue'),
    hiddenLayers: document.getElementById('hiddenLayers'),
    learningRate: document.getElementById('learningRate'),
    epochs: document.getElementById('epochs'),
    trainTestSplit: document.getElementById('trainTestSplit'),
    sampleSize: document.getElementById('sampleSize'),
    clusterSpread: document.getElementById('clusterSpread'),
    clusterSpreadValue: document.getElementById('clusterSpreadValue'),
    trainButton: document.getElementById('trainButton'),
    testButton: document.getElementById('testButton'),
    trainingCanvas: document.getElementById('trainingCanvas'),
    networkCanvas: document.getElementById('networkCanvas'),
    trainAccuracy: document.getElementById('trainAccuracy'),
    testAccuracy: document.getElementById('testAccuracy'),
    trainLoss: document.getElementById('trainLoss'),
    status: document.getElementById('status'),
    epochCount: document.getElementById('epochCount')
};

// Event listeners
elements.datasetSelect.addEventListener('change', handleDatasetChange);
elements.moonNoise.addEventListener('input', handleMoonNoiseChange);
elements.clusterSpread.addEventListener('input', handleClusterSpreadChange);
elements.sampleSize.addEventListener('change', generateAndDisplayDataset);
elements.trainTestSplit.addEventListener('change', generateAndDisplayDataset);
elements.trainButton.addEventListener('click', handleTrain);
elements.testButton.addEventListener('click', handleTest);

function updateMoonNoiseValue() {
    elements.moonNoiseValue.textContent = parseFloat(elements.moonNoise.value).toFixed(2);
}

function resizeCanvasToDisplaySize(canvas) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
}

function drawDecisionBoundaryOverlay(canvas, network, data, labels, colors) {
    // Draw a lower-resolution semi-transparent boundary overlay on the training canvas
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const scale = width / 2;
    const offsetX = width / 2;
    const offsetY = height / 2;
    const resolution = 16;

    // don't clear fully; paint semi-transparent backing
    for (let i = 0; i < width; i += resolution) {
        for (let j = 0; j < height; j += resolution) {
            const x = (i - offsetX) / scale;
            const y = (j - offsetY) / scale;

            const input = DatasetGenerator.toMatrix([[x, y]]);
            const result = network.predict(input);
            const prediction = result.predictions[0];
            const probs = result.probabilities.map(row => row[0]);
            const confidence = probs[prediction] || 0;

            const color = colors[prediction];
            const hex = color.substring(1);
            const r = parseInt(hex.substring(0, 2), 16);
            const g = parseInt(hex.substring(2, 4), 16);
            const b = parseInt(hex.substring(4, 6), 16);

            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.06 + confidence * 0.18})`;
            ctx.fillRect(i, j, resolution, resolution);
        }
    }

    // redraw points on top so they remain visible
    const pointRadius = 6;
    for (let i = 0; i < data.length; i++) {
        const label = labels[i];
        const color = colors[label];
        const x = offsetX + data[i][0] * scale;
        const y = offsetY + data[i][1] * scale;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, pointRadius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

function handleMoonNoiseChange() {
    updateMoonNoiseValue();
    generateAndDisplayDataset();
}

function handleClusterSpreadChange() {
    elements.clusterSpreadValue.textContent = parseFloat(elements.clusterSpread.value).toFixed(3);
    generateAndDisplayDataset();
}

function handleDatasetChange() {
    const selectedDataset = elements.datasetSelect.value;
    
    // Show/hide moon controls
    if (selectedDataset === 'moons') {
        elements.moonsControlsGroup.style.display = 'block';
    } else {
        elements.moonsControlsGroup.style.display = 'none';
    }

    generateAndDisplayDataset();
}

function generateAndDisplayDataset() {
    const selectedDataset = elements.datasetSelect.value;
    const sampleSize = parseInt(elements.sampleSize.value);
    const trainTestSplit = parseInt(elements.trainTestSplit.value) / 100;

    let dataset;

    switch (selectedDataset) {
        case 'diagonals':
            dataset = DatasetGenerator.generateDiagonal3Class(sampleSize, parseFloat(elements.clusterSpread.value));
            break;
        case 'quadrant':
            dataset = DatasetGenerator.generateQuadrantBlobs(sampleSize, parseFloat(elements.clusterSpread.value));
            break;
        case 'moons':
            const moonNoise = parseFloat(elements.moonNoise.value);
            // use moonNoise both for jitter and separation for now, also scale by cluster spread
            const spread = parseFloat(elements.clusterSpread.value);
            dataset = DatasetGenerator.generateMoons(sampleSize, Math.max(0.001, moonNoise * spread * 1.5), moonNoise * 0.6);
            break;
    }

    // Normalize full dataset once (so scaling is consistent) then split using the same shuffled indices
    const normalizedData = DatasetGenerator.normalize(dataset.data);

    // Create shuffled indices and split deterministically here so raw and normalized stay aligned
    const n = dataset.data.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const trainSize = Math.floor(n * trainTestSplit);
    const trainIdx = indices.slice(0, trainSize);
    const testIdx = indices.slice(trainSize);

    const trainDataRaw = trainIdx.map(i => dataset.data[i]);
    const trainLabels = trainIdx.map(i => dataset.labels[i]);
    const testDataRaw = testIdx.map(i => dataset.data[i]);
    const testLabels = testIdx.map(i => dataset.labels[i]);

    const trainDataNorm = trainIdx.map(i => normalizedData[i]);
    const testDataNorm = testIdx.map(i => normalizedData[i]);

    trainingData = {
        X: DatasetGenerator.toMatrix(trainDataNorm),
        y: DatasetGenerator.labelsToOneHot(trainLabels, dataset.numClasses),
        rawData: trainDataRaw,
        rawLabels: trainLabels,
        normalizedData: trainDataNorm
    };

    testData = {
        X: DatasetGenerator.toMatrix(testDataNorm),
        y: DatasetGenerator.labelsToOneHot(testLabels, dataset.numClasses),
        rawData: testDataRaw,
        rawLabels: testLabels,
        normalizedData: testDataNorm
    };

    currentDataset = {
        ...dataset,
        trainNormalized: trainDataNorm,
        testNormalized: testDataNorm
    };

    // Ensure canvas sizes match displayed size
    resizeCanvasToDisplaySize(elements.trainingCanvas);

    // Draw training data (initial). Decision boundary will be drawn on same canvas when training.
    drawDataset(elements.trainingCanvas, currentDataset.trainNormalized, trainingData.rawLabels, currentDataset.colors);

    // Reset stats
    elements.trainAccuracy.textContent = '-';
    elements.testAccuracy.textContent = '-';
    elements.trainLoss.textContent = '-';
    elements.status.textContent = 'Dataset loaded. Ready to train.';
    
    // Draw initial network architecture visualization
    drawNetworkArchitecture(elements.networkCanvas, parseHiddenLayers(), currentDataset.numClasses);
}

function parseHiddenLayers() {
    const input = elements.hiddenLayers.value.trim();
    
    if (!input) {
        return [];
    }

    return input.split(',').map(val => {
        const num = parseInt(val.trim());
        return isNaN(num) || num < 1 ? 16 : num;
    });
}

async function handleTrain() {
    if (!trainingData) {
        alert('Please generate a dataset first.');
        return;
    }

    if (isTraining) {
        alert('Already training...');
        return;
    }

    isTraining = true;
    elements.trainButton.disabled = true;
    elements.testButton.disabled = true;
    elements.status.textContent = 'Training...';

    try {
        const hiddenLayers = parseHiddenLayers();
        const learningRate = parseFloat(elements.learningRate.value);
        const epochs = parseInt(elements.epochs.value);

        // Initialize network
        network = new NeuralNetwork(2, hiddenLayers, currentDataset.numClasses);
        network.learningRate = learningRate;

        // Train network with animated visualization
        await trainNetworkAnimated(network, trainingData, epochs);

        // Calculate accuracy
        const trainAcc = network.accuracy(trainingData.X, trainingData.y);
        elements.trainAccuracy.textContent = trainAcc.toFixed(2) + '%';

        elements.status.textContent = 'Training complete. Click "Test on Test Set" to evaluate.';
    } catch (error) {
        console.error('Training error:', error);
        elements.status.textContent = 'Error during training: ' + error.message;
    }

    isTraining = false;
    elements.trainButton.disabled = false;
    elements.testButton.disabled = false;
}

function trainNetworkAnimated(network, data, epochs) {
    return new Promise((resolve) => {
        let epoch = 0;
        let totalLoss = 0;

        function trainStep() {
            if (epoch < epochs) {
                const forward = network.forward(data.X);
                network.backward(data.X, data.y, forward);

                // Calculate loss
                const predictions = forward.activations[forward.activations.length - 1];
                let loss = 0;
                for (let i = 0; i < predictions.length; i++) {
                    for (let j = 0; j < predictions[i].length; j++) {
                        const pred = Math.max(1e-10, Math.min(1 - 1e-10, predictions[i][j]));
                        loss -= data.y[i][j] * Math.log(pred);
                    }
                }
                loss /= data.X[0].length;
                totalLoss = loss;

                epoch++;

                // Update UI every epoch for live animation
                elements.status.textContent = `Training... Epoch ${epoch}/${epochs}`;
                elements.trainLoss.textContent = totalLoss.toFixed(4);

                // Draw decision boundary and overlay on training canvas every epoch
                drawDecisionBoundaryAnimated(elements.trainingCanvas, network, currentDataset.trainNormalized, trainingData.rawLabels, currentDataset.colors);
                drawNetworkArchitecture(elements.networkCanvas, parseHiddenLayers(), currentDataset.numClasses);

                // Yield to browser to render
                requestAnimationFrame(trainStep);
            } else {
                resolve();
            }
        }

        trainStep();
    });
}

function handleTest() {
    if (!network || !testData) {
        alert('Please train the network first.');
        return;
    }

    const testAcc = network.accuracy(testData.X, testData.y);
    elements.testAccuracy.textContent = testAcc.toFixed(2) + '%';
    elements.status.textContent = 'Test set evaluation complete.';
}

function drawDataset(canvas, data, labels, colors) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const scale = width / 2;
    const offsetX = width / 2;
    const offsetY = height / 2;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    drawCanvasGrid(ctx, canvas);

    // Draw points
    const pointRadius = 6;
    for (let i = 0; i < data.length; i++) {
        const label = labels[i];
        const color = colors[label];
        
        const x = offsetX + data[i][0] * scale;
        const y = offsetY + data[i][1] * scale;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, pointRadius, 0, 2 * Math.PI);
        ctx.fill();

        // White border
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(offsetX, 0);
    ctx.lineTo(offsetX, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, offsetY);
    ctx.lineTo(width, offsetY);
    ctx.stroke();
}

function drawCanvasGrid(ctx, canvas) {
    const width = canvas.width;
    const height = canvas.height;
    const scale = width / 2;
    const offsetX = width / 2;
    const offsetY = height / 2;

    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = -1; i <= 1; i += 0.2) {
        const x = offsetX + i * scale;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        const y = offsetY + i * scale;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
}

function drawDecisionBoundaryAnimated(canvas, network, data, labels, colors) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const scale = width / 2;
    const offsetX = width / 2;
    const offsetY = height / 2;
    const resolution = 10;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Compute predictions on a coarse grid and paint gradient
    const cols = Math.ceil(width / resolution);
    const rows = Math.ceil(height / resolution);
    const preds = new Array(cols);
    for (let ci = 0; ci < cols; ci++) preds[ci] = new Array(rows);

    for (let ci = 0; ci < cols; ci++) {
        for (let r = 0; r < rows; r++) {
            const i = ci * resolution + Math.floor(resolution / 2);
            const j = r * resolution + Math.floor(resolution / 2);
            const x = (i - offsetX) / scale;
            const y = (j - offsetY) / scale;

            const input = DatasetGenerator.toMatrix([[x, y]]);
            const result = network.predict(input);
            const prediction = result.predictions[0];
            const probs = result.probabilities.map(row => row[0]);
            const confidence = probs[prediction] || 0;

            // fill the rectangle
            const color = colors[prediction];
            const hex = color.substring(1);
            const rcol = parseInt(hex.substring(0, 2), 16);
            const gcol = parseInt(hex.substring(2, 4), 16);
            const bcol = parseInt(hex.substring(4, 6), 16);
            ctx.fillStyle = `rgba(${rcol}, ${gcol}, ${bcol}, ${0.18 + confidence * 0.6})`;
            ctx.fillRect(ci * resolution, r * resolution, resolution, resolution);

            preds[ci][r] = prediction;
        }
    }

    // Draw boundary lines where adjacent cells have different predicted class
    ctx.strokeStyle = 'rgba(40,40,40,0.9)';
    ctx.lineWidth = 1.5;
    for (let ci = 0; ci < cols; ci++) {
        for (let r = 0; r < rows; r++) {
            const p = preds[ci][r];
            // check right neighbor
            if (ci + 1 < cols && preds[ci + 1][r] !== p) {
                const x = (ci + 1) * resolution;
                const y1 = r * resolution;
                const y2 = (r + 1) * resolution;
                ctx.beginPath();
                ctx.moveTo(x + 0.5, y1);
                ctx.lineTo(x + 0.5, y2);
                ctx.stroke();
            }
            // check bottom neighbor
            if (r + 1 < rows && preds[ci][r + 1] !== p) {
                const y = (r + 1) * resolution;
                const x1 = ci * resolution;
                const x2 = (ci + 1) * resolution;
                ctx.beginPath();
                ctx.moveTo(x1, y + 0.5);
                ctx.lineTo(x2, y + 0.5);
                ctx.stroke();
            }
        }
    }

    // Draw grid (subtle)
    drawCanvasGrid(ctx, canvas);

    // Draw data points on top
    const pointRadius = 5;
    for (let i = 0; i < data.length; i++) {
        const label = labels[i];
        const color = colors[label];

        const x = offsetX + data[i][0] * scale;
        const y = offsetY + data[i][1] * scale;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, pointRadius, 0, 2 * Math.PI);
        ctx.fill();

        // White border
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(offsetX, 0);
    ctx.lineTo(offsetX, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, offsetY);
    ctx.lineTo(width, offsetY);
    ctx.stroke();
}

function drawNetworkArchitecture(canvas, hiddenLayers, outputClasses) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    const layers = [2, ...hiddenLayers, outputClasses];
    const numLayers = layers.length;
    const layerWidth = width / (numLayers + 1);
    const padding = 40;
    const nodeArea = height - 2 * padding;

    // Draw layers
    for (let l = 0; l < numLayers; l++) {
        const numNodes = layers[l];
        const x = layerWidth * (l + 1);
        const maxNodeHeight = nodeArea / numNodes;
        const nodeRadius = Math.min(20, maxNodeHeight / 2.5);
        const spacing = nodeArea / (numNodes + 1);

        // Draw connections to next layer
        if (l < numLayers - 1) {
            ctx.strokeStyle = 'rgba(102, 126, 234, 0.2)';
            ctx.lineWidth = 1;
            const nextNumNodes = layers[l + 1];
            const nextX = layerWidth * (l + 2);
            const nextSpacing = nodeArea / (nextNumNodes + 1);

            for (let i = 0; i < numNodes; i++) {
                const y = padding + spacing * (i + 1);
                for (let j = 0; j < nextNumNodes; j++) {
                    const nextY = padding + nextSpacing * (j + 1);
                    ctx.beginPath();
                    ctx.moveTo(x + nodeRadius, y);
                    ctx.lineTo(nextX - nodeRadius, nextY);
                    ctx.stroke();
                }
            }
        }

        // Draw nodes
        for (let i = 0; i < numNodes; i++) {
            const y = padding + spacing * (i + 1);

            // Node circle
            ctx.fillStyle = '#667eea';
            ctx.beginPath();
            ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
            ctx.fill();

            // Node border
            ctx.strokeStyle = '#764ba2';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Node label
            ctx.fillStyle = 'white';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(i + 1, x, y);
        }

        // Layer label
        ctx.fillStyle = '#333';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        
        let layerName = '';
        if (l === 0) layerName = 'Input';
        else if (l === numLayers - 1) layerName = 'Output';
        else layerName = 'Hidden ' + l;

        ctx.fillText(layerName, x, padding - 20);
    }
}

// Initialize on load
window.addEventListener('load', () => {
    updateMoonNoiseValue();
    elements.clusterSpreadValue.textContent = parseFloat(elements.clusterSpread.value).toFixed(3);
    generateAndDisplayDataset();
});
