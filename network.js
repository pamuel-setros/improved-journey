// Neural Network Implementation
class NeuralNetwork {
    constructor(inputSize, hiddenLayers = [16, 32], outputSize = 2) {
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers;
        this.outputSize = outputSize;
        this.layers = [];
        this.learningRate = 0.01;
        
        // Initialize network architecture
        const layerSizes = [inputSize, ...hiddenLayers, outputSize];
        
        for (let i = 0; i < layerSizes.length - 1; i++) {
            this.layers.push({
                weights: this.randomMatrix(layerSizes[i + 1], layerSizes[i]),
                biases: this.zeros(layerSizes[i + 1], 1)
            });
        }
    }

    randomMatrix(rows, cols) {
        const matrix = [];
        const scale = Math.sqrt(2.0 / cols);
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = (Math.random() - 0.5) * 2 * scale;
            }
        }
        return matrix;
    }

    zeros(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = Array(cols).fill(0);
        }
        return matrix;
    }

    matMul(a, b) {
        const result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < b.length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    addMatrices(a, b) {
        const result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[i].length; j++) {
                // support broadcasting of b when b has single column (bias)
                const bval = (b[i] && b[i].length === 1) ? b[i][0] : (b[i] ? b[i][j] : 0);
                result[i][j] = a[i][j] + bval;
            }
        }
        return result;
    }

    subtractMatrices(a, b) {
        const result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[i].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    multiplyScalar(matrix, scalar) {
        const result = [];
        for (let i = 0; i < matrix.length; i++) {
            result[i] = [];
            for (let j = 0; j < matrix[i].length; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }
        return result;
    }

    relu(z) {
        const result = [];
        for (let i = 0; i < z.length; i++) {
            result[i] = [];
            for (let j = 0; j < z[i].length; j++) {
                result[i][j] = Math.max(0, z[i][j]);
            }
        }
        return result;
    }

    reluDerivative(z) {
        const result = [];
        for (let i = 0; i < z.length; i++) {
            result[i] = [];
            for (let j = 0; j < z[i].length; j++) {
                result[i][j] = z[i][j] > 0 ? 1 : 0;
            }
        }
        return result;
    }

    softmax(z) {
        // z is expected as [numClasses][numSamples]
        const rows = z.length;
        const cols = z[0].length;
        const result = [];
        for (let i = 0; i < rows; i++) result[i] = Array(cols).fill(0);

        for (let j = 0; j < cols; j++) {
            // compute max across rows for numerical stability
            let maxVal = -Infinity;
            for (let i = 0; i < rows; i++) {
                if (z[i][j] > maxVal) maxVal = z[i][j];
            }

            const exps = Array(rows);
            let sum = 0;
            for (let i = 0; i < rows; i++) {
                exps[i] = Math.exp(z[i][j] - maxVal);
                sum += exps[i];
            }

            for (let i = 0; i < rows; i++) {
                result[i][j] = exps[i] / sum;
            }
        }

        return result;
    }

    forward(input) {
        let activation = input;
        const activations = [activation];
        const zValues = [];

        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            
            // z = W * a + b
            let z = this.matMul(layer.weights, activation);
            z = this.addMatrices(z, layer.biases);
            zValues.push(z);

            // Apply activation function
            if (i < this.layers.length - 1) {
                // ReLU for hidden layers
                activation = this.relu(z);
            } else {
                // Softmax for output layer
                activation = this.softmax(z);
            }

            activations.push(activation);
        }

        return { activations, zValues };
    }

    backward(input, targetOutput, forward) {
        const m = input[0].length;
        const deltas = [];

        // Output layer error
        const lastActivation = forward.activations[forward.activations.length - 1];
        let delta = [];
        for (let i = 0; i < lastActivation.length; i++) {
            delta[i] = [];
            for (let j = 0; j < lastActivation[i].length; j++) {
                delta[i][j] = (lastActivation[i][j] - targetOutput[i][j]);
            }
        }
        deltas.unshift(delta);

        // Backpropagate through hidden layers
        for (let i = this.layers.length - 1; i > 0; i--) {
            const nextLayer = this.layers[i];
            const zValues = forward.zValues[i - 1];
            const activation = forward.activations[i];

            // delta = W^T * delta * ReLU'(z)
            const weightTranspose = [];
            for (let j = 0; j < nextLayer.weights[0].length; j++) {
                weightTranspose[j] = [];
                for (let k = 0; k < nextLayer.weights.length; k++) {
                    weightTranspose[j][k] = nextLayer.weights[k][j];
                }
            }

            delta = this.matMul(weightTranspose, delta);
            const reluDeriv = this.reluDerivative(zValues);
            
            for (let j = 0; j < delta.length; j++) {
                for (let k = 0; k < delta[j].length; k++) {
                    delta[j][k] *= reluDeriv[j][k];
                }
            }

            deltas.unshift(delta);
        }

        // Update weights and biases
        for (let i = 0; i < this.layers.length; i++) {
            const activation = forward.activations[i];
            const delta = deltas[i];

            // dW = (1/m) * delta * a^T
            const dW = this.matMul(delta, this.transpose(activation));
            const dWScaled = this.multiplyScalar(dW, 1 / m);

            // db = (1/m) * sum(delta)
            const db = [];
            for (let j = 0; j < delta.length; j++) {
                let sum = 0;
                for (let k = 0; k < delta[j].length; k++) {
                    sum += delta[j][k];
                }
                db[j] = [sum / m];
            }

            // Update parameters
            this.layers[i].weights = this.subtractMatrices(this.layers[i].weights, this.multiplyScalar(dWScaled, this.learningRate));
            this.layers[i].biases = this.subtractMatrices(this.layers[i].biases, this.multiplyScalar(db, this.learningRate));
        }
    }

    transpose(matrix) {
        const result = [];
        for (let j = 0; j < matrix[0].length; j++) {
            result[j] = [];
            for (let i = 0; i < matrix.length; i++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    predict(input) {
        const result = this.forward(input);
        const output = result.activations[result.activations.length - 1];
        const predictions = [];
        const numSamples = output[0].length;
        for (let s = 0; s < numSamples; s++) {
            const probs = [];
            for (let c = 0; c < output.length; c++) {
                probs.push(output[c][s]);
            }
            const maxIdx = probs.indexOf(Math.max(...probs));
            predictions[s] = maxIdx;
        }

        return { predictions, probabilities: output };
    }

    train(X, y, epochs = 100) {
        const losses = [];
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            const forward = this.forward(X);
            this.backward(X, y, forward);
            
            // Calculate loss
            const predictions = forward.activations[forward.activations.length - 1];
            let loss = 0;
            for (let i = 0; i < predictions.length; i++) {
                for (let j = 0; j < predictions[i].length; j++) {
                    const pred = Math.max(1e-10, Math.min(1 - 1e-10, predictions[i][j]));
                    loss -= y[i][j] * Math.log(pred);
                }
            }
            loss /= X[0].length;
            losses.push(loss);
        }

        return losses;
    }

    accuracy(X, y) {
        const result = this.predict(X);
        const predictions = result.predictions;
        
        let correct = 0;
        const trueLabels = [];
        
        // Convert one-hot to class labels
        for (let i = 0; i < y.length; i++) {
            let maxIdx = 0;
            for (let j = 1; j < y[i].length; j++) {
                if (y[i][j] > y[i][maxIdx]) {
                    maxIdx = j;
                }
            }
            trueLabels.push(maxIdx);
        }

        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] === trueLabels[i]) {
                correct++;
            }
        }

        return (correct / predictions.length) * 100;
    }
}
