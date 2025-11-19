// Dataset generation functions

class DatasetGenerator {
    static generateDiagonal3Class(samplesPerClass = 100, noise = 0.03) {
        // Generate 3 diagonal bands across the square, one color per diagonal
        const data = [];
        const labels = [];
        const colors = ['#FF3B30', '#5A8BF6', '#4CD964']; // Red, Blue, Green

        // diagonals centered along lines y = x + offset
        const offsets = [-0.7, 0.0, 0.7];

        for (let classIdx = 0; classIdx < 3; classIdx++) {
            const offset = offsets[classIdx];
            for (let i = 0; i < samplesPerClass; i++) {
                // sample along x and set y = x + offset with small jitter
                const x = (Math.random() * 2 - 1);
                const y = x + offset + (Math.random() - 0.5) * noise;

                data.push([
                    Math.max(-1, Math.min(1, x)),
                    Math.max(-1, Math.min(1, y))
                ]);
                labels.push(classIdx);
            }
        }

        return { data, labels, colors, numClasses: 3 };
    }

    static generateQuadrantBlobs(samplesPerClass = 100, noise = 0.02) {
        const data = [];
        const labels = [];
        // explicit quadrant colors: red, red, green, blue (two reds)
        const colors = ['#FF3B30', '#FF3B30', '#4CD964', '#5A8BF6'];
        const centers = [
            [-0.7, -0.7],  // Bottom-left
            [0.7, -0.7],   // Bottom-right
            [-0.7, 0.7],   // Top-left
            [0.7, 0.7]     // Top-right
        ];

        for (let classIdx = 0; classIdx < 4; classIdx++) {
            const [centerX, centerY] = centers[classIdx];
            for (let i = 0; i < samplesPerClass; i++) {
                const x = centerX + (Math.random() - 0.5) * noise + (Math.random() - 0.5) * 0.06;
                const y = centerY + (Math.random() - 0.5) * noise + (Math.random() - 0.5) * 0.06;
                data.push([
                    Math.max(-1, Math.min(1, x)),
                    Math.max(-1, Math.min(1, y))
                ]);
                labels.push(classIdx);
            }
        }

        return { data, labels, colors, numClasses: 4 };
    }

    static generateMoons(samplesPerClass = 100, noise = 0.08, separation = 0.1) {
        // Classic two moons in red and green. `separation` pushes the moons closer/further.
        const data = [];
        const labels = [];
        const colors = ['#FF3B30', '#4CD964']; // Red, Green

        const radius = 0.6;
        for (let i = 0; i < samplesPerClass; i++) {
            const t = Math.random() * Math.PI;
            // upper moon
            let x1 = Math.cos(t) * radius;
            let y1 = Math.sin(t) * radius;
            x1 += (Math.random() - 0.5) * noise;
            y1 += (Math.random() - 0.5) * noise;
            data.push([x1 - 0.2, y1 + separation]);
            labels.push(0);

            // lower moon
            let x2 = 1 - Math.cos(t) * radius;
            let y2 = -Math.sin(t) * radius;
            x2 += (Math.random() - 0.5) * noise;
            y2 += (Math.random() - 0.5) * noise;
            data.push([x2 - 0.8, y2 - separation]);
            labels.push(1);
        }

        return { data, labels, colors, numClasses: 2 };
    }

    static trainTestSplit(data, labels, trainRatio = 0.8) {
        const n = data.length;
        const trainSize = Math.floor(n * trainRatio);
        
        // Shuffle indices
        const indices = Array.from({ length: n }, (_, i) => i);
        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const trainData = [];
        const trainLabels = [];
        const testData = [];
        const testLabels = [];

        for (let i = 0; i < n; i++) {
            const idx = indices[i];
            if (i < trainSize) {
                trainData.push(data[idx]);
                trainLabels.push(labels[idx]);
            } else {
                testData.push(data[idx]);
                testLabels.push(labels[idx]);
            }
        }

        return { trainData, trainLabels, testData, testLabels };
    }

    static toMatrix(data) {
        const matrix = [];
        for (let j = 0; j < data[0].length; j++) {
            matrix[j] = [];
            for (let i = 0; i < data.length; i++) {
                matrix[j][i] = data[i][j];
            }
        }
        return matrix;
    }

    static labelsToOneHot(labels, numClasses) {
        const matrix = [];
        for (let i = 0; i < numClasses; i++) {
            matrix[i] = [];
            for (let j = 0; j < labels.length; j++) {
                matrix[i][j] = labels[j] === i ? 1 : 0;
            }
        }
        return matrix;
    }

    static normalize(data) {
        const normalized = data.map(point => [...point]);
        
        // Find min and max
        const dims = data[0].length;
        const mins = Array(dims).fill(Infinity);
        const maxs = Array(dims).fill(-Infinity);

        for (let i = 0; i < data.length; i++) {
            for (let j = 0; j < dims; j++) {
                mins[j] = Math.min(mins[j], data[i][j]);
                maxs[j] = Math.max(maxs[j], data[i][j]);
            }
        }

        // Normalize to [-1, 1]
        for (let i = 0; i < normalized.length; i++) {
            for (let j = 0; j < dims; j++) {
                const range = maxs[j] - mins[j];
                if (range > 0) {
                    normalized[i][j] = 2 * (data[i][j] - mins[j]) / range - 1;
                }
            }
        }

        return normalized;
    }
}
