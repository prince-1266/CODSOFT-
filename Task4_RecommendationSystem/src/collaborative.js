class CollaborativeRecommender {
    constructor(ratings, movies, numFactors = 20, learningRate = 0.005, regularization = 0.02, epochs = 30) {
        this.movies = movies;
        this.numFactors = numFactors;
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.epochs = epochs;
        this.userFactors = {};
        this.itemFactors = {};
        this.userBias = {};
        this.itemBias = {};
        this.globalMean = 0;
        this.trained = false;

        this._prepareData(ratings);
    }

    _prepareData(ratings) {
        this.ratings = ratings;
        this.userIds = [...new Set(ratings.map(r => r.userId))];
        this.movieIds = [...new Set(ratings.map(r => r.movieId))];

        // Build user-movie rating map
        this.userRatings = {};
        ratings.forEach(r => {
            if (!this.userRatings[r.userId]) this.userRatings[r.userId] = {};
            this.userRatings[r.userId][r.movieId] = r.rating;
        });

        this.globalMean = ratings.reduce((s, r) => s + r.rating, 0) / ratings.length;
    }

    train() {
        console.log(`🧠 Training collaborative filtering (${this.epochs} epochs, ${this.numFactors} factors)...`);

        // Initialize random factors
        const initRandom = () => {
            const arr = new Float64Array(this.numFactors);
            for (let i = 0; i < this.numFactors; i++) arr[i] = (Math.random() - 0.5) * 0.1;
            return arr;
        };

        this.userIds.forEach(u => {
            this.userFactors[u] = initRandom();
            this.userBias[u] = 0;
        });
        this.movieIds.forEach(m => {
            this.itemFactors[m] = initRandom();
            this.itemBias[m] = 0;
        });

        // SGD training
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            let totalError = 0;

            // Shuffle ratings
            const shuffled = [...this.ratings].sort(() => Math.random() - 0.5);

            shuffled.forEach(r => {
                const predicted = this._predict(r.userId, r.movieId);
                const error = r.rating - predicted;
                totalError += error * error;

                // Update biases
                this.userBias[r.userId] += this.learningRate * (error - this.regularization * this.userBias[r.userId]);
                this.itemBias[r.movieId] += this.learningRate * (error - this.regularization * this.itemBias[r.movieId]);

                // Update factors
                const uf = this.userFactors[r.userId];
                const mf = this.itemFactors[r.movieId];
                for (let k = 0; k < this.numFactors; k++) {
                    const uOld = uf[k];
                    const mOld = mf[k];
                    uf[k] += this.learningRate * (error * mOld - this.regularization * uOld);
                    mf[k] += this.learningRate * (error * uOld - this.regularization * mOld);
                }
            });

            const rmse = Math.sqrt(totalError / this.ratings.length);
            if ((epoch + 1) % 5 === 0 || epoch === 0) {
                console.log(`   Epoch ${epoch + 1}/${this.epochs} — RMSE: ${rmse.toFixed(4)}`);
            }
        }

        this.trained = true;
        console.log('✅ Training complete!');
    }

    _predict(userId, movieId) {
        const uf = this.userFactors[userId];
        const mf = this.itemFactors[movieId];
        if (!uf || !mf) return this.globalMean;

        let dot = 0;
        for (let k = 0; k < this.numFactors; k++) {
            dot += uf[k] * mf[k];
        }
        let pred = this.globalMean + (this.userBias[userId] || 0) + (this.itemBias[movieId] || 0) + dot;
        return Math.max(0.5, Math.min(5.0, pred)); // Clamp to rating range
    }

    getRecommendations(userId, topN = 10) {
        if (!this.trained) return [];

        const rated = this.userRatings[userId] || {};
        const predictions = [];

        this.movieIds.forEach(movieId => {
            if (rated[movieId]) return; // Skip already rated
            const pred = this._predict(userId, movieId);
            if (this.movies[movieId]) {
                predictions.push({
                    movieId,
                    predictedRating: parseFloat(pred.toFixed(2)),
                    ...this.movies[movieId]
                });
            }
        });

        predictions.sort((a, b) => b.predictedRating - a.predictedRating);
        return predictions.slice(0, topN);
    }

    getUserRatedMovies(userId) {
        const rated = this.userRatings[userId] || {};
        return Object.entries(rated)
            .map(([movieId, rating]) => ({
                movieId: parseInt(movieId),
                rating,
                ...(this.movies[movieId] || {})
            }))
            .sort((a, b) => b.rating - a.rating);
    }

    evaluate() {
        if (!this.trained) return { rmse: 0, mae: 0 };

        let totalSE = 0, totalAE = 0;
        this.ratings.forEach(r => {
            const pred = this._predict(r.userId, r.movieId);
            const err = r.rating - pred;
            totalSE += err * err;
            totalAE += Math.abs(err);
        });

        return {
            rmse: parseFloat(Math.sqrt(totalSE / this.ratings.length).toFixed(4)),
            mae: parseFloat((totalAE / this.ratings.length).toFixed(4))
        };
    }
}

module.exports = { CollaborativeRecommender };