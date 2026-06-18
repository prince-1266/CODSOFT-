class ContentBasedRecommender {
    constructor(movies) {
        this.movies = movies;
        this.movieIds = Object.keys(movies).map(Number);
        this.tfidfVectors = {};
        this.vocabulary = {};
        this._buildTFIDF();
    }

    _buildTFIDF() {
        console.log('🔨 Building TF-IDF vectors...');

        // Build document for each movie: genres + tags
        const docs = {};
        this.movieIds.forEach(id => {
            const movie = this.movies[id];
            const terms = [
                ...movie.genres.map(g => g.toLowerCase().replace(/[^a-z]/g, '')),
                ...(movie.tags || []).map(t => t.toLowerCase().replace(/[^a-z\s]/g, '').trim())
            ].filter(t => t.length > 0);
            docs[id] = terms;
        });

        // Build vocabulary
        const vocab = {};
        let vocabIdx = 0;
        Object.values(docs).forEach(terms => {
            terms.forEach(t => {
                if (!(t in vocab)) {
                    vocab[t] = vocabIdx++;
                }
            });
        });
        this.vocabulary = vocab;
        const vocabSize = Object.keys(vocab).length;

        // Compute TF (term frequency) for each movie
        const tf = {};
        Object.entries(docs).forEach(([id, terms]) => {
            tf[id] = new Float64Array(vocabSize);
            const counts = {};
            terms.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
            Object.entries(counts).forEach(([term, count]) => {
                tf[id][vocab[term]] = count / terms.length;
            });
        });

        // Compute IDF (inverse document frequency)
        const numDocs = this.movieIds.length;
        const df = new Float64Array(vocabSize);
        Object.values(tf).forEach(vec => {
            for (let i = 0; i < vocabSize; i++) {
                if (vec[i] > 0) df[i]++;
            }
        });

        const idf = new Float64Array(vocabSize);
        for (let i = 0; i < vocabSize; i++) {
            idf[i] = df[i] > 0 ? Math.log(numDocs / df[i]) : 0;
        }

        // Compute TF-IDF = TF * IDF
        this.movieIds.forEach(id => {
            const vec = new Float64Array(vocabSize);
            for (let i = 0; i < vocabSize; i++) {
                vec[i] = tf[id][i] * idf[i];
            }
            this.tfidfVectors[id] = vec;
        });

        console.log(`✅ TF-IDF built: ${vocabSize} terms, ${this.movieIds.length} movies`);
    }

    _cosineSimilarity(vecA, vecB) {
        let dot = 0, magA = 0, magB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
            magA += vecA[i] * vecA[i];
            magB += vecB[i] * vecB[i];
        }
        const denom = Math.sqrt(magA) * Math.sqrt(magB);
        return denom === 0 ? 0 : dot / denom;
    }

    getSimilarMovies(movieId, topN = 10) {
        const targetVec = this.tfidfVectors[movieId];
        if (!targetVec) return [];

        const scores = [];
        this.movieIds.forEach(id => {
            if (id === movieId) return;
            const sim = this._cosineSimilarity(targetVec, this.tfidfVectors[id]);
            if (sim > 0) {
                scores.push({ movieId: id, similarity: parseFloat(sim.toFixed(4)), ...this.movies[id] });
            }
        });

        scores.sort((a, b) => b.similarity - a.similarity);
        return scores.slice(0, topN);
    }

    searchMovies(query, limit = 20) {
        const q = query.toLowerCase();
        return this.movieIds
            .filter(id => this.movies[id].title.toLowerCase().includes(q))
            .slice(0, limit)
            .map(id => this.movies[id]);
    }
}

module.exports = { ContentBasedRecommender};