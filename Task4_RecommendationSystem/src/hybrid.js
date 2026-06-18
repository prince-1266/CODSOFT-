class HybridRecommender {
    constructor(collaborative, contentBased, movies) {
        this.collaborative = collaborative;
        this.contentBased = contentBased;
        this.movies = movies;
    }

    getRecommendations(userId, topN = 10) {
        // Get collaborative recommendations (more of them for re-ranking)
        const collabRecs = this.collaborative.getRecommendations(userId, topN * 3);

        if (collabRecs.length === 0) return [];

        // Get user's top-rated movies for content-based boosting
        const userRated = this.collaborative.getUserRatedMovies(userId);
        const topRated = userRated.filter(m => m.rating >= 4).slice(0, 5);

        // Score each collaborative recommendation with content similarity
        const scored = collabRecs.map(rec => {
            let contentBoost = 0;

            if (topRated.length > 0) {
                // Average content similarity to user's top-rated movies
                const similarities = topRated.map(liked => {
                    const similar = this.contentBased.getSimilarMovies(liked.movieId, 100);
                    const match = similar.find(s => s.movieId === rec.movieId);
                    return match ? match.similarity : 0;
                });
                contentBoost = similarities.reduce((a, b) => a + b, 0) / similarities.length;
            }

            // Hybrid score: 70% collaborative + 30% content-based
            const hybridScore = 0.7 * (rec.predictedRating / 5) + 0.3 * contentBoost;

            return {
                ...rec,
                contentBoost: parseFloat(contentBoost.toFixed(4)),
                hybridScore: parseFloat(hybridScore.toFixed(4)),
                method: 'hybrid'
            };
        });

        scored.sort((a, b) => b.hybridScore - a.hybridScore);
        return scored.slice(0, topN);
    }
}

module.exports = { HybridRecommender };