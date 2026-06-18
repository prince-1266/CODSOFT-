const express = require('express');
const path = require('path');
const { loadData } = require('./src/dataLoader');
const { ContentBasedRecommender } = require('./src/contentBased');
const { CollaborativeRecommender } = require('./src/collaborative');
const { HybridRecommender } = require('./src/hybrid');

const app = express();
const PORT = 3000;

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Load data and initialize recommenders
console.log('\n🚀 Initializing Recommendation System...\n');

const dataDir = path.join(__dirname, 'data');
const { movies, ratings, userIds } = loadData(dataDir);

const contentBased = new ContentBasedRecommender(movies);
const collaborative = new CollaborativeRecommender(ratings, movies);
collaborative.train();
const hybrid = new HybridRecommender(collaborative, contentBased, movies);
const metrics = collaborative.evaluate();

console.log(`\n📊 Model Evaluation — RMSE: ${metrics.rmse}, MAE: ${metrics.mae}`);
console.log(`\n✅ Server ready! Open http://localhost:${PORT}\n`);

// API Routes

// Search movies
app.get('/api/search', (req, res) => {
    try {
        const query = req.query.q || '';
        const results = contentBased.searchMovies(query);
        res.json(results);
    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({ error: 'Internal server error during search' });
    }
});

// Content-based: similar movies
app.get('/api/similar/:movieId', (req, res) => {
    try {
        const movieId = parseInt(req.params.movieId);
        if (isNaN(movieId) || !movies[movieId]) {
            return res.status(404).json({ error: 'Movie not found or invalid ID' });
        }
        const topN = parseInt(req.query.n) || 10;
        const results = contentBased.getSimilarMovies(movieId, topN);
        const movie = movies[movieId];
        res.json({ movie, recommendations: results });
    } catch (error) {
        console.error('Similar movies error:', error);
        res.status(500).json({ error: 'Internal server error while fetching similar movies' });
    }
});

// Collaborative: user recommendations
app.get('/api/recommend/:userId', (req, res) => {
    try {
        const userId = parseInt(req.params.userId);
        if (isNaN(userId) || !userIds.includes(userId)) {
            return res.status(404).json({ error: 'User not found or invalid ID' });
        }
        const topN = parseInt(req.query.n) || 10;
        const recommendations = collaborative.getRecommendations(userId, topN);
        const ratedMovies = collaborative.getUserRatedMovies(userId).slice(0, 10);
        res.json({ userId, ratedMovies, recommendations });
    } catch (error) {
        console.error('Recommend error:', error);
        res.status(500).json({ error: 'Internal server error while generating recommendations' });
    }
});

// Hybrid recommendations
app.get('/api/hybrid/:userId', (req, res) => {
    try {
        const userId = parseInt(req.params.userId);
        if (isNaN(userId) || !userIds.includes(userId)) {
            return res.status(404).json({ error: 'User not found or invalid ID' });
        }
        const topN = parseInt(req.query.n) || 10;
        const recommendations = hybrid.getRecommendations(userId, topN);
        const ratedMovies = collaborative.getUserRatedMovies(userId).slice(0, 10);
        res.json({ userId, ratedMovies, recommendations });
    } catch (error) {
        console.error('Hybrid recommend error:', error);
        res.status(500).json({ error: 'Internal server error while generating hybrid recommendations' });
    }
});

// Get available user IDs
app.get('/api/users', (req, res) => {
    try {
        res.json({ users: userIds.slice(0, 50), total: userIds.length });
    } catch (error) {
        console.error('Users fetch error:', error);
        res.status(500).json({ error: 'Internal server error while fetching users' });
    }
});

// Get metrics
app.get('/api/metrics', (req, res) => {
    try {
        res.json({
            rmse: metrics.rmse,
            mae: metrics.mae,
            totalMovies: Object.keys(movies).length,
            totalRatings: ratings.length,
            totalUsers: userIds.length
        });
    } catch (error) {
        console.error('Metrics fetch error:', error);
        res.status(500).json({ error: 'Internal server error while fetching metrics' });
    }
});

app.listen(PORT, () => {
    console.log(`🌐 Server running at http://localhost:${PORT}`);
});