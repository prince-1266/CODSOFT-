const fs = require('fs');
const path = require('path');

function parseCSV(filePath) {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.trim().split('\n');
    const headers = parseCSVLine(lines[0]);
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        const row = {};
        headers.forEach((h, idx) => {
            row[h] = values[idx] || '';
        });
        rows.push(row);
    }
    return rows;
}

function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    result.push(current.trim());
    return result;
}

function loadData(dataDir) {
    const moviesPath = path.join(dataDir, 'movies.csv');
    const ratingsPath = path.join(dataDir, 'ratings.csv');
    const tagsPath = path.join(dataDir, 'tags.csv');

    console.log('📂 Loading movies...');
    const moviesRaw = parseCSV(moviesPath);
    const movies = {};
    moviesRaw.forEach(m => {
        movies[m.movieId] = {
            movieId: parseInt(m.movieId),
            title: m.title,
            genres: m.genres ? m.genres.split('|') : []
        };
    });

    console.log('📂 Loading ratings...');
    const ratingsRaw = parseCSV(ratingsPath);
    const ratings = ratingsRaw.map(r => ({
        userId: parseInt(r.userId),
        movieId: parseInt(r.movieId),
        rating: parseFloat(r.rating)
    }));

    console.log('📂 Loading tags...');
    let tags = [];
    if (fs.existsSync(tagsPath)) {
        const tagsRaw = parseCSV(tagsPath);
        tags = tagsRaw.map(t => ({
            userId: parseInt(t.userId),
            movieId: parseInt(t.movieId),
            tag: t.tag ? t.tag.toLowerCase() : ''
        }));
    }

    // Attach tags to movies
    tags.forEach(t => {
        if (movies[t.movieId]) {
            if (!movies[t.movieId].tags) movies[t.movieId].tags = [];
            movies[t.movieId].tags.push(t.tag);
        }
    });

    // Compute average ratings per movie
    const ratingSum = {};
    const ratingCount = {};
    ratings.forEach(r => {
        ratingSum[r.movieId] = (ratingSum[r.movieId] || 0) + r.rating;
        ratingCount[r.movieId] = (ratingCount[r.movieId] || 0) + 1;
    });
    Object.keys(ratingSum).forEach(id => {
        if (movies[id]) {
            movies[id].avgRating = parseFloat((ratingSum[id] / ratingCount[id]).toFixed(2));
            movies[id].numRatings = ratingCount[id];
        }
    });

    const userIds = [...new Set(ratings.map(r => r.userId))];

    console.log(`✅ Loaded ${Object.keys(movies).length} movies, ${ratings.length} ratings, ${tags.length} tags`);
    console.log(`✅ ${userIds.length} unique users`);

    return { movies, ratings, tags, userIds };
}

module.exports = { loadData };