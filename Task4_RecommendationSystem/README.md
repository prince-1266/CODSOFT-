# Task 4 - Movie Recommendation System

A movie recommendation system that suggests films to users based on their preferences. Built using collaborative filtering, content-based filtering, and a hybrid approach. The system is trained on a dataset of 10,000 movies (5,000 English + 5,000 Hindi) and 126,980 user ratings.

---

## How to Run

**Step 1:** Install dependencies
```bash
npm install
```

**Step 2:** Start the server
```bash
node server.js
```

**Step 3:** Open your browser
```
http://localhost:3000
```

The server will load data, build TF-IDF vectors, and train the collaborative filtering model. This takes about 10-15 seconds. Once you see "Server ready!" in the terminal, the app is ready to use.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Node.js | Server-side runtime |
| Express.js | Web framework and API server |
| HTML5 | Frontend page structure |
| CSS3 | Frontend styling |
| JavaScript (Vanilla) | Frontend logic and API calls |
| TF-IDF Algorithm | Content-based feature extraction |
| Cosine Similarity | Measuring movie similarity |
| Matrix Factorization (SVD) | Collaborative filtering model |
| Stochastic Gradient Descent | Model training optimization |
| MovieLens Dataset | Source of English movie ratings data |

---

## Folder Structure

```
Task4_RecommendationSystem/
|
|-- data/
|   |-- movies.csv              # 10,000 movies (title, genres)
|   |-- ratings.csv             # 126,980 user ratings (userId, movieId, rating)
|   |-- tags.csv                # User-generated tags for movies
|
|-- public/
|   |-- index.html              # Frontend HTML (tabs, search, cards)
|   |-- style.css               # Frontend CSS (layout, colors, responsive)
|   |-- script.js               # Frontend JS (API calls, rendering)
|
|-- src/
|   |-- dataLoader.js           # Reads and parses CSV data files
|   |-- contentBased.js         # TF-IDF + Cosine Similarity engine
|   |-- collaborative.js        # Matrix Factorization with SGD training
|   |-- hybrid.js               # Combines collaborative + content-based
|
|-- server.js                   # Express API server (entry point)
|-- package.json                # Project dependencies
|-- README.md                   # This file
```

---

## Workflow

### Overall Application Flow

```
[Start: node server.js]
        |
        v
[dataLoader.js] ---> Load movies.csv, ratings.csv, tags.csv
        |
        v
[contentBased.js] ---> Build TF-IDF vectors for all 10,000 movies
        |
        v
[collaborative.js] ---> Train Matrix Factorization model (30 epochs of SGD)
        |
        v
[hybrid.js] ---> Initialize hybrid engine (connects both models)
        |
        v
[server.js] ---> Start Express server on port 3000
        |
        v
[public/index.html] ---> User opens browser, interacts with UI
        |
        v
[public/script.js] ---> Sends API requests to server
        |
        v
[server.js API routes] ---> Calls appropriate recommender, returns JSON
        |
        v
[public/script.js] ---> Renders movie cards in the browser
```

### Content-Based Recommendation Flow

```
User searches "Sholay" --> clicks on the result
        |
        v
Frontend calls GET /api/similar/300003
        |
        v
server.js calls contentBased.getSimilarMovies(300003, 10)
        |
        v
ContentBasedRecommender:
  1. Gets TF-IDF vector for "Sholay" (genres: Action|Adventure|Comedy)
  2. Compares it against all other 9,999 movie vectors
  3. Calculates cosine similarity for each pair
  4. Sorts by highest similarity
  5. Returns top 10 most similar movies
        |
        v
Frontend displays similar movie cards with similarity scores
```

### Collaborative Filtering Flow

```
User selects "User #5" --> clicks "Get Recommendations"
        |
        v
Frontend calls GET /api/recommend/5
        |
        v
server.js calls collaborative.getRecommendations(5, 10)
        |
        v
CollaborativeRecommender:
  1. Gets list of movies User #5 has already rated
  2. For every movie User #5 has NOT rated:
     a. Computes: prediction = globalMean + userBias + movieBias + dot(userVector, movieVector)
     b. Clamps prediction to range [0.5, 5.0]
  3. Sorts all predictions from highest to lowest
  4. Returns top 10 highest-predicted movies
        |
        v
server.js also calls collaborative.getUserRatedMovies(5) for display
        |
        v
Frontend displays rated movies + recommended movies with predicted ratings
```

### Hybrid Recommendation Flow

```
User selects "User #5" --> clicks "Get Recommendations" (Hybrid tab)
        |
        v
Frontend calls GET /api/hybrid/5
        |
        v
server.js calls hybrid.getRecommendations(5, 10)
        |
        v
HybridRecommender:
  1. Gets top 30 collaborative recommendations for User #5
  2. Gets User #5's top-rated movies (rating >= 4 stars)
  3. For each collaborative recommendation:
     a. Measures content similarity to user's favorite movies
     b. Calculates: hybridScore = 0.7 * (collaborativeScore) + 0.3 * (contentScore)
  4. Sorts by hybrid score
  5. Returns top 10
        |
        v
Frontend displays movie cards with hybrid score, predicted rating, and content boost
```

---

## File-by-File Explanation

### server.js (Entry Point)

The main file that initializes everything and defines API routes.

| Section | What It Does |
|---|---|
| Lines 1-6 | Imports Express, path, and all recommender modules |
| Lines 8-12 | Creates Express app, sets up static file serving and JSON parsing |
| Lines 14-27 | Loads data, initializes all 3 recommenders, trains the model, prints metrics |
| Lines 29-36 | `GET /api/search?q=` — Searches movies by title using content-based module |
| Lines 38-45 | `GET /api/similar/:movieId` — Returns similar movies using content-based filtering |
| Lines 47-54 | `GET /api/recommend/:userId` — Returns recommendations using collaborative filtering |
| Lines 56-63 | `GET /api/hybrid/:userId` — Returns recommendations using hybrid approach |
| Lines 65-68 | `GET /api/users` — Returns list of available user IDs |
| Lines 70-79 | `GET /api/metrics` — Returns model evaluation metrics (RMSE, MAE, counts) |
| Lines 81-83 | Starts the server on port 3000 |

---

### src/dataLoader.js

Reads CSV files and structures the data for the recommenders.

| Function | Parameters | What It Does |
|---|---|---|
| `parseCSVLine(line)` | `line` (string) | Parses a single CSV line, handling quoted fields with commas correctly |
| `parseCSV(filePath)` | `filePath` (string) | Reads an entire CSV file, splits into rows, uses parseCSVLine on each row, returns array of objects |
| `loadData(dataDir)` | `dataDir` (string) | Main function. Loads movies.csv, ratings.csv, tags.csv. Attaches tags to movies. Calculates average rating and rating count per movie. Returns `{ movies, ratings, tags, userIds }` |

**Returns:**
- `movies` — Object mapping movieId to `{ movieId, title, genres[], tags[], avgRating, numRatings }`
- `ratings` — Array of `{ userId, movieId, rating }`
- `userIds` — Array of unique user IDs

---

### src/contentBased.js (ContentBasedRecommender)

Finds similar movies based on their genre and tag content.

| Method | Parameters | What It Does |
|---|---|---|
| `constructor(movies)` | `movies` (object) | Stores movie data, extracts all movie IDs, calls `_buildTFIDF()` |
| `_buildTFIDF()` | none | Builds TF-IDF vectors for all movies. Step 1: Creates a "document" per movie from genres + tags. Step 2: Builds vocabulary of all unique terms. Step 3: Calculates Term Frequency (how often each term appears in a movie). Step 4: Calculates Inverse Document Frequency (rare terms get higher weight). Step 5: Multiplies TF * IDF to get final vector for each movie. |
| `_cosineSimilarity(vecA, vecB)` | two Float64Arrays | Calculates the cosine of the angle between two vectors. Returns a value between 0 (completely different) and 1 (identical). Formula: dot(A,B) / (magnitude(A) * magnitude(B)) |
| `getSimilarMovies(movieId, topN)` | `movieId` (int), `topN` (int, default 10) | Gets the TF-IDF vector for the target movie, compares it against all other movies using cosine similarity, sorts by similarity score, returns top N results |
| `searchMovies(query, limit)` | `query` (string), `limit` (int, default 20) | Filters movies whose title contains the search query (case-insensitive), returns up to `limit` results |

---

### src/collaborative.js (CollaborativeRecommender)

Learns user preferences from rating history using Matrix Factorization.

| Method | Parameters | What It Does |
|---|---|---|
| `constructor(ratings, movies, numFactors, learningRate, regularization, epochs)` | ratings array, movies object, hyperparameters | Stores configuration. Default: 20 factors, 0.005 learning rate, 0.02 regularization, 30 epochs. Calls `_prepareData()`. |
| `_prepareData(ratings)` | `ratings` (array) | Extracts unique user/movie IDs. Builds a lookup map: `userRatings[userId][movieId] = rating`. Calculates global mean rating. |
| `train()` | none | The core ML training loop. Step 1: Creates random 20-dimensional vectors for every user and every movie. Step 2: Runs 30 epochs of SGD — for each rating, predicts the rating, calculates error, updates user/movie vectors and biases using gradient descent with L2 regularization. Prints RMSE every 5 epochs. |
| `_predict(userId, movieId)` | `userId` (int), `movieId` (int) | Predicts rating using: `globalMean + userBias + movieBias + dot(userFactors, movieFactors)`. Clamps result to [0.5, 5.0]. |
| `getRecommendations(userId, topN)` | `userId` (int), `topN` (int, default 10) | Predicts ratings for all movies the user has NOT rated. Sorts by predicted rating. Returns top N movies with predicted scores. |
| `getUserRatedMovies(userId)` | `userId` (int) | Returns all movies the user has rated, sorted by rating (highest first). |
| `evaluate()` | none | Computes RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) across all training ratings. Used to measure model accuracy. |

**Hyperparameters explained:**
- `numFactors = 20` — Each user/movie is represented by a 20-dimensional vector
- `learningRate = 0.005` — How much to adjust weights per step (small = stable training)
- `regularization = 0.02` — Penalty to prevent overfitting
- `epochs = 30` — Number of full passes over the training data

---

### src/hybrid.js (HybridRecommender)

Combines both recommenders for better results.

| Method | Parameters | What It Does |
|---|---|---|
| `constructor(collaborative, contentBased, movies)` | three objects | Stores references to both recommender instances and the movies data |
| `getRecommendations(userId, topN)` | `userId` (int), `topN` (int, default 10) | Step 1: Gets 3x more collaborative recommendations than needed. Step 2: Finds user's top-rated movies (4+ stars). Step 3: For each candidate, calculates average content similarity to user's favorites. Step 4: Computes hybrid score = 0.7 * collaborative + 0.3 * content. Step 5: Sorts by hybrid score, returns top N. |

---

### public/index.html

| Section | What It Contains |
|---|---|
| Header | Page title "Movie Recommendation System" |
| Metrics bar | Displays total movies, users, and ratings count |
| Tab buttons | 3 tabs: Search Movies, User Recommendations, Hybrid System |
| Search panel | Text input for movie search + results list + similar movies grid |
| Collaborative panel | User dropdown + recommend button + rated movies list + recommendations grid |
| Hybrid panel | User dropdown + recommend button + rated movies list + recommendations grid |
| Footer | "CodSoft Internship - Task 4" credit |

---

### public/style.css

| Section | What It Styles |
|---|---|
| Root variables | Primary color (#007bff), background, text, border colors |
| Body/container | Max-width layout, system font stack, light grey background |
| Navbar | Blue header bar with white text |
| Metrics bar | Horizontal stats display with white background |
| Tabs | Button-style tab switcher with active state |
| Cards | White background panels with border and subtle shadow |
| Search items | Clickable list items with hover highlight |
| Movie grid | Responsive CSS Grid layout for movie cards |
| Movie cards | Individual movie display with title, genres, and stats |
| Rated list | Horizontal scrollable list of previously rated movies |
| Loading overlay | Centered spinner overlay during API calls |

---

### public/script.js

| Function/Section | What It Does |
|---|---|
| Tab switching | Adds click listeners to tab buttons, toggles active panel |
| `showLoading()` / `hideLoading()` | Shows/hides the loading overlay |
| `debounce(fn, delay)` | Delays function execution by 300ms to avoid flooding API during typing |
| Metrics loader | Calls `GET /api/metrics` on page load, populates the metrics bar |
| Users loader | Calls `GET /api/users` on page load, fills both user dropdowns |
| Search handler | Listens to input on search box, calls `GET /api/search`, renders clickable results |
| `findSimilar(movieId)` | Called when user clicks a search result. Calls `GET /api/similar/:id`, renders similar movies |
| Collaborative handler | Listens to recommend button click. Calls `GET /api/recommend/:userId`, renders rated + recommended movies |
| Hybrid handler | Listens to recommend button click. Calls `GET /api/hybrid/:userId`, renders rated + recommended movies |
| `renderMovieCards(container, movies, type)` | Creates movie card HTML for each movie. Shows similarity%, predicted rating, or hybrid score depending on type |
| `renderRatedList(container, movies)` | Creates compact cards showing the user's previously rated movies with star ratings |

---

## API Endpoints

| Method | Endpoint | Description | Example |
|---|---|---|---|
| GET | `/api/search?q=query` | Search movies by title | `/api/search?q=sholay` |
| GET | `/api/similar/:movieId?n=10` | Get content-similar movies | `/api/similar/300003?n=10` |
| GET | `/api/recommend/:userId?n=10` | Collaborative recommendations | `/api/recommend/5?n=10` |
| GET | `/api/hybrid/:userId?n=10` | Hybrid recommendations | `/api/hybrid/5?n=10` |
| GET | `/api/users` | List available user IDs | `/api/users` |
| GET | `/api/metrics` | Model evaluation metrics | `/api/metrics` |

---

## Model Evaluation

After training, the model is evaluated on the training data:

| Metric | Value | Meaning |
|---|---|---|
| RMSE | 0.8684 | On average, predictions are off by 0.87 stars on a 5-star scale |
| MAE | 0.6825 | Average absolute prediction error is 0.68 stars |

---

Created for the CodSoft AI Internship — Task 4: Recommendation System
