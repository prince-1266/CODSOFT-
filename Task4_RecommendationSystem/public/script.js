document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('panel-' + target).classList.add('active');
        });
    });

    // Elements
    const searchInput = document.getElementById('search-input');
    const searchSpinner = document.getElementById('search-spinner');
    const searchResults = document.getElementById('search-results');
    const similarSection = document.getElementById('similar-section');
    const selectedMovieInfo = document.getElementById('selected-movie-info');
    const similarMoviesGrid = document.getElementById('similar-movies-grid');

    const collabUserSelect = document.getElementById('collab-user-select');
    const collabRecommendBtn = document.getElementById('collab-recommend-btn');
    const collabRatedSection = document.getElementById('collab-rated-section');
    const collabRatedMovies = document.getElementById('collab-rated-movies');
    const collabRecsSection = document.getElementById('collab-recs-section');
    const collabRecsGrid = document.getElementById('collab-recs-grid');

    const hybridUserSelect = document.getElementById('hybrid-user-select');
    const hybridRecommendBtn = document.getElementById('hybrid-recommend-btn');
    const hybridRatedSection = document.getElementById('hybrid-rated-section');
    const hybridRatedMovies = document.getElementById('hybrid-rated-movies');
    const hybridRecsSection = document.getElementById('hybrid-recs-section');
    const hybridRecsGrid = document.getElementById('hybrid-recs-grid');

    const loadingOverlay = document.getElementById('loading-overlay');

    // Helpers
    function showLoading() { loadingOverlay.style.display = 'flex'; }
    function hideLoading() { loadingOverlay.style.display = 'none'; }
    function debounce(fn, delay) {
        let timer;
        return function (...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    }

    // Load Metrics
    fetch('/api/metrics').then(res => res.json()).then(data => {
        document.getElementById('total-movies').textContent = data.totalMovies.toLocaleString();
        document.getElementById('total-users').textContent = data.totalUsers.toLocaleString();
        document.getElementById('total-ratings').textContent = data.totalRatings.toLocaleString();
    }).catch(err => console.error('Error fetching metrics:', err));

    // Load Users
    fetch('/api/users').then(res => res.json()).then(data => {
        let options = '<option value="">Select a User ID...</option>';
        data.users.forEach(id => { options += `<option value="${id}">User #${id}</option>`; });
        collabUserSelect.innerHTML = options;
        hybridUserSelect.innerHTML = options;
    }).catch(err => console.error('Error fetching users:', err));

    // Search
    searchInput.addEventListener('input', debounce(function () {
        const query = this.value.trim();
        if (query.length < 2) {
            searchResults.innerHTML = '';
            return;
        }
        searchSpinner.style.display = 'inline-block';
        fetch(`/api/search?q=${encodeURIComponent(query)}`)
            .then(res => res.json())
            .then(data => {
                searchResults.innerHTML = '';
                if(data.length === 0) {
                    searchResults.innerHTML = '<div style="padding: 10px;">No movies found.</div>';
                } else {
                    data.forEach(m => {
                        const div = document.createElement('div');
                        div.className = 'search-item';
                        div.innerHTML = `<strong>${m.title}</strong> <span style="color:#6c757d; font-size:14px;">(${m.genres.join(', ')})</span>`;
                        div.onclick = () => window.findSimilar(m.movieId);
                        searchResults.appendChild(div);
                    });
                }
                searchSpinner.style.display = 'none';
            })
            .catch(err => {
                console.error('Search error:', err);
                searchSpinner.style.display = 'none';
                searchResults.innerHTML = '<div style="padding: 10px; color: red;">Error performing search.</div>';
            });
    }, 300));

    // Similar
    window.findSimilar = function(movieId) {
        showLoading();
        fetch(`/api/similar/${movieId}?n=12`)
            .then(res => res.json())
            .then(data => {
                similarSection.style.display = 'block';
                selectedMovieInfo.textContent = data.movie.title;
                renderMovieCards(similarMoviesGrid, data.recommendations, 'similar');
                hideLoading();
            })
            .catch(err => {
                console.error('Similar movies fetch error:', err);
                hideLoading();
                alert('Failed to load similar movies.');
            });
    };

    // Collaborative
    collabRecommendBtn.addEventListener('click', () => {
        const userId = collabUserSelect.value;
        if (!userId) return;
        showLoading();
        fetch(`/api/recommend/${userId}?n=12`)
            .then(res => res.json())
            .then(data => {
                if(data.ratedMovies.length > 0) {
                    collabRatedSection.style.display = 'block';
                    renderRatedList(collabRatedMovies, data.ratedMovies);
                }
                collabRecsSection.style.display = 'block';
                renderMovieCards(collabRecsGrid, data.recommendations, 'collab');
                hideLoading();
            })
            .catch(err => {
                console.error('Collab recommend fetch error:', err);
                hideLoading();
                alert('Failed to load recommendations.');
            });
    });

    // Hybrid
    hybridRecommendBtn.addEventListener('click', () => {
        const userId = hybridUserSelect.value;
        if (!userId) return;
        showLoading();
        fetch(`/api/hybrid/${userId}?n=12`)
            .then(res => res.json())
            .then(data => {
                if(data.ratedMovies.length > 0) {
                    hybridRatedSection.style.display = 'block';
                    renderRatedList(hybridRatedMovies, data.ratedMovies);
                }
                hybridRecsSection.style.display = 'block';
                renderMovieCards(hybridRecsGrid, data.recommendations, 'hybrid');
                hideLoading();
            })
            .catch(err => {
                console.error('Hybrid recommend fetch error:', err);
                hideLoading();
                alert('Failed to load hybrid recommendations.');
            });
    });

    function renderMovieCards(container, movies, type) {
        container.innerHTML = '';
        if(movies.length === 0) {
            container.innerHTML = '<p>No recommendations available.</p>';
            return;
        }
        movies.forEach(m => {
            const div = document.createElement('div');
            div.className = 'movie-card';
            
            let stats = '';
            if (type === 'similar') {
                stats = `<span>Similarity: ${(m.similarity * 100).toFixed(1)}%</span>`;
            } else if (type === 'collab') {
                stats = `<span>Predicted Score: ${m.predictedRating} ★</span>`;
            } else if (type === 'hybrid') {
                stats = `<span>Hybrid Score: ${(m.hybridScore * 100).toFixed(1)}%</span>`;
            }

            div.innerHTML = `
                <h4>${m.title}</h4>
                <div class="movie-tags">${m.genres.join(', ')}</div>
                <div class="movie-stats">${stats}</div>
            `;
            container.appendChild(div);
        });
    }

    function renderRatedList(container, movies) {
        container.innerHTML = '';
        movies.forEach(m => {
            const div = document.createElement('div');
            div.className = 'rated-item';
            div.innerHTML = `<h5>${m.title}</h5><span>${m.rating} ★</span>`;
            container.appendChild(div);
        });
    }
});