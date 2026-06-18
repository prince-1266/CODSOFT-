/* ══════════════════════════════════════════
   Tic-Tac-Toe AI — Minimax + Alpha-Beta
   CODSOFT Internship | Task 2
   ══════════════════════════════════════════ */

const HUMAN = 'X';
const AI = 'O';
const WINS = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6]
];

let board = Array(9).fill('');
let active = true;
let diff = 'hard';
let scores = { X: 0, O: 0, D: 0 };

const cells = document.querySelectorAll('.cell');
const status = document.getElementById('status');

// ── Cell clicks ──
cells.forEach(c => c.addEventListener('click', () => {
    const i = +c.dataset.i;
    if (board[i] || !active) return;

    play(i, HUMAN);
    if (!active) return;

    active = false;
    status.textContent = '🤖 AI thinking...';
    setTimeout(() => {
        play(bestMove(), AI);
        if (!winner(board, AI) && board.some(c => !c)) {
            active = true;
            status.textContent = 'Your turn!';
            status.className = 'status';
        }
    }, 250);
}));

// ── Place a mark ──
function play(i, p) {
    board[i] = p;
    cells[i].textContent = p;
    cells[i].classList.add('taken', p.toLowerCase());

    const w = winner(board, p);
    if (w) {
        active = false;
        w.forEach(j => cells[j].classList.add('win-cell'));
        if (p === HUMAN) {
            status.textContent = '🎉 You win!';
            status.className = 'status win';
            scores.X++;
        } else {
            status.textContent = '🤖 AI wins!';
            status.className = 'status lose';
            scores.O++;
        }
        updateScores();
        return;
    }

    if (board.every(c => c)) {
        active = false;
        status.textContent = "🤝 It's a draw!";
        status.className = 'status tied';
        scores.D++;
        updateScores();
    }
}

// ── Check winner ──
function winner(b, p) {
    for (const c of WINS)
        if (c.every(i => b[i] === p)) return c;
    return null;
}

// ── Scoreboard ──
function updateScores() {
    document.getElementById('scoreX').textContent = scores.X;
    document.getElementById('scoreO').textContent = scores.O;
    document.getElementById('scoreDraw').textContent = scores.D;
    document.querySelectorAll('.score').forEach(s => {
        s.classList.add('pop');
        setTimeout(() => s.classList.remove('pop'), 400);
    });
}

// ══════════════════════════════════════════
//  AI — Minimax with Alpha-Beta Pruning
// ══════════════════════════════════════════

function bestMove() {
    if (diff === 'easy') return randomMove();
    if (diff === 'medium') return Math.random() < 0.5 ? minimaxMove() : randomMove();
    return minimaxMove();
}

function randomMove() {
    const open = board.map((v,i) => v === '' ? i : -1).filter(i => i >= 0);
    return open[Math.floor(Math.random() * open.length)];
}

function minimaxMove() {
    let best = -Infinity, move = -1;
    for (let i = 0; i < 9; i++) {
        if (board[i]) continue;
        board[i] = AI;
        const s = minimax(0, false, -Infinity, Infinity);
        board[i] = '';
        if (s > best) { best = s; move = i; }
    }
    return move;
}

/**
 * Minimax with Alpha-Beta Pruning
 *
 * Scores: AI win = +10-depth, Human win = depth-10, Draw = 0
 * Alpha = best score maximizer can guarantee
 * Beta  = best score minimizer can guarantee
 * Prune when beta <= alpha (no better option exists)
 */
function minimax(depth, isMax, alpha, beta) {
    if (winner(board, AI))    return 10 - depth;
    if (winner(board, HUMAN)) return depth - 10;
    if (board.every(c => c))  return 0;

    if (isMax) {
        let max = -Infinity;
        for (let i = 0; i < 9; i++) {
            if (board[i]) continue;
            board[i] = AI;
            max = Math.max(max, minimax(depth+1, false, alpha, beta));
            board[i] = '';
            alpha = Math.max(alpha, max);
            if (beta <= alpha) break;
        }
        return max;
    } else {
        let min = Infinity;
        for (let i = 0; i < 9; i++) {
            if (board[i]) continue;
            board[i] = HUMAN;
            min = Math.min(min, minimax(depth+1, true, alpha, beta));
            board[i] = '';
            beta = Math.min(beta, min);
            if (beta <= alpha) break;
        }
        return min;
    }
}

// ══════════════════════════════════════════
//  Controls
// ══════════════════════════════════════════

function resetGame() {
    board = Array(9).fill('');
    active = true;
    status.textContent = 'Your turn!';
    status.className = 'status';
    cells.forEach(c => { c.textContent = ''; c.className = 'cell'; });
}

function setDiff(d) {
    diff = d;
    document.querySelectorAll('.d').forEach(b => b.classList.toggle('active', b.dataset.d === d));
    resetGame();
}
