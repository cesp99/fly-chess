// Board -> planes and move <-> index encoding: a line-by-line port of
// flychess/chessenv/encoding.py (docs/SPEC.md §3).  Depends only on chess.js.
//
// Everything is from the side to move's perspective: when black is to move the board is mirrored
// (colours swapped, ranks flipped: square ^ 56) so that the mover always pushes pawns upwards.
//
// Planes ([plane, rank, file], a1 = [0, 0], flattened plane-major -> Float32Array(1280)):
//   0-5   mover's P N B R Q K        6-11  opponent's P N B R Q K
//   12/13 mover castles K/Q          14/15 opponent castles K/Q          (all-ones planes)
//   16    en-passant target square (single 1, only if an en-passant capture is legal)
//   17    constant ones              18    halfmove clock / 100 clipped to 1 (all squares)
//   19    1 if the current position already occurred earlier in the game (all squares)
//
// Move index (perspective coordinates):
//   from * 64 + to                              non-promotion or queen promotion (0 .. 4095)
//   4096 + (fromFile * 3 + dir) * 3 + piece     under-promotion: dir 0/1/2 = capture-left / push /
//                                               capture-right, piece 0/1/2 = N / B / R (4096 .. 4167)

export const NUM_PLANES = 20;
export const FLAT_INPUT = NUM_PLANES * 64; // 1280
export const NUM_MOVES = 4096 + 72; // 4168
export const UNDERPROMOTION_BASE = 4096;

const PIECE_PLANE = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
const UNDERPROMOTION_CODE = { n: 0, b: 1, r: 2 };
const UNDERPROMOTION_PIECE = ['n', 'b', 'r'];
const FILES = 'abcdefgh';

/** 'e4' -> 28 (chess.Square numbering: a1 = 0, h8 = 63, rank * 8 + file). */
export function squareIndex(name) {
  return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
}

/** 28 -> 'e4'. */
export function squareName(sq) {
  return FILES[sq & 7] + String.fromCharCode(49 + (sq >> 3));
}

function perspectiveSquare(sq, turn) {
  return turn === 'w' ? sq : sq ^ 56;
}

/** First four FEN fields: pieces, side to move, castling rights, legal en-passant square. */
function positionKey(fen) {
  return fen.split(' ').slice(0, 4).join(' ');
}

/**
 * Number of times the current position has occurred in the game, current occurrence included
 * (same pieces, side to move, castling rights and *legal* en-passant square -- python-chess
 * `is_repetition(n)` semantics: `repetitionCount(chess) >= n`). Computed from the `before` FENs of
 * the move history. Use this rather than chess.js' `isThreefoldRepetition()`: its Zobrist counter
 * hashes the en-passant square even when the capture is illegal (pinned), so a position first
 * reached by such a double push is never matched again and the draw goes undetected.
 */
export function repetitionCount(chess) {
  const key = positionKey(chess.fen());
  let n = 1;
  for (const move of chess.history({ verbose: true })) {
    if (positionKey(move.before) === key) n++;
  }
  return n;
}

/** True if the current position already occurred earlier in the game (`is_repetition(2)`). */
export function positionRepeated(chess) {
  return repetitionCount(chess) >= 2;
}

/**
 * Encode the position into Float32Array(1280) planes (plane-major C order).
 * `opts.repeated` overrides the repetition plane; by default it is computed from the history.
 */
export function encodeBoard(chess, opts = {}) {
  const out = new Float32Array(FLAT_INPUT);
  const turn = chess.turn();
  const mover = turn;
  const board = chess.board(); // board[0] is rank 8, board[7] is rank 1
  for (let row = 0; row < 8; row++) {
    const rank = 7 - row;
    for (let file = 0; file < 8; file++) {
      const piece = board[row][file];
      if (!piece) continue;
      const sq = perspectiveSquare(rank * 8 + file, turn);
      const plane = PIECE_PLANE[piece.type] + (piece.color === mover ? 0 : 6);
      out[plane * 64 + sq] = 1;
    }
  }
  const fenFields = chess.fen().split(' ');
  const opp = mover === 'w' ? 'b' : 'w';
  const moverRights = chess.getCastlingRights(mover);
  const oppRights = chess.getCastlingRights(opp);
  const flags = [moverRights.k, moverRights.q, oppRights.k, oppRights.q];
  for (let i = 0; i < 4; i++) {
    if (flags[i]) out.fill(1, (12 + i) * 64, (13 + i) * 64);
  }
  // en-passant: only when a legal en-passant capture exists (chess.js prints it in the FEN only then)
  if (fenFields[3] !== '-') {
    out[16 * 64 + perspectiveSquare(squareIndex(fenFields[3]), turn)] = 1;
  }
  out.fill(1, 17 * 64, 18 * 64);
  const halfmove = parseInt(fenFields[4], 10) || 0;
  out.fill(Math.fround(Math.min(halfmove, 100) / 100), 18 * 64, 19 * 64);
  const repeated = opts.repeated === undefined ? positionRepeated(chess) : Boolean(opts.repeated);
  if (repeated) out.fill(1, 19 * 64, 20 * 64);
  return out;
}

/**
 * Index in [0, NUM_MOVES) of `move` for the side to move of `chess`.
 * `move` may be a chess.js verbose move, `{from, to, promotion?}` or a UCI string ('e7e8n').
 */
export function moveToIndex(move, chess) {
  const m = normaliseMove(move);
  const turn = chess.turn();
  const from = perspectiveSquare(squareIndex(m.from), turn);
  const to = perspectiveSquare(squareIndex(m.to), turn);
  const promo = m.promotion;
  if (!promo || promo === 'q') return from * 64 + to;
  const fromFile = from & 7;
  const dir = (to & 7) - fromFile + 1;
  if (dir < 0 || dir > 2 || from >> 3 !== 6 || to >> 3 !== 7) {
    throw new Error(`not an encodable under-promotion: ${m.from}${m.to}${promo}`);
  }
  return UNDERPROMOTION_BASE + (fromFile * 3 + dir) * 3 + UNDERPROMOTION_CODE[promo];
}

/**
 * Inverse of moveToIndex: `{from, to, promotion?, uci}` in real (unmirrored) coordinates.
 * Queen promotion is inferred from a pawn reaching the last rank.
 */
export function indexToMove(idx, chess) {
  if (!(idx >= 0 && idx < NUM_MOVES)) throw new Error(`move index out of range: ${idx}`);
  const turn = chess.turn();
  let from, to, promotion;
  if (idx < UNDERPROMOTION_BASE) {
    from = Math.floor(idx / 64);
    to = idx % 64;
    if (to >> 3 === 7 && from >> 3 === 6) {
      const piece = chess.get(squareName(perspectiveSquare(from, turn)));
      if (piece && piece.type === 'p') promotion = 'q';
    }
  } else {
    const rem = idx - UNDERPROMOTION_BASE;
    const fromFile = Math.floor(rem / 9);
    const rest = rem % 9;
    const dir = Math.floor(rest / 3);
    const toFile = fromFile + dir - 1;
    if (toFile < 0 || toFile > 7) throw new Error(`under-promotion index off the board: ${idx}`);
    from = 6 * 8 + fromFile;
    to = 7 * 8 + toFile;
    promotion = UNDERPROMOTION_PIECE[rest % 3];
  }
  const result = {
    from: squareName(perspectiveSquare(from, turn)),
    to: squareName(perspectiveSquare(to, turn)),
  };
  if (promotion) result.promotion = promotion;
  result.uci = result.from + result.to + (promotion || '');
  return result;
}

/** Sorted array of the indices of every legal move. */
export function legalMoveIndices(chess) {
  const indices = chess.moves({ verbose: true }).map((m) => moveToIndex(m, chess));
  indices.sort((a, b) => a - b);
  return indices;
}

/** Uint8Array(NUM_MOVES) with 1 at every legal move index. */
export function legalMoveMask(chess) {
  const mask = new Uint8Array(NUM_MOVES);
  for (const idx of legalMoveIndices(chess)) mask[idx] = 1;
  return mask;
}

/** UCI string -> `{from, to, promotion?}` accepted by chess.move(). */
export function uciToMove(uci) {
  const move = { from: uci.slice(0, 2), to: uci.slice(2, 4) };
  if (uci.length > 4) move.promotion = uci[4];
  return move;
}

function normaliseMove(move) {
  if (typeof move === 'string') return uciToMove(move);
  return move;
}
