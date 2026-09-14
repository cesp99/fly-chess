// board.js — dependency-free SVG chess board.
//
//   const board = new Board(el, { onMove(from, to, promotion), orientation: 'white' });
//   board.setPosition(fen);                 // animates piece moves
//   board.setLegal(chess.moves({verbose:true}));
//   board.highlight({ lastMove: ['e2', 'e4'], check: 'e8' });
//   board.setMovable('white' | 'black' | null);
//   board.cancelPromotion();               // dismiss the promotion picker (also closed by any of the above)
//
// Keyboard: the board is focusable; arrow keys move a cursor, Enter/Space selects a piece or drops
// it on the cursor square, Escape cancels the selection / promotion picker.
//
// Pieces are hand-drawn SVG (see PIECES); colours come from CSS custom properties on the host
// (--sq-light, --sq-dark, --piece-w, --piece-w-ink, --piece-b, --piece-b-ink, --hl-*).

const FILES = 'abcdefgh';
const S = 100;                        // square size in viewBox units

// Own artwork: 100×100 boxes, baseline at y=90. `.ink` parts take the outline colour.
const BASE = '<path d="M24 90H76V82Q76 77 70 77H30Q24 77 24 82Z"/>';
export const PIECES = {
  p: `<circle cx="50" cy="31" r="11"/><path d="M42 44H58L67 76H33Z"/>${BASE}`,
  r: `<path d="M27 44V24H36V32H44V24H56V32H64V24H73V44L69 50H31Z"/><path d="M32 50H68L66 76H34Z"/>${BASE}`,
  n: `<path d="M30 78C30 62 36 52 45 46H37C29 45 24 40 27 33L37 31C41 26 45 22 48 20L50 12L55 22L61 14L64 27C72 36 72 54 70 78Z"/><circle cx="55" cy="31" r="2.6" class="ink"/><circle cx="31" cy="37" r="1.8" class="ink"/><path d="M63 30C67 38 68 46 67 54" class="ink" fill="none"/>${BASE}`,
  b: `<circle cx="50" cy="19" r="5"/><path d="M50 25C65 39 66 52 60 62H40C34 52 35 39 50 25Z"/><path d="M50 36V52" class="ink" fill="none"/><path d="M36 62H64L67 76H33Z"/>${BASE}`,
  q: `<circle cx="30" cy="27" r="4"/><circle cx="50" cy="19" r="4"/><circle cx="70" cy="27" r="4"/><path d="M30 31L40 46L50 24L60 46L70 31L67 62H33Z"/><path d="M32 62H68L70 76H30Z"/>${BASE}`,
  k: `<path d="M46 10H54V17H61V25H54V32H46V25H39V17H46Z"/><path d="M35 40Q50 26 65 40L67 62H33Z"/><path d="M32 62H68L70 76H30Z"/>${BASE}`,
};

const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs = {}, parent) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

/** Parse the piece placement field of a FEN into a Map square -> {type, color}. */
export function parseFenPieces(fen) {
  const map = new Map();
  const rows = fen.split(' ')[0].split('/');
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') { file += +ch; continue; }
      const color = ch === ch.toUpperCase() ? 'w' : 'b';
      map.set(FILES[file] + (8 - r), { type: ch.toLowerCase(), color });
      file++;
    }
  }
  return map;
}

export class Board {
  /**
   * @param {HTMLElement} el host element (sized by CSS; the board fills it, keeping a square aspect)
   * @param {{onMove?:(from:string,to:string,promotion?:string)=>void, orientation?:'white'|'black', coordinates?:boolean, onSelect?:(sq:string|null)=>void}} opts
   */
  constructor(el, opts = {}) {
    this.el = el;
    this.onMove = opts.onMove || (() => {});
    this.onSelect = opts.onSelect || (() => {});
    this.orientation = opts.orientation || 'white';
    this.coordinates = opts.coordinates !== false;
    this.legal = [];
    this.movable = null;           // 'w' | 'b' | null
    this.selected = null;
    this.pieces = new Map();       // square -> {type, color, el}
    this.marks = { lastMove: null, check: null };
    this.drag = null;
    this.cursor = null;            // keyboard cursor square (shown only while driven by the keyboard)
    this._kb = false;
    this._build();
    this._bind();
  }

  // ------------------------------------------------------------------ construction
  _build() {
    this.el.classList.add('cb-host');
    const svg = svgEl('svg', {
      viewBox: `0 0 ${8 * S} ${8 * S}`, class: 'cb', role: 'application', tabindex: 0,
      'aria-label': 'chess board. Arrow keys move the cursor, Enter selects or drops a piece, Escape cancels.',
    });
    this.svg = svg;
    this.gSquares = svgEl('g', { class: 'cb-squares' }, svg);
    this.gMarks = svgEl('g', { class: 'cb-marks' }, svg);
    this.gCoords = svgEl('g', { class: 'cb-coords' }, svg);
    this.gPieces = svgEl('g', { class: 'cb-pieces' }, svg);
    this.gDots = svgEl('g', { class: 'cb-dots' }, svg);
    this.gPromo = svgEl('g', { class: 'cb-promo' }, svg);
    this.squareEls = {};
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
      const sq = FILES[f] + (r + 1);
      const rect = svgEl('rect', { width: S, height: S, class: (f + r) % 2 ? 'cb-light' : 'cb-dark', 'data-sq': sq }, this.gSquares);
      this.squareEls[sq] = rect;
    }
    this.el.replaceChildren(svg);
    this._layoutSquares();
  }

  _xy(sq) {
    const f = FILES.indexOf(sq[0]), r = +sq[1] - 1;
    const white = this.orientation === 'white';
    return { x: (white ? f : 7 - f) * S, y: (white ? 7 - r : r) * S };
  }

  _sqAt(px, py) {
    const pt = this.svg.createSVGPoint();
    pt.x = px; pt.y = py;
    const p = pt.matrixTransform(this.svg.getScreenCTM().inverse());
    const col = Math.floor(p.x / S), row = Math.floor(p.y / S);
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    const white = this.orientation === 'white';
    const f = white ? col : 7 - col, r = white ? 7 - row : row;
    return { sq: FILES[f] + (r + 1), x: p.x, y: p.y };
  }

  _layoutSquares() {
    for (const [sq, rect] of Object.entries(this.squareEls)) {
      const { x, y } = this._xy(sq);
      rect.setAttribute('x', x); rect.setAttribute('y', y);
    }
    this.gCoords.replaceChildren();
    if (this.coordinates) {
      const white = this.orientation === 'white';
      for (let i = 0; i < 8; i++) {
        const fileIdx = white ? i : 7 - i, rankIdx = white ? 7 - i : i;         // column i / row i
        const bottomRank = white ? 0 : 7, leftFile = white ? 0 : 7;
        const light = (f, r) => ((f + r) % 2 ? 'on-light' : 'on-dark');
        svgEl('text', { x: i * S + S - 6, y: 8 * S - 6, class: 'cb-coord ' + light(fileIdx, bottomRank), 'text-anchor': 'end' }, this.gCoords).textContent = FILES[fileIdx];
        svgEl('text', { x: 6, y: i * S + 18, class: 'cb-coord ' + light(leftFile, rankIdx) }, this.gCoords).textContent = rankIdx + 1;
      }
    }
    for (const [sq, p] of this.pieces) this._place(p.el, sq, false);
    this._renderMarks();
    this._renderDots();
  }

  _makePiece(type, color) {
    const g = svgEl('g', { class: `cb-piece cb-${color}`, 'data-type': type, 'data-color': color });
    g.innerHTML = PIECES[type];
    this.gPieces.appendChild(g);
    return g;
  }

  _place(el, sq, animate = true) {
    const { x, y } = this._xy(sq);
    el.style.transition = animate ? '' : 'none';
    el.style.transform = `translate(${x}px, ${y}px)`;
    if (!animate) { void el.getBBox?.(); el.style.transition = ''; }
  }

  // ------------------------------------------------------------------ public API
  /** Set orientation ('white' | 'black') and re-layout. */
  setOrientation(color) {
    this._closePromotion();
    this.orientation = color;
    this._layoutSquares();
  }

  flip() { this.setOrientation(this.orientation === 'white' ? 'black' : 'white'); }

  /** Which side the user may move; null locks the board. */
  setMovable(color) {
    this._closePromotion();          // a picker only makes sense for the position/turn it was opened in
    this.movable = color === 'white' ? 'w' : color === 'black' ? 'b' : color || null;
    if (!this.movable) this._select(null);
    this.el.classList.toggle('cb-locked', !this.movable);
  }

  /** Legal moves as chess.js verbose objects ({from, to, promotion?}). */
  setLegal(list) {
    this.legal = list || [];
    this._renderDots();
  }

  /** Update the position from a FEN, animating pieces that moved. */
  setPosition(fen, { animate = true } = {}) {
    this._closePromotion();
    const next = parseFenPieces(fen);
    const removed = [];
    for (const [sq, p] of this.pieces) {
      const n = next.get(sq);
      if (n && n.type === p.type && n.color === p.color) next.delete(sq);
      else removed.push([sq, p]);
    }
    for (const [sq, p] of removed) this.pieces.delete(sq);
    for (const [sq, n] of next) {
      // reuse a vanished piece of the same kind (closest one) so it slides instead of popping
      let best = -1, bestD = Infinity;
      removed.forEach(([osq, op], i) => {
        if (op.type !== n.type || op.color !== n.color) return;
        const d = Math.abs(FILES.indexOf(osq[0]) - FILES.indexOf(sq[0])) + Math.abs(+osq[1] - +sq[1]);
        if (d < bestD) { bestD = d; best = i; }
      });
      let el;
      if (best >= 0) { el = removed[best][1].el; removed.splice(best, 1); this.gPieces.appendChild(el); }
      else { el = this._makePiece(n.type, n.color); if (animate) el.classList.add('cb-appear'); }
      this.pieces.set(sq, { type: n.type, color: n.color, el });
      this._place(el, sq, animate && best >= 0);
    }
    for (const [, p] of removed) {
      p.el.classList.add('cb-vanish');
      const el = p.el;
      setTimeout(() => el.remove(), animate ? 160 : 0);
    }
    this._select(null);
  }

  /** @param {{lastMove?: [string,string]|null, check?: string|null}} marks */
  highlight(marks) {
    this.marks = { ...this.marks, ...marks };
    this._renderMarks();
  }

  /** Dismiss the promotion picker, if open (the pending move is dropped). */
  cancelPromotion() { this._closePromotion(); }

  /** Squares of the pieces currently on the board (for tests / share card). */
  position() {
    const out = {};
    for (const [sq, p] of this.pieces) out[sq] = p.color + p.type;
    return out;
  }

  // ------------------------------------------------------------------ rendering helpers
  _renderMarks() {
    this.gMarks.replaceChildren();
    const { lastMove, check } = this.marks;
    if (lastMove) for (const sq of lastMove) { const { x, y } = this._xy(sq); svgEl('rect', { x, y, width: S, height: S, class: 'cb-last' }, this.gMarks); }
    if (check) { const { x, y } = this._xy(check); svgEl('rect', { x, y, width: S, height: S, class: 'cb-check' }, this.gMarks); }
    if (this.selected) { const { x, y } = this._xy(this.selected); svgEl('rect', { x, y, width: S, height: S, class: 'cb-selected' }, this.gMarks); }
    if (this.cursor && this._kb) { const { x, y } = this._xy(this.cursor); svgEl('rect', { x, y, width: S, height: S, class: 'cb-cursor' }, this.gMarks); }
  }

  _renderDots() {
    this.gDots.replaceChildren();
    if (!this.selected) return;
    const seen = new Set();
    for (const m of this.legal) {
      if (m.from !== this.selected || seen.has(m.to)) continue;
      seen.add(m.to);
      const { x, y } = this._xy(m.to);
      const capture = this.pieces.has(m.to) || /e/.test(m.flags || '');
      if (capture) svgEl('circle', { cx: x + S / 2, cy: y + S / 2, r: S * 0.44, class: 'cb-dot cb-capture' }, this.gDots);
      else svgEl('circle', { cx: x + S / 2, cy: y + S / 2, r: S * 0.15, class: 'cb-dot' }, this.gDots);
    }
  }

  _select(sq) {
    this.selected = sq;
    this._renderMarks();
    this._renderDots();
    this.onSelect(sq);
  }

  _legalTargets(from) {
    return this.legal.filter((m) => m.from === from);
  }

  _tryMove(from, to) {
    const options = this.legal.filter((m) => m.from === from && m.to === to);
    if (options.length === 0) return false;
    if (options.some((m) => m.promotion)) {
      this._askPromotion(from, to, options[0].color || this.pieces.get(from)?.color);
      return true;
    }
    this._select(null);
    this.onMove(from, to, undefined);
    return true;
  }

  _askPromotion(from, to, color) {
    this.gPromo.replaceChildren();
    const { x, y } = this._xy(to);
    const down = y === 0 ? 1 : -1;               // list pieces away from the edge
    const choices = ['q', 'n', 'r', 'b'];
    svgEl('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, class: 'cb-promo-veil' }, this.gPromo)
      .addEventListener('pointerdown', (e) => { e.stopPropagation(); this._closePromotion(); });
    choices.forEach((pc, i) => {
      const yy = y + down * i * S;
      const g = svgEl('g', { class: `cb-promo-choice cb-${color}`, tabindex: 0, role: 'button', 'aria-label': `promote to ${pc}` }, this.gPromo);
      svgEl('rect', { x, y: yy, width: S, height: S, rx: 10 }, g);
      const p = svgEl('g', { class: `cb-piece cb-${color} cb-static` }, g);
      p.innerHTML = PIECES[pc];
      p.style.transform = `translate(${x}px, ${yy}px)`;
      const pick = (e) => { e.stopPropagation(); e.preventDefault(); this._closePromotion(); this._select(null); this.onMove(from, to, pc); };
      g.addEventListener('pointerdown', pick);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') pick(e); });
    });
    if (this._kb) this.gPromo.querySelector('.cb-promo-choice')?.focus?.();
  }

  _closePromotion() {
    if (!this.gPromo.childElementCount) return;
    const hadFocus = typeof document !== 'undefined' && this.gPromo.contains(document.activeElement);
    this.gPromo.replaceChildren();
    if (hadFocus) this.svg.focus?.();          // keep keyboard users on the board
  }

  // ------------------------------------------------------------------ interaction
  _bind() {
    const svg = this.svg;
    svg.addEventListener('pointerdown', (e) => this._down(e));
    svg.addEventListener('pointermove', (e) => this._move(e));
    svg.addEventListener('pointerup', (e) => this._up(e));
    svg.addEventListener('pointercancel', () => this._cancelDrag());
    svg.addEventListener('contextmenu', (e) => e.preventDefault());
    svg.addEventListener('keydown', (e) => this._key(e));
    svg.addEventListener('blur', () => this._setKb(false));
  }

  _setKb(on) {
    if (this._kb === on) return;
    this._kb = on;
    this._renderMarks();
  }

  /** Move the keyboard cursor by (dx, dy) screen squares (viewer's perspective). */
  _moveCursor(dx, dy) {
    const white = this.orientation === 'white';
    let f, r;
    if (this.cursor) { f = FILES.indexOf(this.cursor[0]); r = +this.cursor[1] - 1; }
    else { f = white ? 4 : 3; r = white ? 0 : 7; }                 // start on the near king square
    f += white ? dx : -dx; r += white ? -dy : dy;
    if (f < 0 || f > 7 || r < 0 || r > 7) return;
    this.cursor = FILES[f] + (r + 1);
  }

  _key(e) {
    if (this.gPromo.contains(e.target)) {                           // inside the promotion picker
      if (e.key === 'Escape') { e.preventDefault(); this._closePromotion(); }
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      if (this.gPromo.childElementCount) this._closePromotion();
      else if (this.selected) this._select(null);
      return;
    }
    const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (ARROWS[e.key]) {
      e.preventDefault();
      if (this.gPromo.childElementCount) return;
      this._setKb(true);
      this._moveCursor(...ARROWS[e.key]);
      this._renderMarks();
      return;
    }
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    if (!this.movable || this.gPromo.childElementCount) return;
    this._setKb(true);
    if (!this.cursor) { this._moveCursor(0, 0); this._renderMarks(); return; }
    const sq = this.cursor;
    if (this.selected && this.selected !== sq && this._tryMove(this.selected, sq)) return;
    const piece = this.pieces.get(sq);
    if (piece && piece.color === this.movable && this.selected !== sq) this._select(sq);
    else this._select(null);
  }

  _down(e) {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    this._setKb(false);
    if (!this.movable || this.gPromo.childElementCount) return;
    const hit = this._sqAt(e.clientX, e.clientY);
    if (!hit) return;
    const piece = this.pieces.get(hit.sq);
    if (this.selected && this.selected !== hit.sq && this._tryMove(this.selected, hit.sq)) { e.preventDefault(); return; }
    if (!piece || piece.color !== this.movable) { this._select(null); return; }
    e.preventDefault();
    this._select(hit.sq);
    this.drag = { from: hit.sq, el: piece.el, moved: false, x0: hit.x, y0: hit.y, pointerId: e.pointerId };
    piece.el.classList.add('cb-dragging');
    this.gPieces.appendChild(piece.el);
    try { this.svg.setPointerCapture(e.pointerId); } catch { /* ignore */ }
  }

  _move(e) {
    const d = this.drag;
    if (!d) return;
    const hit = this._sqAt(e.clientX, e.clientY);
    if (!hit) return;
    if (!d.moved && Math.hypot(hit.x - d.x0, hit.y - d.y0) < S * 0.08) return;
    d.moved = true;
    d.el.style.transition = 'none';
    d.el.style.transform = `translate(${hit.x - S / 2}px, ${hit.y - S / 2}px)`;
  }

  _up(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    d.el.classList.remove('cb-dragging');
    try { this.svg.releasePointerCapture(d.pointerId); } catch { /* ignore */ }
    if (!d.moved) { this._place(d.el, d.from, false); return; }  // click-select: keep selection, wait for target
    const hit = this._sqAt(e.clientX, e.clientY);
    d.el.style.transition = '';
    if (hit && hit.sq !== d.from && this._tryMove(d.from, hit.sq)) {
      const still = this.pieces.get(d.from);
      if (this.gPromo.childElementCount) this._place(d.el, d.from, false);        // picker open: snap back
      else if (still && still.el === d.el) this._place(d.el, hit.sq, false);     // app applies the move later
      // otherwise setPosition() already re-homed the piece
      return;
    }
    this._place(d.el, d.from, true);
    if (hit && hit.sq !== d.from) this._select(null);
  }

  _cancelDrag() {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    d.el.classList.remove('cb-dragging');
    this._place(d.el, d.from, true);
  }
}
