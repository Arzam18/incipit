// Standard algebraic notation (SAN) for moves: writing it, and finding the
// legal move a SAN token names. Shared by the engine (EPD suites name moves
// in SAN) and datatools (PGN movetext), which includes this file directly.
use crate::position::*;

/// The legal moves of `pos`.
pub fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut list = MoveList::new();
    pos.gen_moves(&mut list, false);
    let mut v = Vec::with_capacity(list.len());
    for i in 0..list.len() {
        let mut c = *pos;
        if c.make_move(list[i]) {
            v.push(list[i]);
        }
    }
    v
}

fn sq_name(sq: usize) -> String {
    format!("{}{}", (b'a' + (sq % 8) as u8) as char, (b'1' + (sq / 8) as u8) as char)
}

/// SAN for `m` without check/mate suffixes. `legal` is the legal move list.
pub fn san(pos: &Position, m: Move, legal: &[Move]) -> String {
    match mflag(m) {
        F_KCASTLE => return "O-O".to_string(),
        F_QCASTLE => return "O-O-O".to_string(),
        _ => {}
    }
    let (from, to) = (mfrom(m), mto(m));
    let pt = pc_type(pos.board[from]);
    let capture = is_capture(m);
    let mut s = String::new();
    if pt == PAWN {
        if capture {
            s.push((b'a' + (from % 8) as u8) as char);
            s.push('x');
        }
        s.push_str(&sq_name(to));
        if is_promo(m) {
            s.push('=');
            s.push(b"PNBRQK"[promo_pt(m)] as char);
        }
        return s;
    }
    s.push(b"PNBRQK"[pt] as char);
    // Castling never needs telling apart from a normal move: in Chess960 the
    // king's castling destination can also be an ordinary king move.
    let rivals: Vec<usize> = legal
        .iter()
        .filter(|&&o| o != m && !is_castle(o) && mto(o) == to && pc_type(pos.board[mfrom(o)]) == pt)
        .map(|&o| mfrom(o))
        .collect();
    if !rivals.is_empty() {
        if rivals.iter().all(|&r| r % 8 != from % 8) {
            s.push((b'a' + (from % 8) as u8) as char);
        } else if rivals.iter().all(|&r| r / 8 != from / 8) {
            s.push((b'1' + (from / 8) as u8) as char);
        } else {
            s.push_str(&sq_name(from));
        }
    }
    if capture {
        s.push('x');
    }
    s.push_str(&sq_name(to));
    s
}

/// Finds the legal move matching a SAN token (annotations like +, #, !, ? are ignored).
pub fn find_san(pos: &Position, token: &str) -> Option<Move> {
    let want =
        token.trim_end_matches(|c| matches!(c, '+' | '#' | '!' | '?')).replace("0-0-0", "O-O-O").replace("0-0", "O-O");
    let legal = legal_moves(pos);
    legal.iter().copied().find(|&m| san(pos, m, &legal) == want)
}
