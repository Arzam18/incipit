// EPD test suites: `incipit epd <file> [nodes N | movetime MS | depth D]
// [hash MB] [threads T] [perft D] [verbose]`.
//
// Each line is a position (the four EPD fields, optionally followed by the
// halfmove and fullmove numbers) and operations: bm / am (best and avoid
// moves, in SAN or UCI notation), dm (mate in N for the side to move), id,
// and Dn (the perft count at depth n). Other operations (c0-c9, acd, ce, ...)
// are ignored. Operations are normally separated by `;`; a line without any
// `;` is split at the operation names instead (some suites are written so).
//
// Each position is searched from a cleared state. A position is solved when
// the final move is one of the bm moves and none of the am moves, and when
// a dm position's score is a mate in at most N moves. The report gives the
// iteration from which the answer was found and kept. A suite score is a
// check on pruning and on kinds of positions, not a measure of strength.
use crate::position::*;
use crate::san::{find_san, legal_moves, san};
use crate::search::{score_str, Limits, Searcher, MATE, MAX_PLY};
use crate::smp::Pool;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Default, PartialEq)]
pub struct Entry {
    pub fen: String,
    pub id: String,
    pub bm: Vec<String>,
    pub am: Vec<String>,
    pub dm: Option<u32>,
    pub perft: Vec<(u32, u64)>,
}

/// Splits on whitespace and control characters (some suites hold NULs) and
/// returns each `;` as a token of its own; double quoted text (an id such as
/// "BK.01; note") stays one token, quotes kept.
fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for c in s.chars() {
        if quoted {
            cur.push(c);
            quoted = c != '"';
        } else if c == '"' {
            cur.push(c);
            quoted = true;
        } else if c.is_whitespace() || c.is_control() || c == ';' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if c == ';' {
                out.push(";".to_string());
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn perft_op(op: &str) -> Option<u32> {
    op.strip_prefix('D').and_then(|d| d.parse().ok())
}

/// Operation names that can start an operation in a line without `;`. The
/// comment operations c0-c9 are left out: c1..c8 are also pawn moves.
fn is_opcode(t: &str) -> bool {
    matches!(t, "bm" | "am" | "dm" | "id" | "pv" | "pm" | "sm" | "acd" | "acn" | "acs" | "ce") || perft_op(t).is_some()
}

/// Parses one line. Ok(None) for a blank or `#` comment line.
pub fn parse_line(line: &str) -> Result<Option<Entry>, String> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let t = tokens(line);
    if t.len() < 4 || t[..4].iter().any(|x| x == ";") {
        return Err("fewer than four position fields".into());
    }
    let mut fen = t[..4].join(" ");
    let mut i = 4;
    // Optional halfmove and fullmove numbers.
    while i < 6 && i < t.len() && t[i].parse::<u32>().is_ok() {
        fen.push(' ');
        fen.push_str(&t[i]);
        i += 1;
    }
    let rest = &t[i..];
    let ops: Vec<&[String]> = if rest.iter().any(|x| x == ";") {
        rest.split(|x| x == ";").filter(|s| !s.is_empty()).collect()
    } else {
        let mut v: Vec<&[String]> = Vec::new();
        let mut start = None;
        for (k, x) in rest.iter().enumerate() {
            if is_opcode(x) {
                if let Some(s) = start {
                    v.push(&rest[s..k]);
                }
                start = Some(k);
            }
        }
        if let Some(s) = start {
            v.push(&rest[s..]);
        }
        v
    };
    let mut e = Entry { fen, ..Default::default() };
    // Move operands without check or quality marks written apart ("Qd1 +").
    // A promotion written apart ("f8 = Q", "f8 =Q", "f8= Q") is joined again.
    let moves = |args: &[String]| -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for a in args.iter().filter(|a| !a.chars().all(|c| matches!(c, '+' | '#' | '!' | '?'))) {
            match v.last_mut() {
                Some(last) if last.ends_with('=') || a.starts_with('=') => last.push_str(a),
                _ => v.push(a.clone()),
            }
        }
        v
    };
    for op in ops {
        let (name, args) = (op[0].as_str(), &op[1..]);
        match name {
            "bm" => e.bm.extend(moves(args)),
            "am" => e.am.extend(moves(args)),
            "dm" => {
                let n = args.first().and_then(|a| a.parse().ok()).ok_or("dm without a number")?;
                e.dm = Some(n);
            }
            "id" => e.id = args.join(" ").trim_matches('"').to_string(),
            _ => {
                if let Some(d) = perft_op(name) {
                    let n = args.first().and_then(|a| a.parse().ok()).ok_or(format!("{} without a count", name))?;
                    e.perft.push((d, n));
                }
            }
        }
    }
    Ok(Some(e))
}

/// The legal move a suite names: SAN, SAN with the promotion's `=` left out
/// (e8Q), or UCI (e7e8q).
pub fn resolve(pos: &Position, tok: &str) -> Option<Move> {
    if let Some(m) = find_san(pos, tok) {
        return Some(m);
    }
    let bare = tok.trim_end_matches(['+', '#', '!', '?']);
    let b = bare.as_bytes();
    if b.len() >= 3 && b"NBRQ".contains(&b[b.len() - 1]) && b[b.len() - 2].is_ascii_digit() {
        let with_eq = format!("{}={}", &bare[..bare.len() - 1], &bare[bare.len() - 1..]);
        if let Some(m) = find_san(pos, &with_eq) {
            return Some(m);
        }
    }
    let lower = bare.to_ascii_lowercase();
    legal_moves(pos).into_iter().find(|&m| pos.move_uci(m) == lower)
}

/// The legal moves a loosely written SAN token could mean, for suites that
/// leave out the capture mark (Nce5 for Ncxe5) or also a needed
/// disambiguation (Ne5): first the SAN without `x`, then for pieces only the
/// piece and destination.
pub fn resolve_loose(pos: &Position, tok: &str) -> Vec<Move> {
    let no_x = |s: &str| -> String { s.trim_end_matches(['+', '#', '!', '?']).chars().filter(|&c| c != 'x').collect() };
    let piece_dest = |s: &str| -> Option<String> {
        let s = no_x(s);
        match s.as_bytes() {
            [p @ (b'N' | b'B' | b'R' | b'Q' | b'K'), .., f, r] if s.len() >= 3 => {
                Some(format!("{}{}{}", *p as char, *f as char, *r as char))
            }
            _ => None,
        }
    };
    let legal = legal_moves(pos);
    let sans: Vec<String> = legal.iter().map(|&m| san(pos, m, &legal)).collect();
    let want = no_x(tok);
    let exact: Vec<Move> = legal.iter().zip(&sans).filter(|(_, s)| no_x(s) == want).map(|(&m, _)| m).collect();
    if !exact.is_empty() {
        return exact;
    }
    let Some(want) = piece_dest(tok) else { return Vec::new() };
    legal.iter().zip(&sans).filter(|(_, s)| piece_dest(s).as_ref() == Some(&want)).map(|(&m, _)| m).collect()
}

/// Why Position::from_fen refused a position, for the report: the checks it
/// makes on a parsable board, else a malformed FEN.
fn why_rejected(fen: &str) -> &'static str {
    let f: Vec<&str> = fen.split_whitespace().collect();
    let board = f[0];
    let ranks: Vec<&str> = board.split('/').collect();
    let well_formed = ranks.len() == 8
        && ranks.iter().all(|r| {
            r.chars().map(|c| if c.is_ascii_digit() { c as usize - '0' as usize } else { 1 }).sum::<usize>() == 8
                && r.chars().all(|c| c.is_ascii_digit() || "pnbrqkPNBRQK".contains(c))
        });
    if !well_formed || !matches!(f[1], "w" | "b") {
        return "malformed FEN:";
    }
    let count = |set: &str| board.chars().filter(|&c| set.contains(c)).count();
    if count("K") != 1 || count("k") != 1 {
        "not one king a side:"
    } else if count("PNBRQK") > 16 || count("pnbrqk") > 16 {
        "more than 16 pieces a side (impossible in a game; the net's material buckets stop at 32):"
    } else if ranks[0].contains(['p', 'P']) || ranks[7].contains(['p', 'P']) {
        "pawn on a back rank:"
    } else {
        "side not to move is in check:"
    }
}

pub enum Limit {
    Nodes(u64),
    Ms(u64),
    Depth(i32),
}

/// Mate in at most `n` moves for the side to move.
fn mate_within(score: i32, n: u32) -> bool {
    score >= MATE - MAX_PLY as i32 && score >= MATE - (2 * n as i32 - 1)
}

fn short(n: u64) -> String {
    match n {
        0..=9_999 => n.to_string(),
        10_000..=9_999_999 => format!("{}k", n / 1000),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Summary {
    pub searched: usize,
    pub solved: usize,
    pub skipped: usize,
    pub perft_ok: usize,
    pub perft_bad: usize,
}

/// Runs a suite and prints one line per position and a summary.
pub fn run(path: &str, limit: Limit, hash_mb: usize, threads: usize, perft_max: u32, verbose: bool) -> Summary {
    let text = match std::fs::read(path) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => {
            eprintln!("{}: {}", path, e);
            std::process::exit(2);
        }
    };
    let mut s = Searcher::new(hash_mb, Arc::new(AtomicBool::new(false)));
    s.silent = !verbose;
    let pool = Pool::new(threads.max(1) - 1);
    s.helper_slots = pool.slots();
    let lim = match limit {
        Limit::Nodes(n) => Limits { soft_ms: None, hard_ms: None, depth: MAX_PLY as i32, nodes: Some(n) },
        Limit::Ms(ms) => Limits { soft_ms: None, hard_ms: Some(ms), depth: MAX_PLY as i32, nodes: None },
        Limit::Depth(d) => Limits { soft_ms: None, hard_ms: None, depth: d, nodes: None },
    };
    let t0 = Instant::now();
    let (mut tried, mut solved, mut skipped) = (0, 0, 0);
    let (mut perft_ok, mut perft_bad) = (0, 0);
    let mut failed: Vec<String> = Vec::new();
    let mut total_nodes = 0u64;
    for (ln, line) in text.lines().enumerate() {
        let ln = ln + 1;
        let e = match parse_line(line) {
            Ok(Some(e)) => e,
            Ok(None) => continue,
            Err(why) => {
                println!("line {}: skipped: {}", ln, why);
                skipped += 1;
                continue;
            }
        };
        let name = if e.id.is_empty() { format!("line {}", ln) } else { e.id.clone() };
        let Some(pos) = Position::from_fen(&e.fen) else {
            println!("{}: skipped: {} {}", name, why_rejected(&e.fen), e.fen);
            skipped += 1;
            continue;
        };
        for &(d, exp) in &e.perft {
            if d > perft_max {
                continue;
            }
            let n = perft(&pos, d);
            if n == exp {
                perft_ok += 1;
            } else {
                perft_bad += 1;
                println!("{}: perft FAIL D{} got {} expected {} ({})", name, d, n, exp, e.fen);
            }
        }
        if e.bm.is_empty() && e.am.is_empty() && e.dm.is_none() {
            continue;
        }
        let legal = legal_moves(&pos);
        if legal.is_empty() {
            println!("{}: skipped: no legal moves", name);
            skipped += 1;
            continue;
        }
        // A best move written loosely must still name one move; a loose avoid
        // move covers every move it could mean.
        let lookup = |toks: &[String], any: bool| -> Result<Vec<Move>, String> {
            let mut v = Vec::new();
            for t in toks {
                match resolve(&pos, t) {
                    Some(m) => v.push(m),
                    None => match resolve_loose(&pos, t) {
                        l if l.len() == 1 || (any && !l.is_empty()) => v.extend(l),
                        l if l.is_empty() => return Err(format!("unknown or illegal move {}", t)),
                        _ => return Err(format!("ambiguous move {}", t)),
                    },
                }
            }
            Ok(v)
        };
        let (bm, am) = match (lookup(&e.bm, false), lookup(&e.am, true)) {
            (Ok(b), Ok(a)) => (b, a),
            (Err(why), _) | (_, Err(why)) => {
                println!("{}: skipped: {}", name, why);
                skipped += 1;
                continue;
            }
        };
        let good = |m: Move, sc: i32| {
            (bm.is_empty() || bm.contains(&m)) && !am.contains(&m) && e.dm.is_none_or(|n| mate_within(sc, n))
        };
        s.clear();
        s.hash_hist.clear();
        if pool.len() > 0 {
            pool.each(|h| h.clear());
        }
        s.iter_log = Some(Vec::new());
        pool.start(&pos, &[], &s);
        let (m, sc) = s.search(&pos, &lim);
        pool.finish();
        let log = s.iter_log.take().unwrap_or_default();
        let nodes = s.total_nodes();
        total_nodes += nodes;
        tried += 1;
        let mut want = Vec::new();
        if !e.bm.is_empty() {
            want.push(format!("bm {}", e.bm.join(" ")));
        }
        if !e.am.is_empty() {
            want.push(format!("am {}", e.am.join(" ")));
        }
        if let Some(n) = e.dm {
            want.push(format!("dm {}", n));
        }
        let got = format!("{} ({})", san(&pos, m, &legal), score_str(sc, &pos));
        if good(m, sc) {
            solved += 1;
            // The first iteration of the final run of iterations that all
            // had a right answer.
            let kept = log.iter().rposition(|r| !good(r.best, r.score)).map_or(0, |k| k + 1);
            let found = match log.get(kept) {
                Some(r) => format!("found at depth {}, {} nodes, {} ms", r.depth, short(r.nodes), r.ms),
                None => "found in the last, unfinished iteration".to_string(),
            };
            println!("ok    {}: {} -> {}, {}", name, want.join(", "), got, found);
        } else {
            let depth = log.last().map_or(0, |r| r.depth);
            // A mate position whose move was right but whose mate wasn't proven.
            let note = if e.dm.is_some() && !bm.is_empty() && bm.contains(&m) && !am.contains(&m) {
                " (right move, mate not proven)"
            } else {
                ""
            };
            println!("FAIL  {}: {} -> {}, depth {}, {} nodes{}", name, want.join(", "), got, depth, short(nodes), note);
            failed.push(name);
        }
    }
    let el = t0.elapsed();
    println!();
    if tried > 0 {
        println!("solved {} of {} ({:.1}%)", solved, tried, 100.0 * solved as f64 / tried as f64);
    }
    if !failed.is_empty() && failed.len() <= 50 {
        println!("failed: {}", failed.join(", "));
    }
    if perft_ok + perft_bad > 0 {
        println!("perft: {} counts match, {} differ (depth <= {})", perft_ok, perft_bad, perft_max);
    }
    if skipped > 0 {
        println!("skipped {} lines (see above)", skipped);
    }
    if tried > 0 {
        println!(
            "{} nodes, {:.1} s, {} nps",
            total_nodes,
            el.as_secs_f64(),
            (total_nodes as f64 / el.as_secs_f64().max(1e-3)) as u64
        );
    } else {
        println!("{:.1} s", el.as_secs_f64());
    }
    Summary { searched: tried, solved, skipped, perft_ok, perft_bad }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        crate::attacks::init();
        crate::nnue::init();
    }

    #[test]
    fn parses_common_layouts() {
        init();
        let e =
            parse_line("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - ; bm Qd1 + ; id \"BK.01\"").unwrap().unwrap();
        assert_eq!(e.fen, "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - -");
        assert_eq!((e.bm, e.id), (vec!["Qd1".to_string()], "BK.01".to_string()));
        // No semicolons, move numbers present.
        let e = parse_line("r7/4pk2/PP1r4/2R2P2/8/1KP5/8/R7 w - - 0 1 bm Rb5 dm 14").unwrap().unwrap();
        assert_eq!(e.fen, "r7/4pk2/PP1r4/2R2P2/8/1KP5/8/R7 w - - 0 1");
        assert_eq!((e.bm, e.dm), (vec!["Rb5".to_string()], Some(14)));
        // Perft counts, a comment operation and a quoted id holding a `;`.
        let e = parse_line("8/8/8/8/8/8/8/K6k w - -  0 1; D1 3 ;D2 9; c0 \"a; b\"; id \"x;y\"").unwrap().unwrap();
        assert_eq!((e.perft, e.id), (vec![(1, 3), (2, 9)], "x;y".to_string()));
        // An avoid move with its marks written apart.
        let e = parse_line("8/5P2/8/8/8/8/8/K6k w - -; bm f8 = Q; c1 diff=7").unwrap().unwrap();
        assert_eq!(e.bm, vec!["f8=Q".to_string()]);
        let e = parse_line("2k5/8/8/8/8/8/8/K1N5 w - -; am Nb3 + ? ; c1 diff=10").unwrap().unwrap();
        assert_eq!(e.am, vec!["Nb3".to_string()]);
        assert_eq!(parse_line("   "), Ok(None));
        assert!(parse_line("8/8/8 w ;").is_err());
    }

    #[test]
    fn resolves_san_uci_and_bare_promotions() {
        init();
        let pos = Position::from_fen("1n5k/P7/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let q = resolve(&pos, "a8=Q").unwrap();
        assert_eq!(resolve(&pos, "a8Q"), Some(q));
        assert_eq!(resolve(&pos, "a7a8q"), Some(q));
        let x = resolve(&pos, "axb8=N+").unwrap();
        assert_eq!(pos.move_uci(x), "a7b8n");
        assert_eq!(resolve(&pos, "Kb3"), None);
        // Capture mark or disambiguation left out.
        let pos = Position::from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10").unwrap();
        assert_eq!(resolve(&pos, "Ne5"), None);
        let mut both: Vec<String> = resolve_loose(&pos, "Ne5").iter().map(|&m| pos.move_uci(m)).collect();
        both.sort();
        assert_eq!(both, vec!["c6e5", "g4e5"]);
        assert_eq!(resolve_loose(&pos, "Nce5").len(), 1);
        assert_eq!(resolve_loose(&pos, "Bxe5"), vec![]);
    }

    #[test]
    fn rejected_positions_say_why() {
        let w = |fen: &str| why_rejected(fen).split(' ').next().unwrap();
        assert_eq!(w("k7/1p1p1p1p/pPpPpPpP/P1P1P1P1/1B1B1B1B/B1B1B1B1/1B1B1B1B/K1B1B1B1 w - -"), "more");
        assert_eq!(w("rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR K KQkq -"), "malformed");
        assert_eq!(w("rnbqk0nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"), "malformed");
        assert_eq!(w("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"), "not");
        assert_eq!(w("P6k/8/8/8/8/8/8/K7 w - -"), "pawn");
        assert_eq!(w("k7/8/8/8/8/8/8/K6R w - -"), "side");
    }

    #[test]
    fn mate_scores_count_in_moves() {
        assert!(mate_within(MATE - 1, 1) && mate_within(MATE - 3, 2) && !mate_within(MATE - 3, 1));
        assert!(!mate_within(100, 5) && !mate_within(-(MATE - 1), 1));
    }

    /// A small suite end to end: a mate in two, a capture to find, an avoid
    /// move, perft counts, and lines that must be skipped, not crash.
    #[test]
    fn runs_a_suite() {
        init();
        let dir = std::env::temp_dir().join(format!("incipit-epd-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.epd");
        std::fs::write(
            &path,
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - bm Rd8+; dm 1; id \"mate1\";\n\
             r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - ; am Qe2 ; id \"quiet\";\n\
             rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - ;D1 20 ;D2 400 ;D3 8902\n\
             not a position at all\n\
             8/8/8/8/8/8/8/8 w - - ; bm e4\n\
             rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - ; bm Ke8\n",
        )
        .unwrap();
        let sum = run(path.to_str().unwrap(), Limit::Depth(6), 4, 1, 3, false);
        std::fs::remove_dir_all(&dir).unwrap();
        // Searched: mate1 (solved), quiet (solved: Qe2 is a poor move) and the
        // illegal bm Ke8 line is skipped, as are the two broken lines.
        assert_eq!(sum, Summary { searched: 2, solved: 2, skipped: 3, perft_ok: 3, perft_bad: 0 });
    }
}
