# Incipit trainer

A minimal CPU-based NNUE trainer, used to bootstrap Incipit's first networks from its own self-play data. It is multi-threaded and uses only the Rust standard library.

**Legacy.** Since October 2026 networks are trained with [Bullet](https://github.com/jw1912/bullet) (the `train` crate in the private `incipit-research` repo); this crate is kept for the bootstrap recipe and supports only the feature-transformer-plus-output-layer nets (no hidden layer, king buckets 1 or 8). Note that `--kb 8` mirrors the board whatever `--mirror` says, so convert such a net with `datatools net raw ... --mirror`.

## Usage

```
cargo run --release -- <out> <hidden> <epochs> <lr> <lambda> <threads> [options] <data files...>
```

Options: `--cosine`, `--mirror`, `--kb <n>`, `--valfile <file>`, `--init <file>`.
