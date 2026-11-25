// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// divora-core/src/tts/tokens.rs at 62ed6a5. See desktop/LICENSES/DivoraVoiceMod.txt.
//! Kokoro tokenization: phonemes → model input ids, + text chunking.
//!
//! The exact 178-symbol vocab is loaded from the Kokoro ONNX distribution's
//! `config.json` at runtime (`parse_vocab`), so we never hard-code (and risk
//! mis-transcribing) the high-Unicode IPA table. These functions are the
//! pure mapping + chunking logic and are unit-tested with a synthetic vocab.

use std::collections::HashMap;
use std::hash::BuildHasher;

/// Kokoro's max phoneme sequence length (from its `config.json`).
pub const MAX_PHONEME_LENGTH: usize = 510;

/// Map a phoneme string to Kokoro input ids, framed with the pad token (`0`)
/// at both ends (the kokoro-onnx convention). Phonemes absent from the vocab
/// are skipped; the sequence is truncated to `MAX_PHONEME_LENGTH` phonemes.
#[must_use]
pub fn phonemes_to_ids<S: BuildHasher>(phonemes: &str, vocab: &HashMap<char, i64, S>) -> Vec<i64> {
    let mut ids = Vec::with_capacity(phonemes.chars().count() + 2);
    ids.push(0); // leading pad
    for ch in phonemes.chars().take(MAX_PHONEME_LENGTH) {
        if let Some(&id) = vocab.get(&ch) {
            ids.push(id);
        }
    }
    ids.push(0); // trailing pad
    ids
}

/// Split input text into chunks that stay under the model's limit once
/// phonemized. Splits on sentence-ending punctuation, then hard-wraps any
/// over-long run at word boundaries, so long passages synthesize as several
/// concatenated segments. Uses character count as a conservative proxy for
/// phoneme count (phonemes ≈ chars for the alphabets we target).
#[must_use]
pub fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() || max_chars == 0 {
        return Vec::new();
    }

    // Break after sentence-ending punctuation / newlines, keeping it attached.
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in trimmed.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?' | '…' | '\n') {
            let s = cur.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }

    // Hard-wrap any sentence longer than `max_chars` at word boundaries.
    let mut out: Vec<String> = Vec::new();
    for s in sentences {
        if s.chars().count() <= max_chars {
            out.push(s);
            continue;
        }
        let mut chunk = String::new();
        for word in s.split_whitespace() {
            let sep = usize::from(!chunk.is_empty());
            if !chunk.is_empty() && chunk.chars().count() + sep + word.chars().count() > max_chars {
                out.push(std::mem::take(&mut chunk));
            }
            if !chunk.is_empty() {
                chunk.push(' ');
            }
            chunk.push_str(word);
        }
        if !chunk.is_empty() {
            out.push(chunk);
        }
    }
    out
}

/// Parse the `vocab` object from a Kokoro `config.json` into a char→id map.
/// Multi-character keys (if any) are ignored — Kokoro tokens are single chars.
pub fn parse_vocab(config_json: &str) -> Result<HashMap<char, i64>, serde_json::Error> {
    #[derive(serde::Deserialize)]
    struct Config {
        vocab: HashMap<String, i64>,
    }
    let cfg: Config = serde_json::from_str(config_json)?;
    let mut out = HashMap::with_capacity(cfg.vocab.len());
    for (k, v) in cfg.vocab {
        let mut chars = k.chars();
        if let (Some(ch), None) = (chars.next(), chars.next()) {
            out.insert(ch, v);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> HashMap<char, i64> {
        [
            ('h', 50),
            ('ɛ', 86),
            ('l', 54),
            ('o', 57),
            ('ʊ', 135),
            ('ˈ', 156),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn phonemes_to_ids_frames_with_pad_and_skips_unknown() {
        // "hɛlˈoʊ" plus an unknown 'Z' that must be skipped.
        assert_eq!(
            phonemes_to_ids("hɛlˈoʊZ", &vocab()),
            vec![0, 50, 86, 54, 156, 57, 135, 0],
        );
    }

    #[test]
    fn phonemes_to_ids_empty_is_just_pads() {
        assert_eq!(phonemes_to_ids("", &HashMap::new()), vec![0, 0]);
    }

    #[test]
    fn phonemes_to_ids_truncates_to_max() {
        let v: HashMap<char, i64> = [('a', 43)].into_iter().collect();
        let long = "a".repeat(MAX_PHONEME_LENGTH + 50);
        let ids = phonemes_to_ids(&long, &v);
        // leading + MAX phonemes + trailing pad.
        assert_eq!(ids.len(), MAX_PHONEME_LENGTH + 2);
    }

    #[test]
    fn split_text_breaks_on_sentences() {
        assert_eq!(
            split_text("Hello there. How are you?  Fine!", 100),
            vec!["Hello there.", "How are you?", "Fine!"],
        );
    }

    #[test]
    fn split_text_hard_wraps_long_runs_at_words() {
        let parts = split_text("one two three four five", 9);
        for p in &parts {
            assert!(p.chars().count() <= 9, "chunk too long: {p:?}");
        }
        assert_eq!(parts.join(" "), "one two three four five");
    }

    #[test]
    fn split_text_empty_is_empty() {
        assert_eq!(split_text("   ", 100), Vec::<String>::new());
        assert_eq!(split_text("hi", 0), Vec::<String>::new());
    }

    #[test]
    fn parse_vocab_reads_single_char_keys() {
        let json = r#"{ "vocab": { "h": 50, "o": 57, "ab": 99 } }"#;
        let v = parse_vocab(json).unwrap();
        assert_eq!(v.get(&'h'), Some(&50));
        assert_eq!(v.get(&'o'), Some(&57));
        assert_eq!(v.len(), 2); // multi-char "ab" skipped
    }
}
