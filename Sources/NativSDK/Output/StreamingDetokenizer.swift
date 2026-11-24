// Copyright © 2024 Apple Inc.

struct StreamingDetokenizer: IteratorProtocol {
    let tokenizer: any Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokenIds: segmentTokens)
        } else {
            segment = ""
        }
    }

    mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokenIds: segmentTokens)

        // Emit the part of `newSegment` beyond its common prefix with the text
        // already emitted for this segment. A character-count suffix
        // (`newSegment.suffix(newSegment.count - segment.count)`) assumes
        // `decode` is append-only, but SentencePiece decoding is not: appending
        // a token can rewrite earlier whitespace. For example, with Gemma's
        // tokenizer `decode([\n, "  "]) == "\n  "` while
        // `decode([\n, "  ", ","]) == "\n ,"` — a space collapses. The
        // count-based suffix then computes `suffix(0)` and silently drops the
        // newly decoded character, including structural JSON commas, which
        // corrupts guided-generation output into unparseable JSON.
        //
        // This recovers correctly as long as the already-emitted prefix is
        // stable — i.e. the rewrite happens at or after the divergence point.
        // Real SentencePiece decoding only collapses whitespace at the append
        // boundary, so that holds; a decode that rewrote an already-emitted
        // interior character would re-emit the tail (no worse than the old
        // behavior, and not observed in practice).
        //
        // The prefix is measured in Unicode scalars, not `Character`s. A
        // `Character` is a grapheme cluster, and a token routinely appends a
        // combining scalar to the cluster the previous token ended: a
        // variation selector (`🏳` then U+FE0F), a zero-width joiner, a
        // combining accent (`e` then U+0301). Compared as `Character`s the
        // old text's last cluster no longer equals the merged cluster, so a
        // `Character`-level prefix (`commonPrefix(with:)`) stops one cluster
        // early and the whole merged cluster is emitted a second time — the
        // consumer receives `🏳🏳️`, `''️`, `eé`. Scalars compare exactly (no
        // canonical equivalence), and the emitted tail is precisely the
        // scalars the new token added.
        let common = zip(newSegment.unicodeScalars, segment.unicodeScalars)
            .prefix { $0 == $1 }.count
        let new = String(newSegment.unicodeScalars.dropFirst(common))

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if new.last == "\u{fffd}" {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return new
    }
}
