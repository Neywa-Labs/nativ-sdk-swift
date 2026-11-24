/// Shared EOS resolution for loaded resources and manually constructed contexts.
enum StopPolicy {
    static func tokenIDs(
        configured: Set<Int>, supplementalTokens: Set<String>, tokenizer: any Tokenizer
    ) -> Set<Int> {
        var result = configured
        if let eos = tokenizer.eosTokenId { result.insert(eos) }
        for token in supplementalTokens {
            // Unknown strings can map to UNK in some tokenizers. Multi-token
            // delimiters are handled separately by the decoded text stop filter.
            if let id = tokenizer.convertTokenToId(token), id != tokenizer.unknownTokenId {
                result.insert(id)
            }
        }
        return result
    }

    static func tokenIDs(context: ModelContext) -> Set<Int> {
        tokenIDs(
            configured: context.metadata.eosTokenIds,
            supplementalTokens: context.metadata.extraEOSTokens,
            tokenizer: context.tokenizer)
    }
}
