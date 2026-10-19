G1 = """These summaries were produced by summarization systems for CNN/DailyMail news articles and rated by expert annotators.
Many summaries are lower-cased and tokenized (e.g. "it 's", "-lrb-" for "(", "`` quote ''"): that is a display artifact of the dataset. Do not penalize it under any dimension.
Use the full range of levels and judge each dimension independently.

coherence: rate the summary as a whole text. Excellent = reads like a well-organized mini news story: opens with the main event, each sentence follows from the previous one. Good = mostly organized, one awkward jump. Fair = a list of related facts with weak connections or one confusing reference. Poor = disorganized: opens with a minor detail or a dangling reference ("he", "the agency"), jumps between topics, repeats itself, or stops mid-sentence. Very poor = unrelated sentences thrown together, hard to tell what the story is.

consistency: compare every claim to source_document. Most summaries copy or paraphrase the source faithfully: those are Excellent (4). Lower ONLY for unsupported content: Good = one tiny imprecision; Fair = one clearly unsupported or distorted detail; Poor = a wrong key fact (wrong person, number, outcome, misattributed quote); Very poor = several fabricated facts or a story the source does not tell.

fluency: rate sentence-level writing quality, ignoring the lower-casing/tokenization artifacts. Most extracted sentences are Excellent (4). Lower for: duplicated words ("exam exam"), missing words, garbled or fused sentences, fragments, a sentence cut off mid-way, nonsensical word choices. Good = one minor glitch; Fair = a clearly ungrammatical sentence; Poor = several; Very poor = mostly unreadable.

relevance: does the summary convey the main point of the article and its most important facts, without minor details or repetition? Excellent = captures the central story and key facts concisely. Good = central story plus some less important detail, or misses one key fact. Fair = partially on topic: important points missing or diluted by peripheral details. Poor = mostly peripheral details, the main point is barely present. Very poor = misses the main point entirely.

Give a probability over levels that reflects your uncertainty (expert scores are averages of 3 annotators)."""

GUIDANCE = {
    "plain": {"text": None, "rich": False},
    "plain_seeds": {"text": None, "rich": False, "seeds": True},
    "g1": {"text": G1, "rich": True},
    "g1_seeds": {"text": G1, "rich": True, "seeds": True},
}

COT = "\n\nFirst write a key \"analysis\" (2-4 short sentences: main point of the article; any unsupported or wrong facts in the summary; organization; grammar), then the four score distributions. JSON keys order: analysis, coherence, consistency, fluency, relevance."
GUIDANCE["g1_cot"] = {"text": G1 + COT, "rich": True, "max_tokens": 600}
