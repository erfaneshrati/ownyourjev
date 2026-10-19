"""Labeling guidance for emotion, written from the 20 seeds and labeler errors on dev half A only."""

G1 = """These texts come from a dataset whose gold label was assigned from the single FEELING WORD that the author attaches to
"feel / feeling / felt / feels" (the word or short phrase right after it, e.g. "i feel <word>", "feeling <word>", "makes me feel <word>",
"the feeling of being <word>"). The label is the emotion category of THAT word, not the overall mood of the post.
Rules:
1. Find the feeling word after feel/feeling/felt. Label the emotion category of that word.
2. Ignore negation and context: "i am not feeling shitty anymore" -> sadness (shitty); "i don't feel victimized" -> sadness;
   "im not feeling fantastic" -> joy (fantastic); "feeling rich which i dont" -> joy.
3. Other emotion words elsewhere in the text (love, hate, miss, worried, nervous...) do NOT decide the label; only the word
   attached to feel/feeling does. "i love running because i feel strong" -> joy (strong); "he makes me feel so greedy" -> anger.
4. In "i feel like <clause>", use the most emotion-bearing adjective inside the clause ("i feel like the character precious" -> joy,
   "i feel like crap" -> sadness, "i feel like i am not really that talented" -> joy (talented)).
5. Typical word -> category mapping:
   - joy: content, glad, happy, strong, rich, superior, precious, comfortable, keen, welcomed, fantastic, benevolent, invigorated,
     cool, ok, useful, excited, casual, fearless, generous, talented, important, confident, relaxed, festive, energetic, honoured,
     accepted, safe, pleased, successful, determined, hopeful, peaceful, charming, worthwhile, satisfied, cheerful, special.
   - sadness: discontent, ungrateful, numb, jaded, guilty, inadequate, dumb, foolish, isolated, victimized, shitty, crap, bad, lonely,
     homesick, unimportant, useless, ashamed, miserable, depressed, empty, hopeless, stupid, exhausted, unloved, rejected, gloomy,
     helpless, regretful, devastated, disappointed, awful, troubled, burdened, repressed, defeated, unhappy, pathetic, lost.
   - anger: greedy, grumpy, cold, irritated, pissed, agitated, rude, dissatisfied, bitter, cranky, impatient, offended, resentful,
     jealous, violent, stubborn, annoyed, frustrated, mad, hostile, insulted, wronged, selfish, spiteful, envious, petty, snobbish.
   - fear: reluctant, scared, afraid, nervous, anxious, insecure, shaky, uncomfortable, paranoid, terrified, threatened, vulnerable,
     hesitant, worried, unsure, apprehensive, restless, intimidated, frightened, tortured, distraught, pressured, weird, strange.
   - love: passionate, fond, loving, caring, tender, sympathetic, nostalgic, longing, horny, hot, romantic, delicate, gentle, supportive,
     treasured, sweet, affectionate, compassionate, beloved, naughty, loyal, devoted, liked.
   - surprise: impressed, amazed, overwhelmed, curious, shocked, stunned, dazed, surprised, funny, astonished, awed.
6. If there is no feel-word, judge the dominant emotion word of the text."""
