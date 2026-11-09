# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

I built and tested both, and this card compares them. Everything reported here comes from runs I actually did during the project. Where something was not tested, I say so.

## 1. Model Overview

**Model type:**
I compared both models. The rule based analyzer and the machine learning model use the same labeled dataset (`SAMPLE_POSTS` and `TRUE_LABELS` in `dataset.py`), but they work in very different ways.

**Intended purpose:**
Both models try to classify a short text message (like a social media post) into one of four moods: positive, negative, neutral, or mixed. This is a class project for learning how simple sentiment classifiers work and where they break. It is not meant for real-world use.

**How it works (brief):**

- **Rule based model:** I wrote the rules by hand. The text is cleaned and split into tokens, and each token is checked against a list of positive words and a list of negative words. A negation word right before a sentiment word flips its meaning. The label comes from which kinds of evidence were found.
- **ML model:** The text is turned into word counts (bag of words), and a logistic regression model learns from the 14 labeled posts which words tend to go with which label. It has no word lists and no hand-written rules.

The main difference is where the knowledge comes from. In the rule based model it comes from my word lists and rules. In the ML model it comes from patterns in the labeled examples.

## 2. Data

**Dataset description:**
The dataset has **14 posts**. Six came with the starter code and I added eight more in Phase 1. The labels are:

| Label | Count |
|---|---|
| positive | 5 |
| negative | 5 |
| neutral | 2 |
| mixed | 2 |

The eight posts I added were:

- "Lowkey having the best day ever"
- "This assignment is killing me 😭"
- "I passed!! Let's gooo 🔥"
- "I studied all night and still failed"
- "The movie was okay, I guess"
- "I'm exhausted but proud of what I accomplished"
- "I absolutely love sitting in traffic for two hours"
- "Not bad, honestly pretty decent :)"

**Labeling process:**
I chose each label by deciding what the writer was feeling overall. A post got "mixed" when it clearly expressed both a good and a bad feeling, and "neutral" when it showed no strong feeling either way. Some posts could reasonably be labeled differently:

- **"This is fine"** (labeled neutral): it could be neutral, mildly positive, or even negative, since it is also a meme about pretending things are okay.
- **"Feeling tired but kind of hopeful"** (labeled mixed): someone could call it positive, because the "but" puts the weight on hope.
- **"I absolutely love sitting in traffic for two hours"** (labeled negative): this is sarcasm. I labeled it by what the writer actually means, not by the literal words.
- **"I am not happy about this"** (labeled negative): the label is clear, but the feeling is mild enough that someone might call it neutral.

**Important characteristics of your dataset:**

- Contains slang ("lowkey", "gooo") and emojis/emoticons (😭, 🔥, `:)`)
- Includes one sarcastic post
- Includes two mixed-feeling posts
- Contains short and ambiguous messages
- Includes negation ("not happy", "not bad")

**Possible issues with the dataset:**

- It is **very small** (14 posts), so one post changes a lot.
- It is **imbalanced**: only 2 neutral and 2 mixed examples, compared to 5 each for positive and negative.
- There is only **one sarcastic example** and no examples of many other styles (long posts, questions, other slang).
- All labels were assigned by one person (me), so they reflect one person's judgment.
- Some posts are ambiguous, and those labels could be argued either way.

## 3. How the Rule Based Model Works

**Your scoring rules:**

- **Preprocessing (`preprocess()`):** the text is lowercased, then a regular expression splits it into tokens. It keeps words (including apostrophes inside words like "don't"), emoticons like `:)` and `:-(`, and emojis, and it drops other punctuation. So "Not bad, honestly :)" becomes `["not", "bad", "honestly", ":)"]`. Negation words like "not", "no", and "never" are kept. Before this change, text was only split on spaces, so "bad," did not match "bad".
- **Vocabulary:** 15 positive words and 13 negative words. The starter had 10 of each. In Phase 3 I added 5 positive words (hopeful, proud, passed, wonderful, adore) and 3 negative words (exhausted, failed, lose). I deliberately did not add ambiguous slang such as "sick", "fire", or "wicked".
- **Scoring (`score_text()`):** each positive word adds 1 and each negative word subtracts 1. Words that are not in either list are ignored.
- **Negation:** if the token right before a sentiment word is a negation word, that word's meaning is reversed. "not happy" counts as negative and "not bad" counts as positive. It only looks one token back, which keeps the rule simple and easy to explain.
- **Classification (`predict_label()`):** the label is based on the evidence, not just the number.
  - positive and negative evidence both found: **mixed**
  - positive evidence only: **positive**
  - negative evidence only: **negative**
  - no recognized sentiment words: **neutral**

  I used evidence instead of the score because a score of 0 can mean either "mixed" (a positive and a negative word cancel out) or "neutral" (nothing was found).
- **Emojis:** they are kept as tokens but are **not scored**. I did not give them any meaning.
- **Explanations (`explain()`):** it uses the same helper as the scoring, so the explanation always matches the prediction and shows which words counted, including negation.

**Strengths of this approach:**

- It is predictable and easy to inspect. For any prediction I can see exactly which words counted.
- It handled negation in the dataset: "I am not happy about this" came out negative, and "Not bad, honestly pretty decent :)" came out positive.
- It found mixed feelings when both sides used listed words. "I hate waking up early, but I love my morning coffee" came out mixed.
- It needs no training data.

**Weaknesses of this approach:**

- **Unknown words are ignored.** If the sentiment is in a word that is not on a list, the model sees nothing and answers "neutral".
- **It can't tell "no feeling" from "feeling I have no words for".** Neutral is really its default answer.
- **It can't detect sarcasm.** "I love getting stuck in traffic" came out positive.
- **Slang and emojis don't count.** "That party was sick" and "This new game is fire" came out neutral.
- It only improves when I add words by hand, and some words ("passed", "lose") depend on context.

## 4. How the ML Model Works

**Features used:**
Bag of words using `CountVectorizer` with its default settings. It builds a vocabulary from the training posts (58 words) and turns each post into a row of word counts. Word order is thrown away. By default it also lowercases text, drops punctuation, ignores one-letter words, and does not keep emojis. It does not use my preprocessing, my word lists, or my negation rule.

**Training data:**
The model trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`, all 14 posts. It uses scikit-learn's `LogisticRegression(max_iter=1000)`. The training function checks that the posts and labels are the same length before training. The stress-test sentences were never used for training.

**Training behavior:**
I did **not** run a before-and-after experiment with additional training examples. The dataset already had all 14 posts before I first ran the ML model, so I did not observe how its accuracy changes when more examples are added or when labels change.

**Strengths and weaknesses:**

- **Strength:** it learns patterns from the examples automatically, with no hand-written word lists.
- **Weakness, seen in my tests:** with only 14 posts it often relies on words that carry no sentiment. For example, it answered "negative" for "Today was amazing 🔥". The only words it recognized were "today" and "was", which come from the training post "Today was a terrible day", and it had never seen "amazing".
- **Weakness, seen in my tests:** it was not confident. On the stress tests the probability of its top label ranged from 0.35 to 0.53, so many answers were close to guesses.
- **Weakness, by design:** it ignores word order, so it has no way to handle negation. It also drops emojis. It can only use words it saw in training.

## 5. Evaluation

**How you evaluated the model:**
I looked at two different things, and they should not be mixed up.

1. **Accuracy on the 14 labeled posts in `dataset.py`.** For the ML model this is *training accuracy*, because it is scored on the same posts it learned from.
2. **A stress test of 12 sentences** I wrote to break the models: 3 sarcastic, 3 slang, 3 with emojis, and 3 mixed or ambiguous. I scored the rule based analyzer and the ML model on them. The ML model was trained only on the original 14 posts.

| Measurement | Model | Result |
|---|---|---|
| Dataset accuracy, before vocabulary improvements | Rule based | 7/14 (50%) |
| Dataset accuracy, after vocabulary improvements | Rule based | 11/14 (79%) |
| Training accuracy on the 14 posts | ML | 14/14 (100%) |
| Stress test (10 scored sentences) | Rule based | 4/10 (40%) |
| Stress test (10 scored sentences) | ML | 3/10 (30%) |

**How to read these numbers:**

- **The ML model's 100% is training accuracy.** It shows that the model can memorize the 14 posts it was shown. It is **not** evidence that the model works on unseen text. The stress test, where it got 3 of 10, is the better evidence about new sentences, and it is much lower.
- **The rule based 79% is also not a clean score.** I picked the eight new words after seeing which dataset posts failed, and the four dataset posts it fixed all contain those words. So that number partly measures how well I fitted the word lists to this dataset.
- **I can't fairly compare 79% and 100%.** They measure different things and neither is on held-out data.
- **The stress-test scores have several limits:**
  - There are only 10 scored sentences, so a single sentence changes a score by 10 percentage points. The 4/10 vs. 3/10 difference is too small to say that one model is better.
  - The expected labels were my own judgment calls (for example, that the sarcastic sentences should count as negative).
  - Two sentences were ambiguous, so I left them out of the score: "I'm fine 🙂" (neutral or mildly positive?) and "I passed my exam 😭" (tears of joy or distress?).
  - The rule based vocabulary was tuned on examples that overlap with the stress test. Words such as "exhausted", "proud", and "passed" came from the dataset and appear in stress-test sentences, so the rule based result is not a blind test.
  - The ML model has also seen some of the same words in training, so its results are not blind either.

**Examples of correct predictions:**

- **"I am not happy about this"** (rule based, negative): the negation rule flipped "happy" to negative evidence. A model without negation would see a positive word.
- **"I'm exhausted but proud of myself"** (both models, mixed): the rule based model found "exhausted" (negative) and "proud" (positive). The ML model also said mixed, but this is likely memorization, because a very similar post ("I'm exhausted but proud of what I accomplished") is in its training data.
- **"I hate waking up early, but I love my morning coffee"** (rule based, mixed): both "hate" and "love" are in the word lists, so both kinds of evidence were found.
- **"I love getting stuck in traffic"** (ML, negative): the right answer, but probably by memorization. "traffic" and "in" appear in the training post that I labeled negative, so this does not show that the model understood sarcasm.

**Examples of incorrect predictions:**

- **"I love getting stuck in traffic."** The rule based model said positive. It saw "love" and ignored "stuck" and "traffic". It can't detect sarcasm.
- **"That party was sick."** The rule based model said neutral because "sick" is not in the lists. The ML model said negative, using only the filler word "was". Both were wrong, for different reasons.
- **"Today was amazing 🔥"** The rule based model got this right, but the ML model said negative, as described above.
- **"I just adore having three exams tomorrow."** Both models said positive. After I added "adore" to the vocabulary, this sarcastic sentence changed from wrongly neutral to wrongly positive, so the model became confidently wrong.

**How the failures differed:**
The rule based model fails when a sentiment word is not on its lists. In that case it says neutral. The ML model fails because it has too little data. In that case it picks a label from irrelevant words. They agreed on only 3 of the 12 stress sentences, and one of those (the "adore" sentence) was a shared mistake. In the 10 scored sentences, the rule based model got #9, #10, #11, and #12 right and the ML model got #1, #2, and #10 right. The rule based neutral on "It was an okay day, nothing special" is correct, but only because it found no words at all. It did not understand "okay" or "nothing special".

## 6. Limitations

- **Sarcasm:** neither model can detect it. The rule based model failed all three sarcastic stress sentences. The ML model got two of them "right", but one (the traffic sentence) was most likely memorized from a training post and the other had almost no evidence behind it, so I don't count that as detecting sarcasm. I did not try to implement sarcasm detection.
- **Slang:** "sick", "fire", and "wicked" were not recognized by either model. I kept them out of the word lists on purpose, because words like "sick" can mean opposite things depending on context. Slang is also a moving target, so any fixed list will fall behind.
- **Emojis:** the rule based model keeps emojis as tokens but does not score them, and the ML model's default tokenizer drops them. In the stress test, 😭 and 🔥 had no effect on either model, so a post where an emoji is the main signal would be missed.
- **Negation:** the rule based model only handles a negation word directly before a sentiment word. Longer patterns like "not very happy" would not be caught by that rule, though I did not test them. The ML model has no negation handling because it ignores word order.
- **Unknown words:** this is the biggest practical problem. Any sentiment word that is not in the lists (rule based) or not in the training posts (ML) is invisible to the model. The rule based result for those cases is "neutral", which can be wrong.
- **Small dataset:** 14 labeled posts is not enough to learn from or to measure with. The ML model's vocabulary is only 58 words.
- **Class imbalance:** there are only 2 neutral and 2 mixed examples, compared to 5 positive and 5 negative.
- **Overfitting:** the ML model got 100% on its training posts but only 3 of 10 on the stress test, and I saw it rely on irrelevant words like "was". This is what overfitting to a tiny dataset looks like.
- **No held-out test set.** I never set aside data the models hadn't seen, so I can't give a trustworthy accuracy number for either model on new text.
- **Short English text only.** I did not test longer posts, other languages, or different platforms, so I can't say how the models behave on them.

### Language-style comparison (overt bias investigation)

I define overt bias here as systematic failures associated with the language style of the person writing the text. To look for it, I tested the rule based analyzer on five pairs of sentences that express the same feeling in an informal and a more conventional style:

| Pair | Informal | Result | Conventional | Result |
|---|---|---|---|---|
| 1 | "That party was sick." | neutral (missed) | "That party was amazing." | positive |
| 2 | "This game is fire." | neutral (missed) | "This game is excellent." | neutral (missed) |
| 3 | "Lowkey having the best day ever." | neutral (missed) | "I'm having a wonderful day." | positive |
| 4 | "I passed my exam 😭" | positive | "I passed my exam successfully." | positive |
| 5 | "I'm exhausted but proud of myself." | mixed | "I'm tired but happy with my accomplishments." | mixed |

What I saw:

- In pairs 1 and 3, the informal sentence failed and the conventional one worked, even though they express the same feeling.
- In pair 2, **both** sentences failed, because "excellent" is not in the word lists either. So the cause is missing vocabulary, not only informal style. In pair 3 the word "best" is also ordinary and still failed.
- In pairs 4 and 5, both versions got the same label, but pair 4 worked only because of the word "passed". The emoji did nothing. Pair 5 worked partly because I had added "exhausted" and "proud" to the lists.

What I can and can't conclude:

- This is **consistent with** a language-style coverage problem. The word lists I wrote use standard vocabulary, so informal expressions outside those lists get no score.
- It is only five pairs, written by me, and tested only on the rule based model. I did not run the pairs through the ML model.
- It does **not** show that any demographic group, dialect, or community is affected. I have no information about who writes in these styles or who would be harmed, and slang use does not map cleanly onto any group. Checking that would need real text from real writers.

### Covert compliance (observations from AI-assisted development)

**This was not a formal covert-compliance experiment.** I did not design a test for it. These are observations from using an AI assistant to build the project, and they are only what I noticed:

- When adding vocabulary, the assistant pointed out that "passed" and "lose" depend on context (for example "passed the salt" is neutral and "lose weight" can be good). It also flagged that "sick", "fire", and "wicked" are ambiguous slang, which supported my decision not to add them.
- After I added "adore", the assistant warned that the sarcastic sentence "I just adore having three exams tomorrow" went from wrongly neutral to confidently wrong, so vocabulary additions do not fix sarcasm and can make sarcastic text look more clearly positive.
- The assistant pointed out that the 79% rule based accuracy is partly fitted to the dataset, because I chose the new words after looking at which dataset posts failed.

I include these because they show places where the assistant's help could have made my results look better than they were, and where it flagged problems instead. I can't draw wider conclusions from them.

## 7. Ethical Considerations

These points are my reasoning about possible impacts. Apart from the stress-test examples above, I did not test them.

- **Misclassifying distress.** In my tests, sarcastic or slang-heavy messages were often labeled neutral or positive. If a tool like this were used to notice when someone is struggling, it could miss real distress or label it as fine. A message like "I'm fine 🙂" is ambiguous even for a person. A model that treats missing words as "neutral" is not safe for that kind of decision.
- **Language-style coverage.** The word lists reflect the vocabulary I thought of, and informal phrasing was misread more often than conventional phrasing in my pairs. If people who write in informal styles were analyzed by a tool like this, they might be understood less well. I haven't shown that this affects any particular group, but it is a risk to check with real data before using something like this on real users.
- **Overconfidence.** The ML model reports 100% on its training data, and a number like that could be mistaken for real reliability. Reporting training accuracy as if it were test accuracy would be misleading.
- **Privacy.** Mood analysis is done on personal messages. A real application would need consent, a clear explanation of what is analyzed and stored, and careful handling of the data. My project only uses short example posts I wrote.
- **Labels reflect one person's judgment.** All labels were assigned by me, and people could reasonably disagree on several of them. That affects both models and any accuracy numbers.

## 8. Ideas for Improvement

These are proposals only. I have not implemented or tested any of them.

- **Add a real test set.** Keep some labeled posts out of training and tuning so that I can measure accuracy on text the models haven't seen.
- **Add more labeled data**, especially neutral and mixed posts, sarcasm, slang, and emojis, and have more than one person label them.
- **Handle emojis and emoticons** by scoring them, with care for ambiguous ones such as 😭.
- **Improve slang handling** using context instead of fixed word lists, since words like "sick" can mean opposite things.
- **Expand negation handling** beyond one word back.
- **Try TF-IDF** instead of plain counts in the ML model, and try cross-validation to get a more honest accuracy with small data.
- **Try a larger model** such as a small neural network or transformer, which might handle context better, though I haven't tested whether it would.
- **Consider a "not enough information" outcome** for the rule based model, so that unknown words are not reported as neutral.
- **Repeat the language-style comparison with more pairs**, on both models, and with sentences from several writers.
