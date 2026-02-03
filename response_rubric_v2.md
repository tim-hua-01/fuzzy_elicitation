# Rubric for Evaluating Philosophical Answers

## Instructions for Graders

You are evaluating philosophical answers for quality of argumentation and exposition. **Be a calibrated grader.** If an answer genuinely merits a high score, assign it. If it has clear deficiencies, score accordingly. The scale is designed so that competent, solid work scores in the 3-5 range. Scores of 6 and above are reserved for answers that go beyond competent execution to offer genuine novelty, elegance, or insight that would be notable even to experts. An 8 is rare and represents work that could contribute to the published literature.

**Calibration guidance:**
- 0-2: Deficient (missing, confused, or badly executed)
- 3-4: Competent (does the job, standard moves, no major errors)
- 5: Strong (well-executed, hits all expected points skillfully)
- 6: Excellent (strong execution plus notable efficiency or a non-obvious insight)
- 7: Outstanding (genuine novelty—an argument, distinction, or synthesis that would interest experts)
- 8: Exceptional (novel, elegant, and generative; could contribute to published literature)

---

## Criteria

### 1. Thesis Clarity

Does the answer announce a clear position, and is that position well-formulated?

| Score | Description |
|-------|-------------|
| 0 | No discernible thesis; the answer is a collection of observations without a unifying claim |
| 1 | Thesis is buried, vague, or only implicit; reader must work to identify the author's position |
| 2 | Thesis is present but delayed, hedged, or confusingly stated |
| 3 | Thesis is stated clearly, though perhaps not optimally positioned; position is unambiguous |
| 4 | Thesis is stated directly and early; the reader knows exactly what will be argued |
| 5 | Thesis is crisp, bold, and well-positioned; frames the entire answer effectively |
| 6 | Thesis is expertly formulated and the framing itself reveals something non-obvious about the problem |
| 7 | The thesis formulation is itself a contribution—it carves the problem at joints others have missed |
| 8 | The thesis reframes the debate; even someone who disagreed would recognize this as the right way to pose the question |

---

### 2. Argumentative Soundness

Is the reasoning valid? Are the moves justified? Are there logical gaps, unsupported assertions, or fallacies?

| Score | Description |
|-------|-------------|
| 0 | Argument is incoherent, self-contradictory, or riddled with fallacies |
| 1 | Major logical gaps; key moves are unjustified; conclusion doesn't follow from premises |
| 2 | Some valid reasoning but significant gaps or unsupported leaps remain |
| 3 | Argument is basically sound; reasoning is followable though some moves could be better supported |
| 4 | Argument is sound with clearly justified moves; minor gaps if any |
| 5 | Argument is valid and well-supported throughout; logical structure is clear and tight |
| 6 | Argument is sound and efficient—no wasted moves, no unnecessary premises, well-constructed |
| 7 | Argument has an elegant structure; the logical architecture itself is illuminating |
| 8 | Argument feels inevitable; it's hard to imagine a more economical or compelling path to the conclusion |

---

### 3. Dialectical Engagement

Does the answer engage fairly with opposing views and consider how critics might respond?

| Score | Description |
|-------|-------------|
| 0 | Ignores opposing views entirely; no acknowledgment of potential objections |
| 1 | Strawmans opposing views or gestures at objections without engaging seriously |
| 2 | Presents opposing views but weakly; handles obvious objections superficially |
| 3 | Presents opposing views fairly and addresses reasonable objections adequately |
| 4 | Engages charitably with opponents and anticipates non-obvious objections |
| 5 | Steelmans opposing views and responds to strong objections convincingly |
| 6 | Dialectical engagement reveals something new; objections are turned into opportunities to deepen the argument |
| 7 | Engages with opponents in a way that clarifies the true source of disagreement; both sides would recognize the characterization as fair |
| 8 | Transforms the dialectic; shows how apparently opposed positions can be reconciled or reveals that the real disagreement lies elsewhere than previously thought |

---

### 4. Precision in Distinctions

Does the answer carefully distinguish between superficially similar concepts, and does it use terms consistently?

| Score | Description |
|-------|-------------|
| 0 | Conflates importantly different concepts; key terms are used ambiguously or inconsistently |
| 1 | Some awareness of relevant distinctions but fails to maintain them |
| 2 | Makes some necessary distinctions but misses others or draws them imprecisely |
| 3 | Draws the main relevant distinctions clearly enough to avoid confusion |
| 4 | Identifies and maintains important distinctions consistently throughout |
| 5 | Draws precise distinctions, including some that are non-obvious, and deploys them effectively |
| 6 | Distinctions are not merely correct but revealing; they do real work in advancing the argument |
| 7 | Introduces a distinction that reframes the problem—once drawn, one sees the issue differently |
| 8 | Distinction is a genuine contribution; it could become a standard tool in discussions of this topic |

---

### 5. Substantive Contribution

Does the answer offer a positive proposal, take genuine intellectual risks, and add something beyond competent assembly of known considerations?

| Score | Description |
|-------|-------------|
| 0 | Purely negative or entirely noncommittal; no constructive proposal; refuses to take a stance |
| 1 | Nominal position that is hedged into meaninglessness; gesture at a positive view with no development |
| 2 | Offers a position but it is thin, obvious, or the safe consensus view defended without real argument |
| 3 | Defends a clear position with reasonable argument; provides an adequate positive proposal |
| 4 | Develops a substantive proposal that illuminates the problem; takes a stance some would dispute |
| 5 | Advances a well-developed positive account that resolves tensions or challenges received wisdom |
| 6 | Contributes something beyond the expected—a new angle, an unexpected connection, a surprising implication |
| 7 | Offers a genuinely novel argument, synthesis, or approach that would interest experts in the area |
| 8 | Makes a contribution that could advance the scholarly conversation; opens new questions or suggests a research program |

---

### 6. Quality of Examples

Does the answer use concrete examples that do genuine argumentative work?

| Score | Description |
|-------|-------------|
| 0 | No examples, or examples that are confusing, irrelevant, or counterproductive |
| 1 | Examples are present but merely restate the abstract point; they do no real work |
| 2 | Examples illustrate the point but are obvious or unilluminating |
| 3 | Examples are helpful and clarify the argument |
| 4 | Examples are well-chosen and reveal something not obvious from the abstract statement alone |
| 5 | Examples are apt and do significant argumentative work; they strengthen the case materially |
| 6 | Examples are striking or memorable; they make the argument more compelling than abstract reasoning alone |
| 7 | An example reframes how one thinks about the problem; it has power beyond its immediate use |
| 8 | An example is itself a contribution—the kind that might become a standard reference point in discussions of this topic (like Gettier cases, trolley problems, Mary's room) |

---

## Output Format

Return your evaluation as a JSON object with no additional commentary, explanation, or text outside the JSON. Use the following structure:

```json
{
  "thesis_clarity": 0,
  "argumentative_soundness": 0,
  "dialectical_engagement": 0,
  "precision_distinctions": 0,
  "substantive_contribution": 0,
  "example_quality": 0,
  "total": 0
}
```

Replace each `0` with the appropriate score (0-8). The `total` field should be the sum of all six individual scores (maximum 48).

**Do not include any text before or after the JSON object.**