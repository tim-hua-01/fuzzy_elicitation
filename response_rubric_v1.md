# Rubric for Evaluating Philosophical Answers

## Instructions for Graders

You are evaluating philosophical answers for quality of argumentation and exposition. For each criterion, provide a score from 0-5.

---

## Criteria

### 1. Thesis Clarity and Directness

Does the answer announce a clear position early, or does it meander before revealing its stance?

| Score | Description |
|-------|-------------|
| 0 | No discernible thesis; the answer is a collection of observations without a unifying claim |
| 1 | Thesis is buried, vague, or only implicit; reader must work to identify the author's position |
| 2 | Thesis is present but delayed or somewhat hedged; position becomes clear by mid-answer |
| 3 | Thesis is stated clearly, though perhaps not in the opening sentences; position is unambiguous |
| 4 | Thesis is stated directly and early; the reader knows exactly what will be argued from the start |
| 5 | Thesis is crisp, bold, and presented in the opening sentences; frames the entire answer effectively |

---

### 2. Charitable Engagement with Opposing Views

Does the answer engage with the strongest version of positions it criticizes, or does it attack strawmen?

| Score | Description |
|-------|-------------|
| 0 | Ignores or badly misrepresents opposing views; clear strawmanning |
| 1 | Acknowledges opposing views exist but presents them weakly or dismissively |
| 2 | Presents opposing views but misses their strongest formulations or motivations |
| 3 | Presents opposing views fairly, though without fully exploring their appeal |
| 4 | Engages charitably with opposing views; shows why reasonable people might hold them |
| 5 | Steelmans opposing views—presents them more compellingly than their proponents might—before responding |

---

### 3. Anticipation and Handling of Objections

Does the answer consider how critics might respond and address those responses?

| Score | Description |
|-------|-------------|
| 0 | No consideration of potential objections whatsoever |
| 1 | Gestures at possible objections but does not seriously engage with them |
| 2 | Considers one or two obvious objections but misses important ones or handles them superficially |
| 3 | Anticipates reasonable objections and provides adequate responses |
| 4 | Anticipates strong objections, including non-obvious ones, and responds convincingly |
| 5 | Systematically considers the most challenging objections and turns them into opportunities to deepen the argument |

---

### 4. Quality of Examples and Illustrations

Does the answer use concrete examples that do genuine argumentative work, or are examples absent, irrelevant, or merely decorative?

| Score | Description |
|-------|-------------|
| 0 | No examples, or examples that are confusing/irrelevant |
| 1 | Examples are present but do little argumentative work; they merely restate the point abstractly made |
| 2 | Examples illustrate the point but are obvious or unilluminating |
| 3 | Examples are helpful and clarify the argument, though somewhat predictable |
| 4 | Examples are well-chosen and do real argumentative work; they reveal something not obvious from the abstract statement |
| 5 | Examples are striking, memorable, and essential to the argument; they make the case more convincing than abstract reasoning alone could |

---

### 5. Precision in Distinctions

Does the answer carefully distinguish between superficially similar concepts, or does it conflate importantly different ideas?

| Score | Description |
|-------|-------------|
| 0 | Conflates importantly different concepts; key terms are used ambiguously throughout |
| 1 | Acknowledges some distinctions but fails to maintain them consistently |
| 2 | Makes some necessary distinctions but misses others or draws them imprecisely |
| 3 | Draws the main relevant distinctions clearly enough to avoid confusion |
| 4 | Identifies and carefully maintains important distinctions, including some that are non-obvious |
| 5 | Introduces precise distinctions that reframe the problem and reveal previously hidden structure |

---

### 6. Constructive Contribution

Does the answer offer a positive proposal or solution, or does it merely critique without building?

| Score | Description |
|-------|-------------|
| 0 | Purely negative; tears down without offering any alternative or constructive insight |
| 1 | Gesture toward a positive view but it remains undeveloped or unclear |
| 2 | Offers a positive proposal but it is thin, obvious, or inadequately defended |
| 3 | Provides a reasonable positive proposal that addresses the main question |
| 4 | Develops a substantive positive proposal that illuminates the problem |
| 5 | Advances a novel, well-developed positive account that resolves tensions or explains phenomena in a unified way |

---

### 7. Argumentative Risk-Taking

Does the answer defend a substantive, potentially controversial position, or does it retreat into safe, noncommittal observations?

| Score | Description |
|-------|-------------|
| 0 | Completely noncommittal; endless "on the other hand" without taking a stance |
| 1 | Takes a nominal position but hedges it into meaninglessness |
| 2 | Defends a position but chooses the safe/obvious/consensus view without serious argument |
| 3 | Defends a clear position that some would dispute, with reasonable argument |
| 4 | Defends a substantive position that challenges common assumptions or received wisdom |
| 5 | Defends a bold, counterintuitive position with compelling arguments; takes genuine intellectual risks |

---

### 8. Problem Reframing

Does the answer reveal that the question was posed in a misleading way, or identify hidden assumptions that change how we should approach the problem?

| Score | Description |
|-------|-------------|
| 0 | Accepts the framing of the question uncritically; no examination of presuppositions |
| 1 | Notes minor issues with framing but proceeds within the given framework |
| 2 | Identifies some assumptions in the question but doesn't use this to advance the argument significantly |
| 3 | Challenges the framing in a way that somewhat clarifies the issues |
| 4 | Reframes the problem productively, revealing that the original question obscured important considerations |
| 5 | Fundamentally reconceives the problem; shows that the apparent difficulty dissolves or transforms under proper analysis |

---

### 9. Explanatory Unification

Does the answer explain multiple phenomena or considerations through a single principle or framework, or does it treat each point in isolation?

| Score | Description |
|-------|-------------|
| 0 | Treats points as a disconnected list; no unifying thread |
| 1 | Attempts unification but the connecting principle is forced or unconvincing |
| 2 | Some thematic coherence but individual points are not genuinely unified |
| 3 | Offers a framework that connects the main considerations reasonably well |
| 4 | Unifies apparently disparate considerations under a single explanatory principle |
| 5 | Achieves elegant explanatory unification; a single insight illuminates multiple phenomena that seemed unrelated |

---

### 10. Appropriate Scope and Honesty about Limitations

Does the answer accurately represent what it has and hasn't shown, or does it overclaim or underclaim?

| Score | Description |
|-------|-------------|
| 0 | Wildly overclaims (claims to have proven what it hasn't) or underclaims (fails to recognize the significance of its own arguments) |
| 1 | Significant mismatch between what is argued and what is claimed to have been established |
| 2 | Some overclaiming or failure to acknowledge important limitations |
| 3 | Generally accurate about scope, though could be more precise about what remains open |
| 4 | Clear and honest about what has been shown and what limitations remain |
| 5 | Exemplary intellectual honesty; precisely delineates what has been established, what is suggestive but not proven, and what remains unaddressed |

---

## Output format

Return your evaluation as a JSON object with no additional commentary, explanation, or text outside the JSON. Use the following structure:
```json
{
  "thesis_clarity": 0,
  "charitable_engagement": 0,
  "objection_handling": 0,
  "example_quality": 0,
  "precision_distinctions": 0,
  "constructive_contribution": 0,
  "argumentative_risk": 0,
  "problem_reframing": 0,
  "explanatory_unification": 0,
  "scope_honesty": 0,
  "total": 0
}
```