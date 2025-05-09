CLEAN_TEXT_PROMPT = """
You are a helpful assistant. I will give you a story parsed from the internet, and you will return a cleaned version of the story. Remove all HTML tags, malformed 
parses, and other non-text content.

Here is the story:
<story>
{story}
</story>

Return the cleaned story, nothing else. Don't say anything else.
"""

SENTENCIZE_PROMPT = """
You are a helpful assistant that will take a story and break it into sentences. 
I will give you a story and you will return a python list of sentences. Just return the list, nothing else.
Directly copy ALL of the sentences from the story into the list. Don't add any other text.

Here is the story:
<story>
{story}
</story>

Sentences:
"""

TEST_PROMPT = """
You are a helpful assistant. This prompt is a test. You will take a sentence and count the number of words in it.
Just return the number, nothing else.

Here is the sentence:
<sentence>
{sentence}
</sentence>

Number of words:
"""

LABELING_PROMPT = """
You are a helpful assistant. I will give you a sentence from a news article and you will label it with one of the following discourse tags:

<definitions>
1.  **Lede**: The opening sentence or sentences of a story that grabs the reader's attention and highlights the most important and newsworthy aspects, usually including the "who, what, when, where, why and how" elements of the story in succinct language.  
2.  **Nut Graf**: Context for the story: Why the audience cares, why the topic matters, and other crucial details relevant to the story.  
3.  **Background Information**: Contextual information related to the main topic of the story.  
4.  **Attribution**: Person, organization, or document that provides information.  
5.  **Evidence**: Information that supports the main point.  
6.  **Quote**: Exact words from someone you interviewed or documents you reviewed.  
7.  **Counterargument**: A crucial part of the story that displays evidence that the reporter has considered and consulted experts on possible rebuttals to the story's main topic or contention.  
8.  **Transition**: A sentence that indicates a new idea or topic is being introduced.  
9.  **Supporting Detail**: More information about an idea or topic that is intended to help the reader take action, deepen their understanding of the information, or draw a conclusion.  
10. **Source Opinion**: An opinionated statement from an individual who spoke directly or indirectly to a reporter. This label should be applied instead of "quote" if the quote from the source expresses a clear opinion.  
11. **Author Point of View**: An opinion expressed by the author that isn't backed up by evidence in the story and/or the reporter's expertise.  
12. **Analysis**: Where a reporter adds context to the story based on reporting and other knowledge drawn from expertise on the topic developed outside of directly reporting the story.  
13. **Color**: Details that add vivid and specific descriptions and contribute to the story's narrative, helping to paint a picture specific to this time, place, or situation—beyond "just the facts."
14. **Other**: Any other label that doesn't fit into the other 13.
</definitions>

Return the label and a justification for why you chose the label and not any other label. Return the label and justification in a JSON object, nothing else. 

Here are some examples (I'm just showing you the sentences for these examples, not the story, because the story is too long):

<examples>
<sentence>
9 USC Trojans are not taking their upcoming matchup against the California Golden Bears for granted.	
</sentence>
Your response:
{{
    "label": "Lede",
    "justification": "The sentence summarizes the main idea of the story, telling us the two teams and the location of the game. It also appears as one of the first sentences, so we thought it best to label it as a lede."
}}

<sentence>
"Women's History Month is a important celebration of how far women have come in society particularly in the United States," said architecture student Daniela Robles. 
</sentence>
Your response:
{{
    "label": "Color",
    "justification": "The sentence adds an emotional context for the meaning of Women's History Month. It goes beyond 'just the facts' and adds a personal touch, thus it qualifies as a 'Color'."
}}

<sentence>
"Although improvements have been made in encouraging equality, there are always advancements to be made to counteract misogyny. ""
</sentence>
Your response:
{{
    "label": "Transition",
    "justification": "The sentence introduces a new idea in the news article (the need for advancements to counteract misogyny) and helps us tranistion from the previous idea. Thus, the most appropriate label is 'Transition'."
}}

<sentence>
Black-owned coffee shops and safe spaces are especially useful for students.
</sentence>
Your response:
{{
    "label": "Other",
    "justification": "The sentence doesn't fit into any of the other categories."
}}
</examples>

Now it's your turn. Please consider both the story and the sentence, and how it fits into the story. Remember, do not return anything else but the JSON object.

Story: 
<story>
{story}
</story>

Sentence: 
<sentence>
{sentence}
</sentence>

Your response:
"""


MULTI_SENTENCE_LABELING_PROMPT = """
You are a helpful assistant. I will give you {k} sentences from a news article, prefixed with the sentence index, and you will label them each with one of the following discourse tags:

<definitions>
1.  **Lede**: The opening sentence or sentences of a story that grabs the reader's attention and highlights the most important and newsworthy aspects, usually including the "who, what, when, where, why and how" elements of the story in succinct language.  
2.  **Nut Graf**: Context for the story: Why the audience cares, why the topic matters, and other crucial details relevant to the story.  
3.  **Background Information**: Contextual information related to the main topic of the story.  
4.  **Attribution**: Person, organization, or document that provides information.  
5.  **Evidence**: Information that supports the main point.  
6.  **Quote**: Exact words from someone you interviewed or documents you reviewed.  
7.  **Counterargument**: A crucial part of the story that displays evidence that the reporter has considered and consulted experts on possible rebuttals to the story's main topic or contention.  
8.  **Transition**: A sentence that indicates a new idea or topic is being introduced.  
9.  **Supporting Detail**: More information about an idea or topic that is intended to help the reader take action, deepen their understanding of the information, or draw a conclusion.  
10. **Source Opinion**: An opinionated statement from an individual who spoke directly or indirectly to a reporter. This label should be applied instead of "quote" if the quote from the source expresses a clear opinion.  
11. **Author Point of View**: An opinion expressed by the author that isn't backed up by evidence in the story and/or the reporter's expertise.  
12. **Analysis**: Where a reporter adds context to the story based on reporting and other knowledge drawn from expertise on the topic developed outside of directly reporting the story.  
13. **Color**: Details that add vivid and specific descriptions and contribute to the story's narrative, helping to paint a picture specific to this time, place, or situation—beyond "just the facts."
14. **Other**: Any other label that doesn't fit into the other 13.
</definitions>

For each sentence, return the sentence index, label and a justification for why you chose the label and not any other label. 
Return a list of JSON objects, one for each sentence, nothing else. 

Here are some examples (I'm just showing you random sentences for these examples from different articles, not the story, because the story is too long):

<examples>
<sentence>
(9) USC Trojans are not taking their upcoming matchup against the California Golden Bears for granted.	
</sentence>
Your response:
[
...
{{
    "sentence_index": 9,
    "label": "Lede",
    "justification": "The sentence summarizes the main idea of the story, telling us the two teams and the location of the game. It also appears as one of the first sentences, so we thought it best to label it as a lede."
}}
...
]

<sentence>
(5) "Women's History Month is a important celebration of how far women have come in society particularly in the United States," said architecture student Daniela Robles. 
</sentence>
Your response:
[
...
{{
    "sentence_index": 5,
    "label": "Color",
    "justification": "The sentence adds an emotional context for the meaning of Women's History Month. It goes beyond 'just the facts' and adds a personal touch, thus it qualifies as a 'Color'."
}}
...
]

<sentence>
(7) "Although improvements have been made in encouraging equality, there are always advancements to be made to counteract misogyny. ""
</sentence>
Your response:
[
...
{{
    "sentence_index": 7,
    "label": "Transition",
    "justification": "The sentence introduces a new idea in the news article (the need for advancements to counteract misogyny) and helps us tranistion from the previous idea. Thus, the most appropriate label is 'Transition'."
}}
...
]

<sentence>
(2) Black-owned coffee shops and safe spaces are especially useful for students.
</sentence>
Your response:
[
...
{{
    "sentence_index": 2,
    "label": "Other",
    "justification": "The sentence doesn't fit into any of the other categories."
}}
...
]
</examples>

Now it's your turn. Please consider both the story and the sentences, and how they fit into the story. Remember, do not return anything else but the list of JSON objects.

Story: 
<story>
{story}
</story>

Sentences: 
<sentences>
{sentences}
</sentences>

Your response:
"""

COMPARISON_PROMPT = """I will show you a student-written draft and a professionally-edited article, each with structural annotations for each sentence. The structural annotations are the following:

<definitions>
Lede: The opening sentence or sentences of a story that grabs the reader's attention and highlights the most important and newsworthy aspects, usually including the "who, what, when, where, why and how" elements in succinct language.
Nut Graf: Context for the story: Why the audience cares, why the topic matters and other crucial details relevant to the story.
Background Information: Contextual information related to the main topic of the story.
Attribution: Person, organization or document that provides information.
Evidence: Information that supports the main point.
Quote: Exact words from someone you interviewed or documents you reviewed.
Counterargument: This is a crucial part of the story that displays evidence that the reporter has considered and consulted experts on possible rebuttals to the story's main topic or contention.
Transition: A sentence that indicates a new idea or topic is being introduced.
Supporting Detail: More information about an idea or topic that is intended to help the reader take action, deepen their understanding of the information or draw a conclusion.
Source Opinion: An opinionated statement from an individual who spoke directly or indirectly to a reporter.
Author Point of View: An opinion expressed by the author that isn't backed up by evidence in the story and/or the reporters' expertise.
Analysis: This is where a reporter adds context to the story based on reporting and other knowledge drawn from other expertise on the topic developed outside of directly reporting the story.
Color: Details that add vivid and specific descripions and contribute to the story's narrative and help paint a picture specific to this time, place or situation. Beyond "just the facts."
Other: A sentence role we haven't seen before!
</definitions>

Here is the student article:

<student article>
{student_article}
</student article>

Here is the professionally edited article:

<professional article>
{professional_article}
</professional article>

Please give the student concise feedback on the ways their structure differs from the professional article 
and how they might change their article to match the professional article. 
If you don't see tags on the student article (or tags that say [Analyzing...], please infer for yourself what function they play). 
Answer with two concise paragraphs: the first paragraph should compliment what they've gotten right.  
If they already have some of the right structural components in the right places, mention that and compliment them.
The second paragraph should point out missing structural components and make suggestions for improvement. Point out specific places in the professional article that use the 
tags you're pointing out, but also acknowledge that the student article might be on a different topic than the professional article.
Don't write more than two paragraphs total and directly reference the structural labels where possible (but don't format them differently, just return plain text, with all the appropriate capitalization patterns that you see in the Definitions section). 
Speak directly to the student -- i.e. use words like "You should" or "We recommend" instead of "The student article."."""