CLEAN_TEXT_PROMPT = """
You are a helpful assistant. I will give you a story parsed from the internet, and you will return a cleaned version of the story. Remove all HTML tags, malformed 
parses, and other non-text content.

Here is the story:
```{story}```

Return the cleaned story, nothing else. Don't say anything else.
"""

SENTENCIZE_PROMPT = """
You are a helpful assistant that will take a story and break it into sentences. 
I will give you a story and you will return a python list of sentences. Just return the list, nothing else.
Directly copy ALL of the sentences from the story into the list. Don't add any other text.

Here is the story:
```{story}```

Sentences:
"""

TEST_PROMPT = """
You are a helpful assistant. This prompt is a test. You will take a sentence and count the number of words in it.
Just return the number, nothing else.

Here is the sentence:
```{sentence}```

Number of words:
"""

LABELING_PROMPT = """
You are a helpful assistant. I will give you a sentence from a news article and you will label it with one of the following discourse tags:

1. **Headline**: The main title of the article, designed to grab attention and summarize the central theme or most important aspect of the story.
2. **Introduction**: The opening portion of the article, which introduces the topic and provides a brief overview of what will be covered.
3. **Lede**: The opening sentence or paragraph of the article, which highlights the most important details of the story (such as the who, what, when, where, why, and how) and entices the reader to continue.
4. **Nut graf**: A paragraph (often near the beginning of the article) that explains the main point or significance of the story, providing context or deeper insight into why the topic matters.
5. **Background information**: Sentences that provide necessary context or historical details that help the reader understand the broader situation or topic being discussed.
6. **Opinion**: A statement or section that reflects the writer's personal viewpoint, analysis, or interpretation of the facts presented in the article.
7. **Color**: Descriptive language that provides vivid, sensory details to give readers a deeper sense of the atmosphere, mood, or setting of the story, often enhancing the narrative.
8. **Transition**: A sentence or phrase that links different sections of the article or shifts the focus from one idea to another, ensuring smooth flow and coherence.
9. **Supporting detail**: Specific facts, examples, or evidence that reinforce or clarify the main points in the article.
10. **Sourcing/source information**: Information that attributes facts, opinions, or quotes to specific individuals, organizations, or documents, establishing credibility and authority for the article's content.

Just return the label, nothing else. 

Here are some examples (I'm just showing you the sentences for these examples, not the story, because the story is too long):

Sentence: 9 USC Trojans are not taking their upcoming matchup against the California Golden Bears for granted.	
Lede

Sentence: “Women’s History Month is just a celebration of how far women have come in society particularly in the United States,” said architecture student Daniela Robles. 
Color

Sentence: "Although improvements have been made in encouraging equality, there are always advancements to be made to counteract misogyny. “"
Transition

Sentence: Black-owned coffee shops and safe spaces are especially useful for students.
Nut graf

Sentence: Austin is the president of Kó Society, a social club for Black women on campus.
Sourcing/source information

Now it's your turn. Please consider both the story and the sentence, and how it fits into the story.

Story: ```{story}```
Sentence: ```{sentence}```
"""