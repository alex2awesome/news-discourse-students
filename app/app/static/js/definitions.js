// ===========================================
// Structural definitions for article analysis
// ===========================================

// Structural tag metadata (tooltips & info)
const tooltipTexts = {
  "Lede": {
    "definition": "The opening sentence or sentences of a story that grabs the reader's attention and highlights the most important and newsworthy aspects, usually including the \"who, what, when, where, why and how\" elements in succinct language.",
    "instructions": "Your lede should grab the reader's attention and make them want to know more. A hard news lede should include the story's 5W's (who, what, when, where and why) and emphasize its most newsworthy parts. A feature story might use an anecdotal lede to draw the reader in."
  },
  "Nut Graf": {
    "definition": "Context for the story: Why the audience cares, why the topic matters and other crucial details relevant to the story.",
    "instructions": "Your \"nut graf\" should tell us why we care about this story and why you're telling it now."
  },
  "Background Information": {
    "definition": "Contextual information related to the main topic of the story.",
    "instructions": "This gives the reader more information and context for the story, but it's not the main evidence and so comes later in a hard news story."
  },
  "Attribution": {
    "definition": "Person, organization or document that provides information.",
    "instructions": "This where you tell us where the info came from-- for example, a written statement, a website, a YouTube video, a public record or event, an interview with the reporter, an interview published somewhere else, etc. All information in your story needs attribution unless it's super-duper obvious or you witnessed it yourself."
  },
  "Evidence": {
    "definition": "Information that supports the main point.",
    "instructions": "Important information that supports the main point of the story. In a hard news story, the most important evidence comes first."
  },
  "Quote": {
    "definition": "Exact words from someone you interviewed or documents you reviewed.",
    "instructions": "Your story should have quotes from more than one person and represent all relevant points of view. Use the best parts of the quote and paraphrase the rest. Make sure the quote is formatted correctly and doesn't have misspellings, grammar mistakes or filler words like 'um.' Put the attribution at the end of the quote or in the middle if it's long."
  },
  "Counterargument": {
    "definition": "Evidence that you've considered possible rebuttals to the story's main point.",
    "instructions": "Most stories should have a counterargument — a description of the challenges, flaws or opposing views of something you've already described. This isn't the reporter's opinion — it should be attributed to experts."
  },
  "Transition": {
    "definition": "A sentence that indicates a new idea or topic is being introduced.",
    "instructions": "Transitions indicate that you're introducing a new topic, point of view or revisiting a concept you introduced earlier in the piece."
  },
  "Supporting Detail": {
    "definition": "More information about an idea or topic that is intended to help the reader take action, deepen their understanding or draw a conclusion.",
    "instructions": "More important supporting details belong higher in the story; if this detail isn't vital, consider removing it or moving it lower."
  },
  "Source Opinion": {
    "definition": "An opinionated statement from an interviewee or document.",
    "instructions": "Include opinions from multiple perspectives and balance them across the story."
  },
  "Author Point of View": {
    "definition": "The reporter's opinion that isn't backed up by evidence in the story.",
    "instructions": "Unless you're writing an opinion piece, back up statements with evidence or attribution."
  },
  "Analysis": {
    "definition": "Context or explanation provided by the reporter drawing on outside expertise.",
    "instructions": "It's still important to attribute information that is not common knowledge."
  },
  "Color": {
    "definition": "Vivid details that help paint a picture beyond \"just the facts\".",
    "instructions": "Use concrete details — show, don't tell!"
  },
  "Other": {
    "definition": "A sentence role we haven't seen before!",
    "instructions": "Consider whether this sentence is necessary or could be rewritten for clarity."
  }
};

// Color mapping for structural tags
const colorMap = {
  "Lede": "#1f77b4",
  "Nut Graf": "#ff7f0e",
  "Background Information": "#2ca02c",
  "Attribution": "#d62728",
  "Evidence": "#9467bd",
  "Quote": "#8c564b",
  "Counterargument": "#e377c2",
  "Transition": "#7f7f7f",
  "Supporting Detail": "#bcbd22",
  "Source Opinion": "#17becf",
  "Author Point of View": "#ff9896",
  "Analysis": "#c49c94",
  "Color": "#aec7e8",
  "Other": "#c5b0d5"
}; 