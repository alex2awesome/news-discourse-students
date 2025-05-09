// Helper function for title case formatting
function toTitleCase(str) {
  if (!str) return '';
  return str.toLowerCase().split(' ').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
}

// Define tooltips for each structural tag.
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
    "definition": "This is a crucial part of the story that displays evidence that the reporter has considered and consulted experts on possible rebuttals to the story's main topic or contention.",
    "instructions": "Most stories should have a counterargument -- a description of the challenges, flaws or opposing views of something you've already described. This isn't the reporter's opinion -- it should be attributed to experts."
  },
  "Transition": {
    "definition": "A sentence that indicates a new idea or topic is being introduced.",
    "instructions": "Transitions indicate that you're introducing a new topic, point of view or revisiting a concept you introduced earlier in the piece. If that's not what you intended to do with this sentence, you should remove or rewrite it."
  },
  "Supporting Detail": {
    "definition": "More information about an idea or topic that is intended to help the reader take action, deepen their understanding of the information or draw a conclusion.",
    "instructions": "A supporting details shares more information about your topic with the reader that helps them take action, deepen their understanding or draw a conclusion. More important supporting details belong higher in the story; if you don't think this detail is vital to your reader's understanding or ability to take action, consider removing it or moving it down in the story."
  },
  "Source Opinion": {
    "definition": "An opinionated statement from an individual who spoke directly or indirectly to a reporter. This label should be applied instead of \"quote\" if the quote from the source expresses a clear opinion.",
    "instructions": "Including opinionated statements from your sources usually adds great detail to your piece; however, be sure that you're representing a range of opinions and that the opinions expressed in the piece reflect the makeup of opinions you uncovered in your reporting (don't leave out a whole point of view and balance the number of opinions expressed on each side)"
  },
  "Author Point of View": {
    "definition": "An opinion expressed by the author that isn't backed up by evidence in the story and/or the reporters' expertise.",
    "instructions": "This is flagged as the reporter's opinion. Unless you're writing an opinion piece, you should  provide evidence or attribution for the claim."
  },
  "Analysis": {
    "definition": "This is where a reporter adds context to the story based on reporting and other knowledge drawn from other expertise on the topic developed outside of directly reporting the story.",
    "instructions": "Analysis draws on your expertise as a reporter or other knowledge you bring to your reporting. You can give more insight into your expertise in your byline or within the body of the story. It's still important to attribute all your information."
  },
  "Color": {
    "definition": "Details that add vivid and specific descripions and contribute to the story's narrative and help paint a picture specific to this time, place or situation. Beyond \"just the facts.\"",
    "instructions": "Color helps your narrative pop with vivid, specific details and descriptions of relevant actions, situation and locations. Use concrete details instead of  vague generalities - show don't tell!"
  },
  "Other": {
    "definition": "A sentence role we haven't seen before!",
    "instructions": "This doesn't doesn't seem to play a clear role in your story. Do you think it's necessary information? If so, consider adding more detail. What are you trying to get across? You might also consider just removing it."
  }
};

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

// Add this function after the colorMap definition
function formatComparisonText(text) {
  // Create a mapping of variations to their canonical form
  const tagVariations = {
    // Singular forms (case-insensitive)
    'Lede': 'Lede',
    'lede': 'Lede',
    'Nut Graf': 'Nut Graf',
    'nut graf': 'Nut Graf',
    'Background Information': 'Background Information',
    'background information': 'Background Information',
    'Attribution': 'Attribution',
    'attribution': 'Attribution',
    'Evidence': 'Evidence',
    'evidence': 'Evidence',
    'Quote': 'Quote',
    'quote': 'Quote',
    'Counterargument': 'Counterargument',
    'counterargument': 'Counterargument',
    'Transition': 'Transition',
    'transition': 'Transition',
    'Supporting Detail': 'Supporting Detail',
    'supporting detail': 'Supporting Detail',
    'Source Opinion': 'Source Opinion',
    'source opinion': 'Source Opinion',
    'Author Point of View': 'Author Point of View',
    'author point of view': 'Author Point of View',
    'Analysis': 'Analysis',
    'analysis': 'Analysis',
    'Color': 'Color',
    'color': 'Color',
    'Other': 'Other',
    'other': 'Other',
    // Plural forms (case-insensitive)
    'Ledes': 'Lede',
    'ledes': 'Lede',
    'Nut Grafs': 'Nut Graf',
    'nut grafs': 'Nut Graf',
    'Background Informations': 'Background Information',
    'background informations': 'Background Information',
    'Attributions': 'Attribution',
    'attributions': 'Attribution',
    'Evidences': 'Evidence',
    'evidences': 'Evidence',
    'Quotes': 'Quote',
    'quotes': 'Quote',
    'Counterarguments': 'Counterargument',
    'counterarguments': 'Counterargument',
    'Transitions': 'Transition',
    'transitions': 'Transition',
    'Supporting Details': 'Supporting Detail',
    'supporting details': 'Supporting Detail',
    'Source Opinions': 'Source Opinion',
    'source opinions': 'Source Opinion',
    'Author Point of Views': 'Author Point of View',
    'author point of views': 'Author Point of View',
    'Analyses': 'Analysis',
    'analyses': 'Analysis',
    'Colors': 'Color',
    'colors': 'Color',
    'Others': 'Other',
    'others': 'Other'
  };

  // Create a regex pattern that matches any of the variations, including with punctuation
  const labelPattern = new RegExp(
    `\\b(${Object.keys(tagVariations).join('|')})\\b[.,-]?`,
    'gi'  // Added 'i' flag for case-insensitive matching
  );
  
  // Replace each label with a styled span
  return text.replace(labelPattern, (match) => {
    // Remove any trailing punctuation
    const cleanMatch = match.replace(/[.,-]$/, '');
    // Get the canonical form of the tag
    const canonicalTag = tagVariations[cleanMatch];
    if (!canonicalTag) return match; // Return original if no match found
    
    const color = colorMap[canonicalTag];
    const textColor = !['Background Information', 'Color', 'Other'].includes(canonicalTag) ? 'white' : 'black';
    
    // Preserve any trailing punctuation
    const punctuation = match.slice(cleanMatch.length);
    return `<span class="label" style="background-color: ${color}; color: ${textColor}; display: inline-block; padding: 2px 6px; border-radius: 3px; margin: 0 2px; font-size: 0.9em;">${canonicalTag}</span>${punctuation}`;
  });
}

// Initialize when document is ready
$(document).ready(function() {
  checkLoginStatus();
  
  // Add cache for retrievals and their comparisons
  let cachedRetrievals = {
    annenberg: null,
    latimes: null
  };
  
  // Add storage for analyzed labels
  let analyzedLabels = [];
  
  // Carousel functionality
  let currentPosition = 0;
  
  function updateCarousel() {
    const container = $('.carousel');
    const containerWidth = $('.carousel-container').width();
    const cardWidth = containerWidth; // Each card takes full width
    const translateX = -currentPosition * cardWidth;
    
    container.css('transform', `translateX(${translateX}px)`);
    
    // Update button states
    $('#prevButton').prop('disabled', currentPosition === 0);
    $('#nextButton').prop('disabled', 
      currentPosition >= $('#similarArticlesList .article-card').length - 1);
  }
  
  // Add window resize handler to update carousel
  $(window).resize(function() {
    if ($('#similarArticlesList .article-card').length > 0) {
      updateCarousel();
    }
  });
  
  $('#prevButton').click(function() {
    if (currentPosition > 0) {
      currentPosition--;
      updateCarousel();
    }
  });
  
  $('#nextButton').click(function() {
    const maxPosition = $('#similarArticlesList .article-card').length - 1;
    if (currentPosition < maxPosition) {
      currentPosition++;
      updateCarousel();
    }
  });
  
  $('#showSimilarToggle').change(function() {
    if (this.checked) {
      $('#similarArticlesContainer').show();
    } else {
      $('#similarArticlesContainer').hide();
    }
  });
  
  let currentSource = 'annenberg';

  // Function to format an article for comparison
  function formatArticleForComparison(article) {
    let formatted = '';
    if (article.sentences && article.sentences.length > 0) {
      article.sentences.forEach((sentence, idx) => {
        const label = article.labels ? toTitleCase(article.labels[idx]) : 'Analyzing...';
        formatted += `[${label}] ${sentence}\n`;
      });
    }
    return formatted;
  }

  // Function to get comparison between articles
  async function getArticleComparison(studentArticle, professionalArticle) {
    try {
      const response = await fetch('/api/compare_articles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          student_article: studentArticle,
          professional_article: professionalArticle
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Network response was not ok');
      }
      
      const data = await response.json();
      console.log('Received from /api/compare_articles:', {
        prompt: data.prompt,
        comparison: data.comparison
      });
      return data.comparison;
    } catch (error) {
      console.error('Error getting article comparison:', error);
      return null;
    }
  }

  // Function to display similar articles
  async function displaySimilarArticles(articles, source) {
    const container = $('#similarArticlesList');
    container.empty();
    
    // Create all article cards immediately with loading states
    articles.forEach(article => {
      const articleDiv = $('<div>').addClass('article-card');
      
      // Add headline
      const headlineDiv = $('<h3>').addClass('article-headline').text(article.headline);
      articleDiv.append(headlineDiv);
      
      // Add subheader
      if (article.subheader) {
        const subheaderDiv = $('<h4>').addClass('article-subheader').text(article.subheader);
        articleDiv.append(subheaderDiv);
      }
      
      // Add URL
      if (article.url) {
        const urlDiv = $('<div>').addClass('article-url');
        const urlLink = $('<a>')
          .attr('href', article.url)
          .attr('target', '_blank')
          .text('Read full article online')
          .addClass('article-link');
        urlDiv.append(urlLink);
        articleDiv.append(urlDiv);
      }
      
      // Add similarity score
      const scoreDiv = $('<div>').addClass('similarity-score')
        .text(`Similarity Score: ${article.score.toFixed(3)}`);
      articleDiv.append(scoreDiv);
      
      if (article.sentences && article.sentences.length > 0) {
        const sentencesDiv = $('<div>').addClass('sentences');
        article.sentences.forEach((sentence, idx) => {
          const label = article.labels ? toTitleCase(article.labels[idx]) : '';
          const justification = article.justification ? article.justification[idx] : '';
          
          const sentenceDiv = $('<div>').addClass('sentence');
          const labelSpan = $('<span>').addClass('label').text(label);
          
          if (colorMap[label]) {
            labelSpan.css('background-color', colorMap[label]);
            if (!['Background Information', 'Color', 'Other'].includes(label)) {
              labelSpan.css('color', 'white');
            }
          }
          
          // Add tooltip with justification if available
          if (tooltipTexts[label]) {
            const tooltipText = '<strong>' + label + '</strong><br><br>' +
              '<u>Why we tagged this:</u> ' + justification;
            
            labelSpan.attr({
              'data-bs-toggle': 'tooltip',
              'data-bs-placement': 'top',
              'title': tooltipText,
              'data-bs-html': 'true'
            });
            
            // Initialize tooltip
            new bootstrap.Tooltip(labelSpan[0], { html: true });
            
            // Add hover effects
            labelSpan.on('mouseenter', function() {
              $(this).css('border', '2px solid red');
            }).on('mouseleave', function() {
              $(this).css('border', '');
            });
          }
          
          sentenceDiv.append(labelSpan);
          sentenceDiv.append($('<span>').addClass('text').text(sentence));
          sentencesDiv.append(sentenceDiv);
        });
        articleDiv.append(sentencesDiv);

        // Add comparison section
        const comparisonDiv = $('<div>').addClass('article-comparison');
        const comparisonHeader = $('<h4>').text('Structural Comparison');
        comparisonDiv.append(comparisonHeader);
        
        // If we have a cached comparison, show it immediately
        if (article.comparison) {
          const comparisonText = $('<div>').addClass('comparison-text')
            .html(formatComparisonText(article.comparison.replace(/\n/g, '<br>')));
          comparisonDiv.append(comparisonText);
        } else {
          const loadingDiv = $('<div>').addClass('comparison-loading')
            .text('Analyzing structural differences...');
          comparisonDiv.append(loadingDiv);
        }
        
        articleDiv.append(comparisonDiv);
      }
      
      container.append(articleDiv);
    });

    // Reset position and update carousel immediately
    currentPosition = 0;
    updateCarousel();
    
    // Show similar articles container if toggle is checked
    if ($('#showSimilarToggle').is(':checked')) {
      $('#similarArticlesContainer').show();
    }
  }

  // Update source selector handler
  $('.source-button').click(function() {
    // Remove active class from all buttons
    $('.source-button').removeClass('active');
    // Add active class to clicked button
    $(this).addClass('active');
    // Update current source
    currentSource = $(this).data('source');
    if ($('#showSimilarToggle').is(':checked')) {
      // Use cached data if available
      if (cachedRetrievals[currentSource]) {
        displaySimilarArticles(cachedRetrievals[currentSource], currentSource);
      } else {
        fetchSimilarArticles();
      }
    }
  });

  async function fetchSimilarArticles() {
    const story = $('#promptInput').val();
    if (!story) return;

    try {
      const response = await fetch('/api/find_similar', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          story: story,
          source: currentSource
        })
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      
      // Cache the retrievals
      cachedRetrievals[currentSource] = data.articles;
      
      // Display articles immediately with loading state for comparisons
      await displaySimilarArticles(data.articles, currentSource);
      
      // Store articles for later comparison - we'll do the actual comparison in the 'complete' event
      window.similarArticles = data.articles;
    } catch (error) {
      console.error('Error fetching similar articles:', error);
    }
  }

  // Update the sendButton click handler
  $('#sendButton').click(function() {
    const story = $('#promptInput').val();
    if (!story) {
      alert('Please enter a story first');
      return;
    }
    
    // Show output container and add heading/explanation
    $('#outputContainer').show();
    $('#outputContainer').prepend(`
      <h2>We've analyzed the structure of your story for you</h2>
      <p class="analysis-explanation"><u>Mouse over each label to see more information about how each label was annotated.</u> Please take these results with a grain of salt. Confusing labels might be the result of mislabeling on our part. If you didn't to have the structure you see, it could be an indication that you didn't write sentences clearly in one style.</p>
    `);
    $('.similar-articles-section').show();
    $('#sendButton').hide();
    $('#stopButton').show();
    
    // Clear previous results and cache
    $('#analysisResults').empty();
    $('#similarArticlesList').empty();
    currentPosition = 0;
    cachedRetrievals = {
      annenberg: null,
      latimes: null
    };
    
    // Start streaming with POST request
    fetch('/api/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        story: story,
        source: currentSource
      })
    }).then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      function processText(text) {
        const lines = text.split('\n');
        for (let line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              handleEvent(data);
            } catch (e) {
              console.error('Error parsing event data:', e);
            }
          }
        }
      }

      async function read() {
        try {
          const {value, done} = await reader.read();
          if (done) {
            if (buffer.length > 0) {
              processText(buffer);
            }
            // After analysis is complete, fetch similar articles but don't generate comparisons yet
            fetchSimilarArticles();
            return;
          }
          
          buffer += decoder.decode(value, {stream: true});
          const lines = buffer.split('\n\n');
          buffer = lines.pop(); // Keep the last incomplete chunk in the buffer
          
          for (const line of lines) {
            processText(line);
          }
          
          return read();
        } catch (error) {
          console.error('Error reading stream:', error);
          $('#sendButton').show();
          $('#stopButton').hide();
        }
      }

      return read();
    }).catch(error => {
      console.error('Error:', error);
      $('#sendButton').show();
      $('#stopButton').hide();
    });
  });
  
  async function handleEvent(data) {
    switch(data.type) {
      case 'clean_text':
        // Handle cleaned text if needed
        break;
        
      case 'sentences':
        // Store sentences for later use
        window.parsed_sentences = data.sentences;
        // Reset analyzed labels
        analyzedLabels = new Array(data.sentences.length).fill('Analyzing...');
        // Create the main article card
        const mainArticleDiv = $('<div>').addClass('article-card');
        mainArticleDiv.append($('<h3>').text('Your article'));
        const mainSentencesDiv = $('<div>').addClass('sentences');
        data.sentences.forEach((sentence, idx) => {
          const sentenceDiv = $('<div>').addClass('sentence');
          const labelSpan = $('<span>').addClass('label').text('Analyzing...');
          sentenceDiv.append(labelSpan);
          sentenceDiv.append($('<span>').addClass('text').text(sentence));
          mainSentencesDiv.append(sentenceDiv);
        });
        mainArticleDiv.append(mainSentencesDiv);
        $('#analysisResults').append(mainArticleDiv);
        break;
        
      case 'similar_articles':
        // Display articles immediately with loading state for comparisons
        await displaySimilarArticles(data.articles, data.source);
        // Store articles for later comparison
        window.similarArticles = data.articles;
        break;
        
      case 'analysis':
        // Log the raw response
        console.log('Raw analysis response:', data.analysis);
        
        // Parse the JSON response
        let analysisData;
        try {
          analysisData = JSON.parse(data.analysis);
          console.log('Parsed analysis data:', analysisData);
        } catch (e) {
          console.error('Error parsing analysis JSON:', e);
          console.error('Failed to parse string:', data.analysis);
          analysisData = { label: data.analysis, justification: '' };
        }
        
        // Update the label for the analyzed sentence
        const sentenceDiv = $('#analysisResults .sentence').eq(data.index);
        const labelSpan = sentenceDiv.find('.label');
        const analysisLabel = toTitleCase(analysisData.label);
        console.log('Formatted label:', analysisLabel);
        
        // Store the analyzed label
        analyzedLabels[data.index] = analysisLabel;
        
        // Update the label text
        labelSpan.text(analysisLabel);
        
        // Apply color if available
        if (colorMap[analysisLabel]) {
          labelSpan.css('background-color', colorMap[analysisLabel]);
          if (!['Background Information', 'Color', 'Other'].includes(analysisLabel)) {
            labelSpan.css('color', 'white');
          }
        } else {
          console.warn('No color mapping found for label:', analysisLabel);
        }
        
        // Add tooltip if available
        if (tooltipTexts[analysisLabel]) {
          const tooltipText = '<strong>' + analysisLabel + '</strong><br><br>' +
            '<u>Why we tagged this:</u> ' + analysisData.justification;
          
          labelSpan.attr({
            'data-bs-toggle': 'tooltip',
            'data-bs-placement': 'top',
            'title': tooltipText,
            'data-bs-html': 'true'
          });
          
          // Initialize tooltip
          new bootstrap.Tooltip(labelSpan[0], { html: true });
          
          // Add hover effects
          labelSpan.on('mouseenter', function() {
            $(this).css('border', '2px solid red');
          }).on('mouseleave', function() {
            $(this).css('border', '');
          });
        } else {
          console.warn('No tooltip found for label:', analysisLabel);
        }
        break;
        
      case 'complete':
        $('#sendButton').show();
        $('#stopButton').hide();
        
        // Now that analysis is complete, generate comparisons for stored articles
        if (window.similarArticles) {
          // Verify all labels are analyzed
          const unanalyzedLabels = analyzedLabels.filter(label => label === 'Analyzing...');
          if (unanalyzedLabels.length > 0) {
            console.error('Not all labels have been analyzed yet:', unanalyzedLabels.length, 'remaining');
            // Force update any remaining "Analyzing..." labels to "Other"
            analyzedLabels = analyzedLabels.map(label => label === 'Analyzing...' ? 'Other' : label);
            
            // Update any remaining "Analyzing..." labels in the UI
            $('#analysisResults .sentence').each(function(index) {
              const labelSpan = $(this).find('.label');
              if (labelSpan.text() === 'Analyzing...') {
                labelSpan.text('Other');
                if (colorMap['Other']) {
                  labelSpan.css('background-color', colorMap['Other']);
                }
              }
            });
          }
          
          // Generate comparisons for all sources in parallel
          const sources = ['annenberg', 'latimes'];
          const allComparisonPromises = sources.map(async (source) => {
            if (!cachedRetrievals[source]) {
              try {
                const response = await fetch('/api/find_similar', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({
                    story: $('#promptInput').val(),
                    source: source
                  })
                });
                
                if (!response.ok) throw new Error('Network response was not ok');
                const data = await response.json();
                cachedRetrievals[source] = data.articles;
                
                // Generate comparisons for all articles
                const comparisonPromises = data.articles.map(async (article, index) => {
                  const studentArticle = formatArticleForComparison({
                    sentences: window.parsed_sentences,
                    labels: analyzedLabels
                  });
                  const professionalArticle = formatArticleForComparison(article);
                  
                  try {
                    const comparison = await getArticleComparison(studentArticle, professionalArticle);
                    article.comparison = comparison;
                    
                    // Update UI if this is the current source
                    if (source === currentSource) {
                      const articleDiv = $(`#similarArticlesList .article-card`).eq(index);
                      const loadingDiv = articleDiv.find('.comparison-loading');
                      if (loadingDiv.length) {
                        loadingDiv.remove();
                        const comparisonText = $('<div>').addClass('comparison-text')
                          .html(formatComparisonText(comparison.replace(/\n/g, '<br>')));
                        articleDiv.find('.article-comparison').append(comparisonText);
                      }
                    }
                  } catch (error) {
                    console.error('Error generating comparison:', error);
                    article.comparison = 'Error generating comparison. Please try again.';
                    if (source === currentSource) {
                      const articleDiv = $(`#similarArticlesList .article-card`).eq(index);
                      const loadingDiv = articleDiv.find('.comparison-loading');
                      if (loadingDiv.length) {
                        loadingDiv.remove();
                        articleDiv.find('.article-comparison').append(
                          $('<div>').addClass('comparison-error')
                            .text('Error generating comparison. Please try again.')
                        );
                      }
                    }
                  }
                });
                
                await Promise.all(comparisonPromises);
              } catch (error) {
                console.error(`Error fetching ${source} articles:`, error);
              }
            }
          });
          
          // Wait for all sources to complete
          await Promise.all(allComparisonPromises);
          
          // Update the carousel after all comparisons are done
          updateCarousel();
        }
        break;
        
      case 'stopped':
        $('#sendButton').show();
        $('#stopButton').hide();
        break;
    }
  }
  
  $('#stopButton').click(function() {
    $.post('/api/stop');
  });
});

async function checkLoginStatus() {
  const loginContainer = $("#loginContainer");
  const loadingSpinner = $("#loadingSpinner");
  const loginButton = $("a[href='/login']");
  const loginStatus = $("#loginStatus");

  try {
    loadingSpinner.show();
    loginButton.hide();
    loginStatus.text("Checking login status...");

    const response = await fetch("/api/ask", {
      method: 'GET',
      headers: {'Content-Type': 'application/json'}
    });
    
    if (response.status === 401) {
      loginContainer.show();
      $("#inputContainer").hide();
      $("#outputContainer").hide();
      loginButton.show();
      loginStatus.text("");
    } else {
      loginContainer.hide();
      $("#inputContainer").show();
      $("#outputContainer").show();
    }
  } catch (error) {
    console.error('Error checking login status:', error);
    loginContainer.show();
    $("#inputContainer").hide();
    $("#outputContainer").hide();
    loginButton.show();
    loginStatus.text("Error checking login status. Please try again.");
  } finally {
    loadingSpinner.hide();
  }
}
