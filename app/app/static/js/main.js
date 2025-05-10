// ===========================================
// Core application functionality 
// ===========================================

// -----------------------------------------------------------
// Main document logic
// -----------------------------------------------------------
$(document).ready(function () {
  checkLoginStatus();
  
  /* ---------- Local state ---------- */
  const state = {
    parsedSentences: [],
    analyzedLabels: [],
    cachedRetrievals: {},      // source -> [articles]
    comparisonCache: {},       // `${source}_${headline}` -> comparison text
    currentSource: null,
    currentPosition: 0,
    requestController: null,   // Abort controller so we can stop mid-stream if needed
    sliderPosition: 50,        // Initial horizontal slider position (percentage)
    verticalSliderPosition: 70 // Initial vertical slider position (percentage)
  };
  
  /* ---------- Initialize layout ---------- */
  // Set initial heights based on verticalSliderPosition
  $('.article-cards-grid').css('height', `${state.verticalSliderPosition}vh`);
  $('.structural-comparison-section').css('margin-top', `calc(${state.verticalSliderPosition}vh + 20px)`);

  /* ---------- Initialize column slider ---------- */
  function initializeColumnSlider() {
    // Remove any existing sliders first to prevent duplicates
    $('.column-resize-slider').remove();
    
    // Remove previous event handlers to prevent duplicates
    $(document).off('mousemove.columnSlider mouseup.columnSlider');
    
    // Create the slider element
    const $slider = $('<div>').addClass('column-resize-slider');
    $('.article-cards-grid').append($slider);
    
    // Set initial position based on stored value
    $slider.css('left', `${state.sliderPosition}%`);
    
    // Variables to track dragging state
    let isDragging = false;
    let startX, startWidth, containerWidth;
    
    // Add event listeners for dragging
    $slider.on('mousedown', function(e) {
      isDragging = true;
      $slider.addClass('dragging');
      
      // Store initial positions
      startX = e.clientX;
      containerWidth = $('.article-cards-grid').width();
      startWidth = $('.analysis-side').width();
      
      // Prevent text selection during drag
      e.preventDefault();
    });
    
    // Use namespaced events for easier cleanup
    $(document).on('mousemove.columnSlider', function(e) {
      if (!isDragging) return;
      
      // Calculate the width difference
      const dx = e.clientX - startX;
      const newWidth = startWidth + dx;
      
      // Calculate percentage (constrained between 20% and 80%)
      let percentage = Math.min(80, Math.max(20, (newWidth / containerWidth) * 100));
      
      // Snap to 50% when within 45-55% range
      if (percentage >= 45 && percentage <= 55) {
        percentage = 50;
      }
      
      // Update column widths
      $('.analysis-side').css('flex', `0 0 ${percentage}%`);
      $('.similar-side').css('flex', `0 0 ${100 - percentage}%`);
      
      // Update slider position
      $slider.css('left', `${percentage}%`);
      
      // Store the current position
      state.sliderPosition = percentage;
    });
    
    $(document).on('mouseup.columnSlider', function() {
      if (isDragging) {
        isDragging = false;
        $slider.removeClass('dragging');
        
        // Trigger resize event to adjust content
        $(window).trigger('resize');
      }
    });
  }
  
  /* ---------- Initialize vertical slider ---------- */
  function initializeVerticalSlider() {
    // Remove any existing vertical sliders first to prevent duplicates
    $('.vertical-resize-slider').remove();
    
    // Remove previous event handlers to prevent duplicates
    $(document).off('mousemove.verticalSlider mouseup.verticalSlider');
    
    // Create the slider element
    const $slider = $('<div>').addClass('vertical-resize-slider');
    $('.article-cards-grid').append($slider);
    
    // Variables to track dragging state
    let isDragging = false;
    let startY, startHeight, containerHeight;
    
    // Add event listeners for dragging
    $slider.on('mousedown', function(e) {
      isDragging = true;
      $slider.addClass('dragging');
      
      // Store initial positions
      startY = e.clientY;
      containerHeight = $('.article-cards-grid').height();
      startHeight = containerHeight;
      
      // Prevent text selection during drag
      e.preventDefault();
    });
    
    // Use namespaced events for easier cleanup
    $(document).on('mousemove.verticalSlider', function(e) {
      if (!isDragging) return;
      
      // Calculate the height difference
      const dy = e.clientY - startY;
      const newHeight = startHeight + dy;
      
      // Calculate new height (constrained between 300px and 80vh)
      const minHeight = 300; // Minimum height in pixels
      const maxHeight = window.innerHeight * 0.8; // 80vh in pixels
      const constrainedHeight = Math.min(maxHeight, Math.max(minHeight, newHeight));
      
      // Update article-cards-grid height
      $('.article-cards-grid').css('height', `${constrainedHeight}px`);
      
      // Store the current position as percentage of viewport height
      state.verticalSliderPosition = (constrainedHeight / window.innerHeight) * 100;
    });
    
    $(document).on('mouseup.verticalSlider', function() {
      if (isDragging) {
        isDragging = false;
        $slider.removeClass('dragging');
        
        // Trigger resize event to adjust content
        $(window).trigger('resize');
      }
    });
  }
  
  /* ---------- Initialize sliders when the similar side is shown ---------- */
  $(document).on('click', '.source-button', function() {
    if ($(this).hasClass('active')) {
      // If we're hiding the similar side, remove the sliders
      $('.column-resize-slider, .vertical-resize-slider').remove();
    } else if ($('.column-resize-slider').length === 0) {
      // Only add the sliders if they don't exist yet
      setTimeout(function() {
        initializeColumnSlider();
        initializeVerticalSlider();
        
        // Make sure the structural comparison section is positioned properly
        $('.structural-comparison-section').css('margin-top', '20px');
      }, 300); // Add after transition completes
    }
  });

  /* ---------- UI helpers ---------- */
  function showDiveDeeper() {
    $('.dive-deeper-section').show();
  }

  /* ---------- Header height adjustment ---------- */
  function adjustArticleHeaderHeights() {
    // Reset heights first to get true content height
    $('.article-card-header').css('height', 'auto');
    
    // Calculate maximum height across all headers
    let maxHeight = 0;
    $('.article-card-header').each(function() {
      const contentHeight = $(this)[0].scrollHeight;
      maxHeight = Math.max(maxHeight, contentHeight);
    });
    
    // Apply the same height to all headers to maintain alignment
    $('.article-card-header').css('height', maxHeight + 'px');
  }

  function formatArticleForComparison(article) {
    let formatted = '';
    if (article.sentences && article.sentences.length) {
      article.sentences.forEach((sentence, idx) => {
        const lbl = article.labels ? toTitleCase(article.labels[idx]) : 'Analyzing...';
        formatted += `[${lbl}] ${sentence}\n`;
      });
    }
    return formatted;
  }

  async function getArticleComparison(article) {
    const cacheKey = `${state.currentSource}_${article.headline}`;
    if (state.comparisonCache[cacheKey]) return state.comparisonCache[cacheKey];

    $('#structuralComparison').html('<div class="comparison-loading">Analyzing structural differences...</div>');

    try {
      const studentFormatted = formatArticleForComparison({
        sentences: state.parsedSentences,
        labels: state.analyzedLabels
      });
      const professionalFormatted = formatArticleForComparison(article);

      // Prepare request payload
      const payload = {
        student_article: studentFormatted,
        professional_article: professionalFormatted
      };
      
      // Log the comparison request for debugging
      console.log('Sending comparison request:', payload);

      const resp = await fetch('/api/compare_articles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error('Network response was not ok');
      const data = await resp.json();
      
      // Log the response including prompt
      console.log('Received comparison response:', {
        prompt: data.prompt,
        comparison: data.comparison
      });
      
      state.comparisonCache[cacheKey] = data.comparison;
      return data.comparison;
    } catch (err) {
      console.error(err);
      return 'Error generating comparison. Please try again.';
    }
  }

  function updateCarousel() {
    // Hide all cards
    $('#similarArticlesList .article-card').css('display', 'none');
    
    // Show only the current card
    $('#similarArticlesList .article-card').eq(state.currentPosition).css('display', 'block');
    
    // Update button states
    $('#prevButton').prop('disabled', state.currentPosition === 0);
    
    // Get the number of articles
    const articleCount = $('#similarArticlesList .article-card').length;
    $('#nextButton').prop('disabled', state.currentPosition >= articleCount - 1);
    
    // Update comparison content for the active article
    const activeArticles = state.cachedRetrievals[state.currentSource] || [];
    const currentArticle = activeArticles[state.currentPosition];
    if (currentArticle) {
      getArticleComparison(currentArticle).then(cmp => {
        $('#structuralComparison').html(formatComparisonText(cmp.replace(/\n/g, '<br>')));
      });
    }
  }

  function buildSentenceDiv(label, sentence, justification = '') {
    const $sentenceDiv = $('<div>').addClass('sentence');
    const $label = $('<span>').addClass('label').text(label);
          
          if (colorMap[label]) {
      $label.css('background-color', colorMap[label]);
            if (!['Background Information', 'Color', 'Other'].includes(label)) {
        $label.css('color', 'white');
            }
          }
          
          if (tooltipTexts[label]) {
      const tooltip = `<strong>${label}</strong><br><br><u>Why we tagged this:</u> ${justification}`;
      $label.attr({
              'data-bs-toggle': 'tooltip',
              'data-bs-placement': 'top',
        'title': tooltip,
              'data-bs-html': 'true'
            });
      new bootstrap.Tooltip($label[0], { html: true });
    }

    return $sentenceDiv.append($label).append($('<span>').addClass('text').text(sentence));
  }

  function displaySimilarArticles(articles, source) {
    const $container = $('#similarArticlesList');
    $container.empty();

    articles.forEach((article, index) => {
      const $card = $('<div>').addClass('article-card');
      
      // Set display:none for all cards except the first one
      if (index > 0) {
        $card.css('display', 'none');
      }
      
      // Create a consistent card header area
      const $cardHeader = $('<div>').addClass('article-card-header');
      
      // Ensure headline is clearly visible with proper styling
      const $headline = $('<h3>').addClass('article-headline').text(article.headline);
      $cardHeader.append($headline);
      
      // Add URL to header (not between header and sentences)
      if (article.url) {
        $cardHeader.append($('<div>').addClass('article-url').append(
          $('<a>').addClass('article-link').attr({ href: article.url, target: '_blank' }).text(article.url)
        ));
      }
      
      // Add the header to the card
      $card.append($cardHeader);

      if (article.sentences && article.sentences.length) {
        const $sentencesDiv = $('<div>').addClass('sentences');
        article.sentences.forEach((sentence, idx) => {
          const lbl = article.labels ? toTitleCase(article.labels[idx]) : '';
          const just = article.justification ? article.justification[idx] : '';
          $sentencesDiv.append(buildSentenceDiv(lbl, sentence, just));
        });
        $card.append($sentencesDiv);
      }

      $container.append($card);
    });

    state.currentPosition = 0;
    updateCarousel();
    
    // Adjust header heights after content is loaded
    setTimeout(adjustArticleHeaderHeights, 50);
  }

  function fetchSimilarArticles(source, storyText) {
    // Log the fetch request
    console.log(`Fetching similar articles from ${source}`);
    
    return fetch('/api/find_similar', {
        method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ story: storyText, source })
    })
      .then(resp => {
        if (!resp.ok) throw new Error('Network response was not ok');
        return resp.json();
      })
      .then(data => {
        console.log(`Received ${data.articles.length} similar articles from ${source}`);
        state.cachedRetrievals[source] = data.articles;
        return data.articles;
      });
  }

  /* ---------- Streaming event handler ---------- */
  function handleEvent(evt) {
    switch (evt.type) {
      case 'sentences': {
        state.parsedSentences = evt.sentences;
        state.analyzedLabels = new Array(evt.sentences.length).fill('Analyzing...');

        const $articleCard = $('<div>').addClass('article-card');
        // Create a consistent card header area
        const $cardHeader = $('<div>').addClass('article-card-header');
        $cardHeader.append($('<h3>').addClass('article-headline').text('Your article'));
        $articleCard.append($cardHeader);
        
        // Create sentences container
        const $sentDiv = $('<div>').addClass('sentences');
        evt.sentences.forEach(s => {
          $sentDiv.append(buildSentenceDiv('Analyzing...', s));
        });
        $articleCard.append($sentDiv);
        $('#analysisResults').append($articleCard);
        
        // Adjust header heights after the article card is added
        setTimeout(adjustArticleHeaderHeights, 50);
        break;
      }
      case 'analysis': {
        const idx = evt.index;
        let analysisData;
        try { analysisData = JSON.parse(evt.analysis); } catch { analysisData = { label: evt.analysis, justification: '' }; }
        const lbl = toTitleCase(analysisData.label);
        state.analyzedLabels[idx] = lbl;

        const $sentence = $('#analysisResults .sentence').eq(idx);
        const $label = $sentence.find('.label');
        $label.text(lbl);
        if (colorMap[lbl]) {
          $label.css('background-color', colorMap[lbl]);
          if (!['Background Information', 'Color', 'Other'].includes(lbl)) $label.css('color', 'white');
        }
        if (tooltipTexts[lbl]) {
          const tt = `<strong>${lbl}</strong><br><br><u>Definition:</u> ${tooltipTexts[lbl].definition}<br><br><u>Instructions:</u> ${tooltipTexts[lbl].instructions}<br><br><u>Why we tagged this:</u> ${analysisData.justification}`;
          $label.attr({ 'data-bs-toggle': 'tooltip', 'data-bs-placement': 'top', title: tt, 'data-bs-html': 'true' });
          new bootstrap.Tooltip($label[0], { html: true });
        }
        break;
      }
      case 'complete': {
        // Replace any still-analyzing labels with Other
        state.analyzedLabels = state.analyzedLabels.map(l => l === 'Analyzing...' ? 'Other' : l);
        $('#analysisResults .label').filter((_, el) => $(el).text() === 'Analyzing...').each((_, el) => {
          $(el).text('Other').css('background-color', colorMap['Other']);
        });
        showDiveDeeper();
        break;
      }
      case 'similar_articles': {
        state.cachedRetrievals[evt.source] = evt.articles;
        if (state.currentSource === evt.source) displaySimilarArticles(evt.articles, evt.source);
        break;
      }
      case 'stopped': {
        $('#sendButton').show();
        $('#stopButton').hide();
        break;
      }
    }
  }

  /* ---------- Send/Stop handlers ---------- */
  $('#sendButton').on('click', function () {
    const story = $('#promptInput').val();
    if (!story) return alert('Please paste an article first.');

    // Reset UI
    $('#analysisResults').empty();
    $('#structuralComparison').empty();
    $('#similarArticlesList').empty();
    $('.dive-deeper-section').hide();
    $('.similar-side').hide();
    $('.source-button').removeClass('active');
    
    // Reset state
    state.parsedSentences = [];
    state.analyzedLabels = [];
    state.cachedRetrievals = {};
    state.comparisonCache = {};
    state.currentSource = null;
    state.currentPosition = 0;
    
    // Show output and processing indicator
    $('#outputContainer').show();
    $('#sendButton').hide();
    $('#stopButton').show();
    
    const controller = new AbortController();
    state.requestController = controller;

    fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ story }),
      signal: controller.signal
    }).then(resp => {
      if (!resp.ok) throw new Error('Network response was not ok');
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      function process(bufferText) {
        bufferText.split('\n').forEach(line => {
          if (line.startsWith('data: ')) {
            try { handleEvent(JSON.parse(line.slice(6))); } catch (e) { console.error('Bad JSON', e); }
          }
        });
      }

      function read() {
        reader.read().then(({ value, done }) => {
          if (done) { $('#sendButton').show(); $('#stopButton').hide(); return; }
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop();
          parts.forEach(process);
          read();
        }).catch(err => console.error(err));
        }
      read();
    }).catch(err => {
      if (err.name !== 'AbortError') console.error(err);
      $('#sendButton').show();
      $('#stopButton').hide();
    });
  });
  
  $('#stopButton').on('click', function () {
    if (state.requestController) state.requestController.abort();
    // Notify backend so it can clean up any server-side streaming
    fetch('/api/stop', { method: 'POST' }).catch(() => {});
        $('#sendButton').show();
        $('#stopButton').hide();
  });

  /* ---------- Source buttons & carousel controls ---------- */
  $(document).on('click', '.source-button', function () {
    const $btn = $(this);
    const source = $btn.data('source');

    if ($btn.hasClass('active')) {
      // Toggle off
      $btn.removeClass('active');
      state.currentSource = null;
      
      // Instead of hide(), first set opacity to trigger transition
      $('.similar-side').css({
        'opacity': '0',
        'transform': 'translateX(20px)'
      });
      
      // Then actually hide after transition completes
      setTimeout(function() {
        $('.similar-side').hide();
        // Remove the sliders
        $('.column-resize-slider, .vertical-resize-slider').remove();
        // Reset the analysis side to full width
        $('.analysis-side').css('flex', '1');
      }, 300);
      
      $('.structural-comparison-section').hide();
      return;
    }

    $('.source-button').removeClass('active');
    $btn.addClass('active');
    state.currentSource = source;

    // First make display block but with 0 opacity
    $('.similar-side').css({
      'display': 'block',
      'opacity': '0',
      'transform': 'translateX(20px)'
    });
    
    // Then trigger transition by setting opacity after a tiny delay
    setTimeout(function() {
      $('.similar-side').css({
        'opacity': '1',
        'transform': 'translateX(0)'
      });
      
      // Apply stored slider positions if they exist
      if (state.sliderPosition !== 50) {
        $('.analysis-side').css('flex', `0 0 ${state.sliderPosition}%`);
        $('.similar-side').css('flex', `0 0 ${100 - state.sliderPosition}%`);
      }
      
      // Add the sliders after the transition completes
      setTimeout(function() {
        initializeColumnSlider();
        initializeVerticalSlider();
      }, 300);
    }, 10);
    
    $('.structural-comparison-section').show();

    if (state.cachedRetrievals[source]) {
      displaySimilarArticles(state.cachedRetrievals[source], source);
    } else {
      fetchSimilarArticles(source, $('#promptInput').val())
        .then(arts => displaySimilarArticles(arts, source))
        .catch(err => {
          console.error(err);
          $('#structuralComparison').html('<div class="comparison-error">Error fetching articles.</div>');
        });
    }
  });

  $('#prevButton').on('click', function () {
    if (state.currentPosition > 0) { state.currentPosition--; updateCarousel(); }
  });
  $('#nextButton').on('click', function () {
    const max = $('#similarArticlesList .article-card').length - 1;
    if (state.currentPosition < max) { state.currentPosition++; updateCarousel(); }
                });
                
  /* ---------- Window resize handler ---------- */
  $(window).on('resize', function () {
    // Debounce the resize event to prevent excessive calculations
    clearTimeout(window.resizeTimer);
    window.resizeTimer = setTimeout(function() {
      // Adjust article card headers to maintain equal heights
      adjustArticleHeaderHeights();
      
      // Run existing carousel update if articles exist
      if ($('#similarArticlesList .article-card').length) {
        updateCarousel();
      }
      
      // Reposition sliders if they exist
      if ($('.column-resize-slider').length && $('.similar-side').is(':visible')) {
        $('.column-resize-slider').css('left', `${state.sliderPosition}%`);
      }
      if ($('.vertical-resize-slider').length && $('.similar-side').is(':visible')) {
        $('.article-cards-grid').css('height', `${state.verticalSliderPosition}vh`);
      }
    }, 250);
  });

  /* ---------- Check and update layout on window load ---------- */
  $(window).on('load', function() {
    // Force a resize event to make sure everything is laid out correctly
    $(window).trigger('resize');
  });

  /* ---------- Label hover interactions ---------- */
  function initLabelHoverInteractions() {
    // Track the current index for each label
    const labelIndices = {};
    let currentTermWithTooltip = null;
    let clickAgainTooltip = null;
    
    // Function to find and scroll to matching label in a container
    function scrollToMatchingLabel(labelText, container, cycleToNext = false) {
      // Make sure the container exists
      if (!$(container).length) {
        console.log(`Container ${container} not found`);
        return { found: false, totalMatches: 0 };
      }
      
      // Normalize the labelText by trimming whitespace
      labelText = labelText.trim();
      
      // Create a unique key for this label+container combination
      const labelKey = `${labelText}_${container}`;
      
      // Initialize the index if not already set, or cycle to next if requested
      if (labelIndices[labelKey] === undefined || !cycleToNext) {
        labelIndices[labelKey] = 0;
      } else if (cycleToNext) {
        labelIndices[labelKey]++;
      }
      
      // Find all labels in the container
      const labels = $(container).find('.label');
      const matchingLabels = [];
      
      // Collect all matching labels
      for (let i = 0; i < labels.length; i++) {
        const currentLabel = $(labels[i]).text().trim();
        if (currentLabel === labelText) {
          matchingLabels.push($(labels[i]));
        }
      }
      
      // If no matches found
      if (matchingLabels.length === 0) {
        console.log(`No match found for label "${labelText}"`);
        return { found: false, totalMatches: 0 };
      }
      
      // Make sure the index is in range (cycle back to 0 if needed)
      if (labelIndices[labelKey] >= matchingLabels.length) {
        labelIndices[labelKey] = 0;
      }
      
      // Get the current match
      const currentMatch = matchingLabels[labelIndices[labelKey]];
      const sentenceElement = currentMatch.closest('.sentence');
      
      // Make sure we found a valid sentence element
      if (sentenceElement.length === 0) {
        console.log('No parent sentence element found');
        return { found: false, totalMatches: matchingLabels.length };
      }
      
      console.log(`Found matching label "${labelText}" at position ${labelIndices[labelKey]} of ${matchingLabels.length}`);
      
      const containerElement = $(container);
      
      // Calculate scroll position with better visibility (position it at 40% from top of viewport)
      const containerHeight = containerElement.height();
      const scrollOffset = containerHeight * 0.4; // Increased from 20% to 40% from the top
      const scrollTop = sentenceElement.position().top + containerElement.scrollTop() - scrollOffset;
      
      // Scroll with animation
      containerElement.animate({
        scrollTop: scrollTop
      }, 300);
      
      // Highlight the matching sentence briefly
      sentenceElement.addClass('highlighted-sentence');
      setTimeout(() => {
        sentenceElement.removeClass('highlighted-sentence');
      }, 1500);
      
      return { 
        found: true, 
        totalMatches: matchingLabels.length, 
        currentIndex: labelIndices[labelKey]
      };
    }
    
    // Helper to show "click again" tooltip
    function showClickAgainTooltip(element, currentIndex, totalMatches) {
      // Remove any existing tooltip first
      if (clickAgainTooltip) {
        clickAgainTooltip.remove();
      }
      
      // Only show tooltip if there are multiple matches
      if (totalMatches <= 1) return;
      
      // Create the tooltip
      clickAgainTooltip = $('<div>')
        .addClass('click-again-tooltip')
        .text(`Click to see next match (${currentIndex + 1}/${totalMatches})`)
        .appendTo('body');
      
      // Position the tooltip above the element
      const elemOffset = element.offset();
      clickAgainTooltip.css({
        top: elemOffset.top - clickAgainTooltip.outerHeight() - 10,
        left: elemOffset.left + (element.outerWidth() / 2) - (clickAgainTooltip.outerWidth() / 2)
      });
      
      // Store the current element with tooltip
      currentTermWithTooltip = element;
    }
    
    // When hovering over labels or structure terms in the first paragraph (student article)
    $(document).on('mouseenter', '.student-paragraph .label, .student-paragraph .structure-term', function() {
      const $this = $(this);
      const labelText = $this.text().trim();
      const result = scrollToMatchingLabel(labelText, '#analysisResults .sentences');
      
      if (result.found && result.totalMatches > 1) {
        showClickAgainTooltip($this, result.currentIndex, result.totalMatches);
      }
    }).on('mouseleave', '.student-paragraph .label, .student-paragraph .structure-term', function() {
      // Hide the tooltip when mouse leaves
      if (clickAgainTooltip) {
        clickAgainTooltip.remove();
        clickAgainTooltip = null;
      }
    });
    
    // When hovering over labels or structure terms in the second paragraph (professional article)
    $(document).on('mouseenter', '.professional-paragraph .label, .professional-paragraph .structure-term', function() {
      const $this = $(this);
      const labelText = $this.text().trim();
      const result = scrollToMatchingLabel(labelText, '#similarArticlesList .article-card:visible .sentences');
      
      if (result.found && result.totalMatches > 1) {
        showClickAgainTooltip($this, result.currentIndex, result.totalMatches);
      }
    }).on('mouseleave', '.professional-paragraph .label, .professional-paragraph .structure-term', function() {
      // Hide the tooltip when mouse leaves
      if (clickAgainTooltip) {
        clickAgainTooltip.remove();
        clickAgainTooltip = null;
      }
    });
    
    // Make the structure terms and labels clickable to cycle through matches
    $(document).on('click', '.student-paragraph .structure-term, .student-paragraph .label', function() {
      const labelText = $(this).text().trim();
      const result = scrollToMatchingLabel(labelText, '#analysisResults .sentences', true);
      
      if (result.found && result.totalMatches > 1) {
        showClickAgainTooltip($(this), result.currentIndex, result.totalMatches);
      }
    });
    
    $(document).on('click', '.professional-paragraph .structure-term, .professional-paragraph .label', function() {
      const labelText = $(this).text().trim();
      const result = scrollToMatchingLabel(labelText, '#similarArticlesList .article-card:visible .sentences', true);
      
      if (result.found && result.totalMatches > 1) {
        showClickAgainTooltip($(this), result.currentIndex, result.totalMatches);
      }
    });
    
    // Clear tooltip when clicking elsewhere on the page
    $(document).on('click', function(e) {
      if (clickAgainTooltip && currentTermWithTooltip && 
          !$(e.target).is(currentTermWithTooltip) && 
          !$(e.target).closest(currentTermWithTooltip).length) {
        clickAgainTooltip.remove();
        clickAgainTooltip = null;
      }
    });
  }

  // Initialize label hover interactions when document is ready
  initLabelHoverInteractions();
});

// -----------------------------------------------------------
// Login-status helper
// -----------------------------------------------------------
async function checkLoginStatus() {
  const loginContainer = $("#loginContainer");
  const loadingSpinner = $("#loadingSpinner");
  const loginButton = $("a[href='/login']");
  const loginStatus = $("#loginStatus");

  try {
    loadingSpinner.show();
    loginButton.hide();
    loginStatus.text("Checking login status...");

    const resp = await fetch("/api/check_login", { method: 'GET', headers: { 'Content-Type': 'application/json' } });
    if (resp.status === 401) {
      loginContainer.show();
      $("#inputContainer, #outputContainer").hide();
      loginButton.show();
      loginStatus.text("");
    } else {
      loginContainer.hide();
      $("#inputContainer, #outputContainer").show();
    }
  } catch (err) {
    console.error('Error checking login status:', err);
    loginContainer.show();
    $("#inputContainer, #outputContainer").hide();
    loginButton.show();
    loginStatus.text("Error checking login status. Please try again.");
  } finally {
    loadingSpinner.hide();
  }
}

function formatComparisonText(text) {
  // First, preserve the original [Label] format but also add styling
  // Pattern to match [Label] format
  const labelPattern = /\[(.*?)\]/g;
  text = text.replace(labelPattern, function(match, label) {
    const trimmedLabel = label.trim();
    return match.replace(trimmedLabel, 
      `<span class="label" style="background-color: ${colorMap[trimmedLabel] || '#e9ecef'}">${trimmedLabel}</span>`
    );
  });
  
  // Then, identify structural elements in the regular text
  // Create a pattern to match all structural elements
  const structuralElements = Object.keys(colorMap);
  const structuralPattern = new RegExp(`\\b(${structuralElements.join('|')})\\b`, 'g');
  
  // Replace structural elements with spans (but not inside already created spans)
  let parts = text.split(/<span class="label"/);
  for (let i = 0; i < parts.length; i++) {
    // Skip the first part or if it's inside a span
    if (i === 0 || !parts[i].includes('</span>')) {
      // For the first part or text before a span
      let contentBeforeSpan = parts[i];
      if (i > 0) {
        // For middle parts, find the end of the span
        const endPos = parts[i].indexOf('</span>') + 7;
        contentBeforeSpan = parts[i].substring(endPos);
        parts[i] = parts[i].substring(0, endPos) + contentBeforeSpan.replace(structuralPattern, function(match) {
          return `<span class="structure-term" style="background-color: ${colorMap[match] || '#e9ecef'}">${match}</span>`;
        });
      } else {
        // For the first part
        parts[i] = parts[i].replace(structuralPattern, function(match) {
          return `<span class="structure-term" style="background-color: ${colorMap[match] || '#e9ecef'}">${match}</span>`;
        });
      }
    }
  }
  text = parts.join('<span class="label"');
  
  // Split the text into paragraphs
  const paragraphs = text.split('<br><br>').filter(p => p.trim());
  
  // Wrap each paragraph in a div with a class for targeting
  let formattedText = '';
  if (paragraphs.length >= 1) {
    formattedText += `<div class="comparison-paragraph student-paragraph">${paragraphs[0]}</div>`;
  }
  if (paragraphs.length >= 2) {
    formattedText += `<div class="comparison-paragraph professional-paragraph">${paragraphs[1]}</div>`;
  }
  
  // Add any remaining paragraphs
  for (let i = 2; i < paragraphs.length; i++) {
    formattedText += `<div class="comparison-paragraph">${paragraphs[i]}</div>`;
  }
  
  return formattedText;
}

// Helper function to get color for a label
function getLabelColor(label) {
  // Trim the label and convert to title case to match our existing colorMap
  const trimmedLabel = label.trim();
  return colorMap[trimmedLabel] || '#e9ecef'; // Default color if not found
}
