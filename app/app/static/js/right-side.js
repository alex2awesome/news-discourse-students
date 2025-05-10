$(document).ready(function() {
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

    // Store the original handleEvent function
    const originalHandleEvent = window.handleEvent;

    // Override the handleEvent function
    window.handleEvent = function(data) {
        // Call the original function first
        if (originalHandleEvent) {
            originalHandleEvent(data);
        }

        // Handle our specific cases
        if (data.type === 'complete') {
            console.log('Analysis complete, showing dive deeper section');
            showDiveDeeper();
        }
    };

    // Function to show dive deeper section
    function showDiveDeeper() {
        $('.dive-deeper-section').show();
    }

    // Check if analysis is already complete
    function checkAnalysisComplete() {
        // If we have analysis results and no more "Analyzing..." labels
        if ($('#analysisResults').children().length > 0 && 
            !$('.label:contains("Analyzing...")').length) {
            console.log('Analysis appears complete, showing dive deeper section');
            showDiveDeeper();
        }
    }

    // Set up periodic checking
    setInterval(checkAnalysisComplete, 1000);

    // Remove active class from all buttons and prevent main.js from setting it
    $('.source-button').removeClass('active').off('click');
    
    // Cache for article comparisons
    const comparisonCache = {};
    
    // Function to get comparison for an article
    async function getArticleComparison(article) {
        const cacheKey = `${window.currentSource}_${article.headline}`;
        
        // Return cached comparison if available
        if (comparisonCache[cacheKey]) {
            return comparisonCache[cacheKey];
        }
        
        // Show loading state
        $('#structuralComparison').html('<div class="comparison-loading">Analyzing structural differences...</div>');
        
        try {
            const studentArticle = {
                sentences: window.parsed_sentences,
                labels: window.analyzedLabels
            };
            
            const studentFormatted = formatArticleForComparison(studentArticle);
            const professionalFormatted = formatArticleForComparison(article);
            
            const response = await fetch('/api/compare_articles', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    student_article: studentFormatted,
                    professional_article: professionalFormatted
                })
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            
            const comparisonData = await response.json();
            
            // Cache the comparison
            comparisonCache[cacheKey] = comparisonData.comparison;
            
            return comparisonData.comparison;
        } catch (error) {
            console.error('Error getting comparison:', error);
            return 'Error generating comparison. Please try again.';
        }
    }
    
    // Function to display similar articles
    function displaySimilarArticles(articles, source) {
        const container = $('#similarArticlesList');
        container.empty();
        
        articles.forEach((article, index) => {
            const articleDiv = $('<div>').addClass('article-card');
            
            const headlineDiv = $('<h3>').addClass('article-headline').text(article.headline);
            articleDiv.append(headlineDiv);
            
            if (article.url) {
                const urlDiv = $('<div>').addClass('article-url');
                const urlLink = $('<a>')
                    .attr('href', article.url)
                    .attr('target', '_blank')
                    .text(article.url)
                    .addClass('article-link');
                urlDiv.append(urlLink);
                articleDiv.append(urlDiv);
            }
            
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
                    
                    if (tooltipTexts[label]) {
                        const tooltipText = '<strong>' + label + '</strong><br><br>' +
                            '<u>Why we tagged this:</u> ' + justification;
                        
                        labelSpan.attr({
                            'data-bs-toggle': 'tooltip',
                            'data-bs-placement': 'top',
                            'title': tooltipText,
                            'data-bs-html': 'true'
                        });
                        
                        new bootstrap.Tooltip(labelSpan[0], { html: true });
                    }
                    
                    sentenceDiv.append(labelSpan);
                    sentenceDiv.append($('<span>').addClass('text').text(sentence));
                    sentencesDiv.append(sentenceDiv);
                });
                articleDiv.append(sentencesDiv);
            }
            
            container.append(articleDiv);
        });

        // Reset position and update carousel
        currentPosition = 0;
        updateCarousel();
        
        // Get comparison for the first article
        if (articles.length > 0) {
            getArticleComparison(articles[0]).then(comparison => {
                $('#structuralComparison').html(formatComparisonText(comparison.replace(/\n/g, '<br>')));
            });
        }
    }
    
    // Handle source buttons
    $('.source-button').click(function() {
        const $this = $(this);
        const source = $this.data('source');
        
        // If the button is already active, deactivate it and hide similar articles
        if ($this.hasClass('active')) {
            $this.removeClass('active');
            const analysisColumn = $('.analysis-column');
            const similarColumn = $('.similar-column');
            analysisColumn.removeClass('col-6').addClass('col-12');
            similarColumn.hide();
            $('.structural-comparison-section').removeClass('show');
            return;
        }
        
        // Otherwise, activate the clicked button and deactivate others
        $('.source-button').removeClass('active');
        $this.addClass('active');
        
        // Show similar articles section
        const analysisColumn = $('.analysis-column');
        const similarColumn = $('.similar-column');
        analysisColumn.removeClass('col-12').addClass('col-6');
        similarColumn.show();
        
        // Show structural comparison section
        $('.structural-comparison-section').addClass('show');
        
        // Update current source
        window.currentSource = source;
        
        // Clear existing articles
        $('#similarArticlesList').empty();
        
        // Use cached articles if available
        if (window.cachedRetrievals && window.cachedRetrievals[source]) {
            displaySimilarArticles(window.cachedRetrievals[source], source);
        } else {
            // Fetch articles for the selected source
            fetch('/api/find_similar', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    story: $('#promptInput').val(),
                    source: source
                })
            })
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => {
                // Cache the retrievals
                if (!window.cachedRetrievals) {
                    window.cachedRetrievals = {};
                }
                window.cachedRetrievals[source] = data.articles;
                
                // Display the articles
                displaySimilarArticles(data.articles, source);
            })
            .catch(error => {
                console.error('Error fetching similar articles:', error);
                $('#structuralComparison').html(
                    '<div class="comparison-error">Error fetching articles. Please try again.</div>'
                );
            });
        }
    });

    // Carousel functionality
    let currentPosition = 0;
    
    function updateCarousel() {
        const container = $('.carousel');
        const containerWidth = $('.carousel-container').width();
        const cardWidth = containerWidth;
        const translateX = -currentPosition * cardWidth;
        
        container.css('transform', `translateX(${translateX}px)`);
        
        // Update button states
        $('#prevButton').prop('disabled', currentPosition === 0);
        $('#nextButton').prop('disabled', 
            currentPosition >= $('#similarArticlesList .article-card').length - 1);
            
        // Update comparison for current article
        if (window.cachedRetrievals && window.cachedRetrievals[window.currentSource]) {
            const currentArticle = window.cachedRetrievals[window.currentSource][currentPosition];
            if (currentArticle) {
                getArticleComparison(currentArticle).then(comparison => {
                    $('#structuralComparison').html(formatComparisonText(comparison.replace(/\n/g, '<br>')));
                });
            }
        }
    }

    // Add window resize handler
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

    // Also check when new analysis results are added
    const originalAppend = $.fn.append;
    $.fn.append = function() {
        const result = originalAppend.apply(this, arguments);
        if (this.selector === '#analysisResults') {
            checkAnalysisComplete();
        }
        return result;
    };
}); 