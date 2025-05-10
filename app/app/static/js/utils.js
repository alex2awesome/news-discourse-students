// ===========================================
// Utility functions for article analysis
// ===========================================

/**
 * Converts a string to title case
 * @param {string} str - The string to convert
 * @returns {string} The title-cased string
 */
function toTitleCase(str) {
  if (!str) return '';
  return str.toLowerCase().split(' ').map(word =>
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
}

/**
 * Formats comparison text by highlighting structural tags
 * @param {string} text - The raw comparison text
 * @returns {string} HTML with highlighted tags
 */
function formatComparisonText(text) {
  const tagVariations = {};
  Object.keys(colorMap).forEach(tag => {
    tagVariations[tag] = tag;
    tagVariations[tag.toLowerCase()] = tag;
    tagVariations[`${tag}s`] = tag;            // plural
    tagVariations[`${tag.toLowerCase()}s`] = tag;
  });

  const pattern = new RegExp(`\\b(${Object.keys(tagVariations).join('|')})\\b[.,-]?`, 'gi');

  return text.replace(pattern, match => {
    const clean = match.replace(/[.,-]$/, '');
    const canonical = tagVariations[clean] || clean;
    const colour = colorMap[canonical] || '#333';
    const textColour = ['Background Information', 'Color', 'Other'].includes(canonical) ? 'black' : 'white';
    const punctuation = match.slice(clean.length);
    return `<span class="label" style="background-color:${colour};color:${textColour};padding:2px 6px;border-radius:3px;margin:0 2px;font-size:0.9em;">${canonical}</span>${punctuation}`;
  });
}

/**
 * Formats an article for comparison
 * @param {object} article - Article with sentences and labels
 * @returns {string} Formatted string for comparison
 */
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

/**
 * Builds a sentence div with labeled content
 * @param {string} label - The structural label
 * @param {string} sentence - The sentence text
 * @param {string} justification - Justification for the label
 * @returns {jQuery} jQuery object for the sentence div
 */
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

/**
 * Checks the login status and updates UI accordingly
 * @returns {Promise<void>}
 */
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