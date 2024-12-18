const colorMap = {
  "Headline": "#1f77b4",
  "Introduction": "#ff7f0e",
  "Lede": "#2ca02c",
  "Nut graf": "#d62728",
  "Background information": "#9467bd",
  "Opinion": "#8c564b",
  "Color": "#e377c2",
  "Transition": "#7f7f7f",
  "Supporting detail": "#bcbd22",
  "Sourcing/source information": "#17becf"
};

async function checkLoginStatus() {
  const response = await fetch("/api/ask", {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({})
  });
  if (response.status === 401) {
    $("#loginContainer").show();
    $("#inputContainer").hide();
    $("#outputContainer").hide();
  } else {
    $("#loginContainer").hide();
    $("#inputContainer").show();
    $("#outputContainer").show();
  }
}

$("#sendButton").on("click", async function() {
  const userPrompt = $("#promptInput").val().trim();
  if (!userPrompt) {
    alert("Please enter a prompt.");
    return;
  }
  $("#stopButton").show();
  $("#sendButton").hide();
  $("#outputContainer").empty();
  const container = $("<div>").addClass("container");
  $("#outputContainer").append(container);
  
  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ story: userPrompt })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let sentences = [];

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      let sentences_written = false;

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          
          switch(data.type) {
            case 'clean_text':
              $("#promptInput").val(data.text);
              break;
            case 'sentences':
              sentences = data.sentences;
              sentences.forEach((sentence, index) => {
                const row = $("<div>").addClass("row mb-3").attr('id', `sentence-${index}`);
                const labelCol = $("<div>").addClass("col-md-3").text("Analyzing...");
                const sentenceCol = $("<div>").addClass("col-md-9").text(sentence);
                row.append(labelCol, sentenceCol);
                container.append(row);
              });
              sentences_written = true;
              break;
            case 'analysis':
              updateSentenceLabel(data.index, data.analysis, colorMap[data.analysis] || "#ffffff");
              break;
            case 'error':
              if (!sentences_written) {
                displayOutput("Error: " + data.message);
                return;
              } else {
                updateSentenceLabel(data.index, "Error: " + data.message, "#ff0000");
              }
              break;
            case 'complete':
              $("#sendButton").show();
              return;
            case 'stopped':
              if (!sentences_written) {
                displayOutput("Analysis stopped by user.");
              } else {
                updateSentenceLabel(data.index, "Stopped", "#ff0000");
              }
              return;
          }
        }
      }
    }

  } catch (error) {
    console.error("Request error:", error);
    displayOutput("Error: " + error.message);
  } finally {
    $("#stopButton").hide();
    $("#sendButton").show();
  }
});

$("#stopButton").on("click", async function() {
  try {
    const response = await fetch('/api/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    displayOutput("Analysis stopped by user.");
    $("#stopButton").hide();
    $("#sendButton").show();
  } catch (error) {
    console.error("Stop request error:", error);
    displayOutput("Error stopping analysis: " + error.message);
  }
});

function displayOutput(text) {
  const p = $("<p>").text(text);
  $("#outputContainer").append(p);
}

function updateSentenceLabel(index, text, backgroundColor = null) {
  const row = $(`#sentence-${index}`);
  const labelCol = row.find('.col-md-3');
  labelCol.text(text);
  if (backgroundColor) {
    labelCol.css("background-color", backgroundColor);
  }
}