// MESSAGE INPUT
const textarea = document.querySelector('.chatbox-message-input');
const chatboxForm = document.querySelector('.chatbox-message-form');
const today = new Date();

textarea.addEventListener('input', function () {
    let line = textarea.value.split('\n').length;

    if (textarea.rows < 6 || line < 6) {
        textarea.rows = line;
    }

    if (textarea.rows > 1) {
        chatboxForm.style.alignItems = 'flex-end';
    } else {
        chatboxForm.style.alignItems = 'center';
    }
});

// Your existing JavaScript code

// TOGGLE CHATBOX
const chatboxToggle = document.querySelector('.chatbox-toggle');
const chatboxMessage = document.querySelector('.chatbox-message-wrapper');

chatboxToggle.addEventListener('click', function () {
    console.log('Toggle button clicked');
    chatboxMessage.classList.toggle('show');
});


// DROPDOWN TOGGLE
const dropdownToggle = document.querySelector('.chatbox-message-dropdown-toggle');
const dropdownMenu = document.querySelector('.chatbox-message-dropdown-menu');

dropdownToggle.addEventListener('click', function () {
    dropdownMenu.classList.toggle('show');
});

document.addEventListener('click', function (e) {
    if (!e.target.matches('.chatbox-message-dropdown, .chatbox-message-dropdown *')) {
        dropdownMenu.classList.remove('show');
    }
});

// CHATBOX MESSAGE
const csrfToken = document.querySelector('input[name=csrfmiddlewaretoken]').value;
const chatboxMessageWrapper = document.querySelector('.chatbox-message-content');
const chatboxNoMessage = document.querySelector('.chatbox-message-no-message');

chatboxForm.addEventListener('submit', function (e) {
    e.preventDefault();

    if (isValid(textarea.value)) {
        const userMessage = textarea.value; // Get the user's message

        // Display user's message immediately
        displaySentMessage(userMessage);
        displayTypingIndicator(); // Show typing indicator for AI response

        // Clear the input field and focus
        textarea.value = '';
        textarea.rows = 1;
        textarea.focus();
        chatboxNoMessage.style.display = 'none';

        // Send the user's message to the server
        fetch('/ai-response/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({ userMessage }), // Ensure userMessage is sent as payload
        })
        .then(response => {
            if (response.ok) {
                return response.text(); // Response is plain text
            } else {
                throw new Error('No Model found here in my JS code');
            }
        })
        .then(data => {
            // Hide typing indicator and display AI's response after receiving it from the server
            hideTypingIndicator(); // Hide the "AI is typing..." message
            displayReceivedMessage(data);
            scrollBottom(); // Scroll to the bottom after messages are added
        })
        .catch(error => {
            console.error('Error:', error);
            // Display an error message if AI response fails
            hideTypingIndicator();
            displayReceivedMessage('No Model found');
            scrollBottom(); // Scroll to the bottom even if there's an error
        });
    }
});

function displaySentMessage(message) {
    const userMessageElement = `
        <div class="chatbox-message-item sent">
            <span class="chatbox-message-item-text">
                ${message}
            </span>
            <span class="chatbox-message-item-time">${addZero(today.getHours())}:${addZero(today.getMinutes())}</span>
        </div>
    `;

    chatboxMessageWrapper.insertAdjacentHTML('beforeend', userMessageElement);
}

function displayReceivedMessage(message) {
    const aiResponseElement = `
        <div class="chatbox-message-item received">
            <span class="chatbox-message-item-text">
                ${message}
            </span>
            <span class="chatbox-message-item-time">${addZero(today.getHours())}:${addZero(today.getMinutes())}</span>
        </div>
    `;

    chatboxMessageWrapper.insertAdjacentHTML('beforeend', aiResponseElement);
}

function displayTypingIndicator() {
    const typingIndicator = `
        <div class="chatbox-message-item received typing-indicator">
            <span class="chatbox-message-item-text typing-indicator-text">AI is typing...</span>
        </div>
    `;

    chatboxMessageWrapper.insertAdjacentHTML('beforeend', typingIndicator);
}


function hideTypingIndicator() {
    const typingIndicator = document.querySelector('.chatbox-message-item.received.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function addZero(num) {
    return num < 10 ? '0' + num : num;
}

function scrollBottom() {
    chatboxMessageWrapper.scrollTo(0, chatboxMessageWrapper.scrollHeight);
}

function isValid(value) {
    let text = value.replace(/\n/g, '');
    text = text.replace(/\s/g, '');

    return text.length > 0;
}
