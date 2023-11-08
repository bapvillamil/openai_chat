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

        // Send the user's message to the server
        fetch('/ai-response/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken, // Include the CSRF token in the headers
            },
            body: JSON.stringify({ userMessage }),
        })
        .then(response => {
            if (response.status === 200) {
                return response.json();
            } else {
                return Promise.reject('No Model found');
            }
        })
        .then(data => {
            console.log('Data received from the API:', data);

            // Append the user's message to the chatbox
            const userMessageElement = `
                <div class="chatbox-message-item sent">
                    <span class="chatbox-message-item-text">
                        ${userMessage}
                    </span>
                    <span class="chatbox-message-item-time">${addZero(today.getHours())}:${addZero(today.getMinutes())}</span>
                </div>
            `;

            chatboxMessageWrapper.insertAdjacentHTML('beforeend', userMessageElement);

            // Append the AI's response to the chatbox
            const aiResponseElement = `
                <div class="chatbox-message-item received">
                    <span class="chatbox-message-item-text">
                        ${data.reply}
                    </span>
                    <span class="chatbox-message-item-time">${addZero(today.getHours())}:${addZero(today.getMinutes())}</span>
                </div>
            `;
            chatboxMessageWrapper.insertAdjacentHTML('beforeend', aiResponseElement);

            scrollBottom();
        })
        .catch(error => {
            console.error('Error:', error);
            const errorMessage = `
                <div class="chatbox-message-item received">
                    <span class="chatbox-message-item-text">
                        No Model found
                    </span>
                    <span class="chatbox-message-item-time">${addZero(today.getHours())}:${addZero(today.getMinutes())}</span>
                </div>
            `;
            chatboxMessageWrapper.insertAdjacentHTML('beforeend', errorMessage);
            scrollBottom();
        });

        // Clear the input field and focus
        textarea.value = '';
        textarea.rows = 1;
        textarea.focus();
        chatboxNoMessage.style.display = 'none';
    }
});

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