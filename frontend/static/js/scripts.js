document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
    const messages = document.getElementById('messages');

    sendButton.addEventListener('click', async () => {
        const inputText = userInput.value.trim();
        if (inputText === '') return;

        // Display user message
        messages.innerHTML += `<div class="user-message">${inputText}</div>`;
        userInput.value = '';

        // Send user input to backend
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: inputText }),
            });

            const data = await response.json();
            if (response.ok) {
                // Display chatbot response
                messages.innerHTML += `<div class="chatbot-response">${data.response}</div>`;
            } else {
                // Handle errors
                messages.innerHTML += `<div class="error-message">${data.error || 'An error occurred'}</div>`;
            }

            // Scroll to the bottom of the chatbox
            messages.scrollTop = messages.scrollHeight;
        } catch (error) {
            console.error('Error:', error);
            messages.innerHTML += `<div class="error-message">An error occurred</div>`;
        }
    });
});
