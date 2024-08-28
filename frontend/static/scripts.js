
const sr = ScrollReveal({
    distance: '65px',
    duration: 2600,
    delay: 450,
    reset: true,
});
sr.reveal('.hero-text', { delay: 200, origin: 'top' });
sr.reveal('.hero-img', { delay: 450, origin: 'top' });
sr.reveal('.icons', { delay: 550, origin: 'left' });
//sr.reveal('.chatbot-button', { delay: 550, origin: 'left' });

document.querySelectorAll('.circle-button').forEach(card => {
    card.addEventListener('mouseover', () => {
        card.style.backgroundColor = '#2e2e2e';
    });
    card.addEventListener('mouseout', () => {
        card.style.backgroundColor = '#1e1e1e';
    });
});

function toggleChatbot() {
    const chatbotContainer = document.getElementById('chatbot-container');
    const backgroundOverlay = document.getElementById('background-overlay');
    if (chatbotContainer.style.display === 'none' || chatbotContainer.style.display === '') {
        chatbotContainer.style.display = 'flex'; 
        backgroundOverlay.style.display = 'block'; 
    } else {
        chatbotContainer.style.display = 'none'; 
        backgroundOverlay.style.display = 'none'; 
    }
}

function sendMessage() {
    const inputField = document.getElementById('chat-input-field');
    const message = inputField.value.trim();
    if (message) {
        const chatMessages = document.querySelector('.chat-messages');
        const newMessage = document.createElement('p');
        newMessage.textContent = message;
        chatMessages.appendChild(newMessage);
        inputField.value = '';
        chatMessages.scrollTop = chatMessages.scrollHeight; 
    }
}
