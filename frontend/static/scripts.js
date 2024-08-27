
const sr = ScrollReveal({
    distance: '65px',
    duration: 2600,
    delay: 450,
    reset: true,
});
sr.reveal('.hero-text', { delay: 200, origin: 'top' });
sr.reveal('.hero-img', { delay: 450, origin: 'top' });
sr.reveal('.icons', { delay: 550, origin: 'left' });
sr.reveal('.circle-button', { delay: 550, origin: 'left' });

document.querySelectorAll('.circle-button').forEach(card => {
    card.addEventListener('mouseover', () => {
        card.style.backgroundColor = '#2e2e2e';
    });
    card.addEventListener('mouseout', () => {
        card.style.backgroundColor = '#1e1e1e';
    });
});

