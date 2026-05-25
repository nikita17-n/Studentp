const cards = document.querySelectorAll('.stat-card');

cards.forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.boxShadow = '0 10px 20px rgba(0,0,0,0.2)';
    });

    card.addEventListener('mouseleave', () => {
        card.style.boxShadow = 'none';
    });
});