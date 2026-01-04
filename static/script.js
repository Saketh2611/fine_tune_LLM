async function send() {
    const input = document.getElementById('user-input');
    const window = document.getElementById('chat-window');
    const msg = input.value.trim();
    
    if(!msg) return;

    // 1. Add User Message
    window.innerHTML += `<div class="message user-msg">${msg}</div>`;
    input.value = '';
    window.scrollTop = window.scrollHeight;

    // 2. Add Loading Indicator
    const loadingId = "load-" + Date.now();
    window.innerHTML += `<div class="message ai-msg" id="${loadingId}">Typing...</div>`;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        const data = await response.json();
        
        // 3. Update with AI Response
        document.getElementById(loadingId).innerText = data.response;
    } catch (error) {
        document.getElementById(loadingId).innerText = "Error connecting to AI.";
    }
    
    window.scrollTop = window.scrollHeight;
}

// Allow pressing 'Enter' to send
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') send();
});