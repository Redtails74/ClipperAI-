import React, { useState } from 'react';
import './HomePage.css'; // Ensure this file exists and styles are defined

const HomePage = () => {
    const [input, setInput] = useState('');
    const [chat, setChat] = useState([]);
    const [loading, setLoading] = useState(false); // Track loading state

    const sendMessage = async () => {
        if (!input.trim()) return;

        // Update the chat to show user's message
        updateChat('User', input, 'user-message');
        
        // Show the loading message until the response is fetched
        setLoading(true);
        updateChat('AI', 'Loading...', 'loading');

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: input })
            });

            if (response.ok) {
                const data = await response.json();

                // Loop over responses and update the chat for each model's response
                Object.values(data.responses).forEach((responseMessage) => {
                    updateChat('AI', responseMessage, 'ai-message');
                });

                // Optionally, update the conversation memory as well if needed
                // data.conversation.forEach(msg => updateChat(msg.speaker, msg.message, msg.className));

            } else {
                updateChat('Error', response.statusText, 'ai-message');
            }
        } catch (error) {
            updateChat('Error', error.message, 'ai-message');
        }

        // Turn off loading after response is processed
        setLoading(false);
        setInput('');
    };

    const updateChat = (speaker, message, className) => {
        setChat((prevChat) => [...prevChat, { speaker, message, className }]);
    };

    return (
        <div id="app">
            <img src="/static/Code%20Clipper%20Logo-01.jpg" alt="Clipper Logo" /> {/* Adjust the logo path */}
            <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
            />
            <button onClick={sendMessage} disabled={loading}>
                {loading ? 'Sending...' : 'Send'}
            </button>
            <div id="chat">
                {chat.map((msg, index) => (
                    <div key={index} className={msg.className}>
                        <strong>{msg.speaker}:</strong> {msg.message}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default HomePage;
