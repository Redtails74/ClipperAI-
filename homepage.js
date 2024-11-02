import React, { useState } from 'react';
import './HomePage.css'; // Create and style this separately

const HomePage = () => {
    const [input, setInput] = useState('');
    const [chat, setChat] = useState([]);

    const sendMessage = async () => {
        if (!input.trim()) return;

        updateChat('User', input, 'user-message');
        updateChat('AI', 'Loading...', 'loading'); // Show loading message

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
                updateChat('AI', data.response, 'ai-message');
            } else {
                updateChat('Error', response.statusText, 'ai-message');
            }
        } catch (error) {
            updateChat('Error', error.message, 'ai-message');
        }

        setInput('');
    };

    const updateChat = (speaker, message, className) => {
        setChat(prevChat => [...prevChat, { speaker, message, className }]);
    };

    return (
        <div id="app">
            <img src="/path/to/your/logo.jpg" alt="Clipper Logo" />
            <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
            />
            <button onClick={sendMessage}>Send</button>
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
