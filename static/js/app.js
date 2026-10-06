/**
 * Medical Chatbot – Frontend Application
 * RTB Cycle 16 – Generative AI Advance
 * 
 * Handles chat interactions, message rendering, history management,
 * typing indicators, and responsive sidebar behavior.
 */

(function () {
    'use strict';

    // ─── DOM Elements ───
    const elements = {
        messagesContainer: document.getElementById('messagesContainer'),
        messageInput: document.getElementById('messageInput'),
        sendBtn: document.getElementById('sendBtn'),
        welcomeScreen: document.getElementById('welcomeScreen'),
        sidebar: document.getElementById('sidebar'),
        sidebarToggle: document.getElementById('sidebarToggle'),
        mobileMenuBtn: document.getElementById('mobileMenuBtn'),
        newChatBtn: document.getElementById('newChatBtn'),
        clearChatBtn: document.getElementById('clearChatBtn'),
        chatHistoryList: document.getElementById('chatHistoryList'),
        statusIndicator: document.getElementById('statusIndicator'),
    };

    // ─── State ───
    const state = {
        messages: [],
        isLoading: false,
        conversationCount: 0,
    };

    // ─── Initialization ───
    function init() {
        bindEvents();
        checkStatus();
        autoResizeTextarea();
    }

    function bindEvents() {
        // Send message
        elements.sendBtn.addEventListener('click', sendMessage);
        elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Input state
        elements.messageInput.addEventListener('input', () => {
            const hasText = elements.messageInput.value.trim().length > 0;
            elements.sendBtn.disabled = !hasText || state.isLoading;
            autoResizeTextarea();
        });

        // Sidebar
        elements.sidebarToggle.addEventListener('click', toggleSidebar);
        elements.mobileMenuBtn.addEventListener('click', toggleMobileSidebar);

        // New chat / clear
        elements.newChatBtn.addEventListener('click', newConversation);
        elements.clearChatBtn.addEventListener('click', clearChat);

        // Suggestion chips
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.dataset.query;
                if (query) {
                    elements.messageInput.value = query;
                    elements.sendBtn.disabled = false;
                    sendMessage();
                }
            });
        });
    }

    // ─── Chat Operations ───

    async function sendMessage() {
        const message = elements.messageInput.value.trim();
        if (!message || state.isLoading) return;

        // Hide welcome screen
        if (elements.welcomeScreen) {
            elements.welcomeScreen.style.display = 'none';
        }

        // Add user message
        addMessage('user', message);
        elements.messageInput.value = '';
        elements.sendBtn.disabled = true;
        autoResizeTextarea();

        // Show typing indicator
        state.isLoading = true;
        showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();

            hideTypingIndicator();
            state.isLoading = false;

            if (response.ok) {
                addMessage('bot', data.response, data.sources, data.processing_time);
            } else {
                addMessage('bot', data.response || 'Sorry, something went wrong. Please try again.');
            }

            updateHistory(message);
        } catch (error) {
            hideTypingIndicator();
            state.isLoading = false;
            addMessage('bot', 'Unable to connect to the server. Please check that the chatbot is running and try again.');
            console.error('Chat error:', error);
        }

        // Re-enable input
        const hasText = elements.messageInput.value.trim().length > 0;
        elements.sendBtn.disabled = !hasText;
        elements.messageInput.focus();
    }

    function addMessage(role, text, sources = [], processingTime = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatarLabel = role === 'user' ? 'U' : '⚕';
        const formattedText = formatMessageText(text);

        let sourcesHTML = '';
        if (sources && sources.length > 0) {
            const sourceItems = sources.map((s, i) => `
                <div class="source-item">
                    <span class="source-score">${(s.score * 100).toFixed(1)}%</span>
                    ${escapeHtml(s.text)}
                </div>
            `).join('');

            sourcesHTML = `
                <div class="message-sources">
                    <button class="sources-toggle" onclick="this.classList.toggle('expanded'); this.nextElementSibling.classList.toggle('visible');">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                        ${sources.length} source${sources.length > 1 ? 's' : ''} referenced
                    </button>
                    <div class="sources-list">${sourceItems}</div>
                </div>
            `;
        }

        let metaHTML = '';
        if (processingTime !== null) {
            metaHTML = `<div class="message-meta"><span>${processingTime}s</span></div>`;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatarLabel}</div>
            <div class="message-content">
                <div class="message-text">${formattedText}</div>
                ${sourcesHTML}
                ${metaHTML}
            </div>
        `;

        elements.messagesContainer.appendChild(messageDiv);
        scrollToBottom();

        state.messages.push({ role, text, sources, processingTime });
    }

    function formatMessageText(text) {
        if (!text) return '';

        // Escape HTML first
        let formatted = escapeHtml(text);

        // Convert markdown-style bold
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert markdown-style italic
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Convert bullet points (• or -)
        formatted = formatted.replace(/^[•\-]\s+(.+)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Convert newlines to paragraphs
        const paragraphs = formatted.split(/\n\n+/);
        if (paragraphs.length > 1) {
            formatted = paragraphs.map(p => `<p>${p.trim()}</p>`).join('');
        }

        // Convert single newlines to <br>
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ─── Typing Indicator ───

    function showTypingIndicator() {
        let indicator = document.querySelector('.typing-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'typing-indicator';
            indicator.innerHTML = `
                <div class="message-avatar">⚕</div>
                <div class="typing-bubble">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
            elements.messagesContainer.appendChild(indicator);
        }
        indicator.classList.add('visible');
        scrollToBottom();
    }

    function hideTypingIndicator() {
        const indicator = document.querySelector('.typing-indicator');
        if (indicator) {
            indicator.classList.remove('visible');
            indicator.remove();
        }
    }

    // ─── History Management ───

    function updateHistory(message) {
        state.conversationCount++;
        const preview = message.length > 35 ? message.substring(0, 35) + '...' : message;

        // Remove empty history placeholder
        const emptyMsg = elements.chatHistoryList.querySelector('.empty-history');
        if (emptyMsg) emptyMsg.remove();

        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="min-width:14px;">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span>${escapeHtml(preview)}</span>
        `;
        elements.chatHistoryList.prepend(item);
    }

    function newConversation() {
        clearChat();
    }

    async function clearChat() {
        // Clear UI
        elements.messagesContainer.innerHTML = '';
        state.messages = [];

        // Restore welcome screen
        const welcome = createWelcomeScreen();
        elements.messagesContainer.appendChild(welcome);

        // Clear server-side history
        try {
            await fetch('/clear', { method: 'POST' });
        } catch (e) {
            console.warn('Could not clear server history:', e);
        }
    }

    function createWelcomeScreen() {
        const div = document.createElement('div');
        div.className = 'welcome-screen';
        div.id = 'welcomeScreen';
        div.innerHTML = `
            <div class="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </div>
            <h2>Welcome to Medical AI Assistant</h2>
            <p>Ask me questions about medical conditions, symptoms, treatments, and medications.</p>
            <div class="suggestion-chips">
                <button class="chip" data-query="What are the symptoms and treatment options for hypertension?">Hypertension treatment</button>
                <button class="chip" data-query="How is Type 2 diabetes diagnosed and managed?">Diabetes management</button>
                <button class="chip" data-query="What medications are used to treat asthma?">Asthma medications</button>
                <button class="chip" data-query="What are the symptoms of iron deficiency anemia?">Anemia symptoms</button>
            </div>
            <p class="disclaimer">⚕️ This chatbot provides educational information only. Always consult a healthcare professional.</p>
        `;

        // Rebind chip events
        div.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.dataset.query;
                if (query) {
                    elements.messageInput.value = query;
                    elements.sendBtn.disabled = false;
                    sendMessage();
                }
            });
        });

        return div;
    }

    // ─── Sidebar ───

    function toggleSidebar() {
        elements.sidebar.classList.toggle('collapsed');
    }

    function toggleMobileSidebar() {
        const isOpen = elements.sidebar.classList.contains('mobile-open');

        if (isOpen) {
            elements.sidebar.classList.remove('mobile-open');
            removeOverlay();
        } else {
            elements.sidebar.classList.add('mobile-open');
            addOverlay();
        }
    }

    function addOverlay() {
        let overlay = document.querySelector('.sidebar-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'sidebar-overlay visible';
            overlay.addEventListener('click', () => {
                elements.sidebar.classList.remove('mobile-open');
                removeOverlay();
            });
            document.body.appendChild(overlay);
        } else {
            overlay.classList.add('visible');
        }
    }

    function removeOverlay() {
        const overlay = document.querySelector('.sidebar-overlay');
        if (overlay) {
            overlay.classList.remove('visible');
            setTimeout(() => overlay.remove(), 200);
        }
    }

    // ─── Utilities ───

    function scrollToBottom() {
        requestAnimationFrame(() => {
            elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
        });
    }

    function autoResizeTextarea() {
        const textarea = elements.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    async function checkStatus() {
        const dot = elements.statusIndicator.querySelector('.status-dot');
        const text = elements.statusIndicator.querySelector('.status-text');

        try {
            const response = await fetch('/status');
            if (response.ok) {
                const data = await response.json();
                dot.classList.add('online');
                dot.classList.remove('error');
                text.textContent = `Ready · ${data.vector_store_count || 0} docs indexed`;
            } else if (response.status === 503) {
                text.textContent = 'Initializing...';
                // Retry after 3s
                setTimeout(checkStatus, 3000);
            } else {
                dot.classList.add('error');
                text.textContent = 'Error';
            }
        } catch {
            dot.classList.add('error');
            dot.classList.remove('online');
            text.textContent = 'Disconnected';
            // Retry after 5s
            setTimeout(checkStatus, 5000);
        }
    }

    // ─── Start ───
    document.addEventListener('DOMContentLoaded', init);
})();
