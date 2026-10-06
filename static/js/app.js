/**
 * Medical Chatbot – Frontend Application
 * RTB Cycle 16 – Generative AI Advance
 * 
 * Handles chat interactions, message rendering, history management,
 * typing indicators, responsive sidebar behavior, and file attachments
 * (photos, PDF reports, CSV/Excel data, text) via picker, drag & drop or paste.
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
        engineBadge: document.getElementById('engineBadge'),
        // Attachments
        uploadKind: document.getElementById('uploadKind'),
        attachBtn: document.getElementById('attachBtn'),
        fileInput: document.getElementById('fileInput'),
        attachmentPreview: document.getElementById('attachmentPreview'),
        attachmentThumb: document.getElementById('attachmentThumb'),
        attachmentName: document.getElementById('attachmentName'),
        attachmentMeta: document.getElementById('attachmentMeta'),
        attachmentRemove: document.getElementById('attachmentRemove'),
        dropOverlay: document.getElementById('dropOverlay'),
    };

    const MAX_FILE_BYTES = 10 * 1024 * 1024;

    // File types accepted for each dropdown option.
    const KIND_ACCEPT = {
        auto: '.png,.jpg,.jpeg,.webp,.gif,.heic,.heif,.pdf,.csv,.xlsx,.xls,.txt,.md',
        image: '.png,.jpg,.jpeg,.webp,.gif,.heic,.heif',
        pdf: '.pdf',
        data: '.csv,.xlsx,.xls',
        text: '.txt,.md',
    };
    const KIND_ICON = { image: '📷', pdf: '📄', data: '📊', text: '📝' };
    const KIND_LABEL = { image: 'Photo', pdf: 'PDF report', data: 'Data file', text: 'Text file' };

    const BASIS_INFO = {
        KB: { label: 'Knowledge base', icon: '📚', cls: 'basis-kb', tip: 'Answer grounded in the indexed medical knowledge base' },
        GENERAL: { label: 'AI general knowledge', icon: '🧠', cls: 'basis-general', tip: 'Knowledge base had nothing relevant; answered from the AI model\'s medical knowledge' },
        BOTH: { label: 'Knowledge base + AI', icon: '🔗', cls: 'basis-both', tip: 'Combines knowledge-base passages with the AI model\'s own knowledge' },
    };

    const state = {
        messages: [],
        isLoading: false,
        conversationCount: 0,
        attachment: null,       // File object
        multimodal: false,      // true when Gemini is active
        sessions: JSON.parse(localStorage.getItem('medbot_sessions') || '[]'),
        currentSessionId: null
    };

    // Snapshot of the original welcome screen so "New conversation" restores it exactly.
    const welcomeTemplate = elements.welcomeScreen ? elements.welcomeScreen.outerHTML : '';

    // ─── Initialization ───
    function init() {
        bindEvents();
        checkStatus();
        autoResizeTextarea();
        updateAccept();
        renderSidebar();
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
            updateSendState();
            autoResizeTextarea();
        });

        // Paste an image / file straight into the chat box
        elements.messageInput.addEventListener('paste', (e) => {
            const file = e.clipboardData && e.clipboardData.files && e.clipboardData.files[0];
            if (file) {
                e.preventDefault();
                setAttachment(file);
            }
        });

        // Sidebar
        elements.sidebarToggle.addEventListener('click', toggleSidebar);
        elements.mobileMenuBtn.addEventListener('click', toggleMobileSidebar);

        // New chat / clear
        elements.newChatBtn.addEventListener('click', newConversation);
        elements.clearChatBtn.addEventListener('click', clearChat);

        // Suggestion chips (delegated so they work after the welcome screen is re-created)
        elements.messagesContainer.addEventListener('click', (e) => {
            const chip = e.target.closest('.chip');
            if (!chip) return;
            if (chip.dataset.kind) {
                elements.uploadKind.value = chip.dataset.kind;
                updateAccept();
                elements.fileInput.click();
            } else if (chip.dataset.query) {
                elements.messageInput.value = chip.dataset.query;
                updateSendState();
                sendMessage();
            }
        });

        // Attachments
        elements.attachBtn.addEventListener('click', () => elements.fileInput.click());
        elements.uploadKind.addEventListener('change', () => {
            updateAccept();
            // Re-validate an already-selected file against the new choice.
            if (state.attachment && !fileMatchesKind(state.attachment, elements.uploadKind.value)) {
                showToast(`"${state.attachment.name}" doesn't match the selected type – removed.`);
                clearAttachment();
            }
        });
        elements.fileInput.addEventListener('change', () => {
            const file = elements.fileInput.files[0];
            if (file) setAttachment(file);
            elements.fileInput.value = ''; // allow re-selecting the same file
        });
        elements.attachmentRemove.addEventListener('click', clearAttachment);

        // Drag & drop anywhere on the chat
        let dragDepth = 0;
        const main = document.querySelector('.chat-main');
        main.addEventListener('dragenter', (e) => {
            if (!hasFiles(e)) return;
            e.preventDefault();
            dragDepth++;
            elements.dropOverlay.hidden = false;
        });
        main.addEventListener('dragover', (e) => {
            if (hasFiles(e)) e.preventDefault();
        });
        main.addEventListener('dragleave', () => {
            dragDepth = Math.max(0, dragDepth - 1);
            if (dragDepth === 0) elements.dropOverlay.hidden = true;
        });
        main.addEventListener('drop', (e) => {
            if (!hasFiles(e)) return;
            e.preventDefault();
            dragDepth = 0;
            elements.dropOverlay.hidden = true;
            const file = e.dataTransfer.files[0];
            if (file) setAttachment(file);
        });
    }

    // ─── Attachments ───

    function hasFiles(e) {
        return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files');
    }

    function extOf(name) {
        const i = name.lastIndexOf('.');
        return i >= 0 ? name.slice(i).toLowerCase() : '';
    }

    function detectKind(file) {
        const ext = extOf(file.name);
        for (const kind of ['image', 'pdf', 'data', 'text']) {
            if (KIND_ACCEPT[kind].split(',').includes(ext)) return kind;
        }
        return null;
    }

    function fileMatchesKind(file, kind) {
        const detected = detectKind(file);
        return detected !== null && (kind === 'auto' || kind === detected);
    }

    function updateAccept() {
        elements.fileInput.accept = KIND_ACCEPT[elements.uploadKind.value] || KIND_ACCEPT.auto;
    }

    function setAttachment(file) {
        const kind = elements.uploadKind.value;
        const detected = detectKind(file);
        if (!detected) {
            showToast('Unsupported file type. Use an image, PDF, CSV/Excel or text file.');
            return;
        }
        if (kind !== 'auto' && kind !== detected) {
            showToast(`That looks like a ${KIND_LABEL[detected].toLowerCase()}, but "${KIND_LABEL[kind]}" is selected. Change the dropdown or pick Auto-detect.`);
            return;
        }
        if (file.size > MAX_FILE_BYTES) {
            showToast('File is too large. Maximum size is 10 MB.');
            return;
        }
        if (detected === 'image' && !state.multimodal) {
            showToast('Heads-up: image reading needs a GEMINI_API_KEY. Without it, only text questions are answered.');
        }

        state.attachment = file;
        elements.attachmentName.textContent = file.name;
        elements.attachmentMeta.textContent = `${KIND_LABEL[detected]} · ${formatBytes(file.size)}`;
        elements.attachmentThumb.innerHTML = '';
        if (detected === 'image' && /^image\/(png|jpeg|webp|gif)$/.test(file.type)) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = '';
            img.onload = () => URL.revokeObjectURL(img.src);
            elements.attachmentThumb.appendChild(img);
        } else {
            elements.attachmentThumb.textContent = KIND_ICON[detected];
        }
        elements.attachmentPreview.hidden = false;
        elements.inputWrapper && elements.inputWrapper.classList.add('has-attachment');
        updateSendState();
        elements.messageInput.focus();
    }

    function clearAttachment() {
        state.attachment = null;
        elements.attachmentPreview.hidden = true;
        elements.attachmentThumb.innerHTML = '';
        updateSendState();
    }

    function formatBytes(n) {
        if (n < 1024) return `${n} B`;
        if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
        return `${(n / 1024 / 1024).toFixed(1)} MB`;
    }

    function updateSendState() {
        const hasText = elements.messageInput.value.trim().length > 0;
        elements.sendBtn.disabled = (!hasText && !state.attachment) || state.isLoading;
    }

    // ─── Chat Operations ───

    async function sendMessage() {
        const message = elements.messageInput.value.trim();
        const file = state.attachment;
        if ((!message && !file) || state.isLoading) return;

        // Hide welcome screen
        const welcome = document.getElementById('welcomeScreen');
        if (welcome) welcome.style.display = 'none';

        // Add user message (with attachment preview)
        addMessage('user', message, { file });
        elements.messageInput.value = '';
        const kind = elements.uploadKind.value;
        clearAttachment();
        autoResizeTextarea();

        // Show typing indicator
        state.isLoading = true;
        updateSendState();
        showTypingIndicator(file ? `Reading ${file.name}…` : null);

        try {
            let response;
            if (file) {
                const form = new FormData();
                form.append('message', message);
                form.append('kind', kind);
                form.append('file', file, file.name);
                response = await fetch('/chat', { method: 'POST', body: form });
            } else {
                response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message }),
                });
            }

            let data;
            try {
                data = await response.json();
            } catch {
                data = { response: `Server error (${response.status}). Please try again.` };
            }

            hideTypingIndicator();

            if (response.ok) {
                addMessage('bot', data.response, {
                    sources: data.sources,
                    processingTime: data.processing_time,
                    basis: data.basis,
                    engine: data.engine,
                });
            } else {
                addMessage('bot', data.response || 'Sorry, something went wrong. Please try again.', { isError: true });
            }
        } catch (error) {
            hideTypingIndicator();
            addMessage('bot', 'Unable to connect to the server. Please check that the chatbot is running and try again.', { isError: true });
            console.error('Chat error:', error);
        }

        state.isLoading = false;
        updateSendState();
        elements.messageInput.focus();
    }

    function addMessage(role, text, opts = {}) {
        addMessageToDOM(role, text, opts);
        updateSession(role, text, opts);
    }

    function addMessageToDOM(role, text, opts = {}) {
        const { sources = [], processingTime = null, basis = null, engine = null, file = null, fileName, fileSize, fileKind, isError = false } = opts;
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}${isError ? ' error' : ''}`;

        const avatarLabel = role === 'user' ? 'U' : '⚕';

        // Attachment shown inside the user's bubble
        let attachmentHTML = '';
        if (file || fileName) {
            const name = fileName || (file ? file.name : '');
            const size = fileSize || (file ? file.size : 0);
            const kind = fileKind || (file ? detectKind(file) : 'text') || 'text';
            
            if (kind === 'image' && file && /^image\/(png|jpeg|webp|gif)$/.test(file.type)) {
                attachmentHTML = `<div class="msg-attachment msg-attachment-image"><img src="${URL.createObjectURL(file)}" alt="${escapeHtml(name)}"></div>`;
            } else {
                attachmentHTML = `
                    <div class="msg-attachment msg-attachment-file">
                        <span class="msg-attachment-icon">${KIND_ICON[kind] || '📎'}</span>
                        <span class="msg-attachment-text">
                            <span class="msg-attachment-name">${escapeHtml(name)}</span>
                            <span class="msg-attachment-meta">${KIND_LABEL[kind] || 'File'} · ${formatBytes(size)}</span>
                        </span>
                    </div>`;
            }
        }

        let sourcesHTML = '';
        if (sources && sources.length > 0) {
            const sourceItems = sources.map((s) => `
                <div class="source-item">
                    <span class="source-label">${escapeHtml(s.label || '')}</span>
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
        if (role === 'bot' && (processingTime !== null || basis)) {
            const b = BASIS_INFO[basis];
            const basisHTML = b ? `<span class="basis-badge ${b.cls}" title="${b.tip}">${b.icon} ${b.label}</span>` : '';
            const engineHTML = engine ? `<span class="engine-tag">${engine === 'gemini' ? 'Gemini' : 'Local model'}</span>` : '';
            const timeHTML = processingTime !== null ? `<span>${processingTime}s</span>` : '';
            metaHTML = `<div class="message-meta">${basisHTML}${engineHTML}${timeHTML}</div>`;
        }

        const textHTML = text ? `<div class="message-text">${formatMessageText(text)}</div>` : '';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatarLabel}</div>
            <div class="message-content">
                ${attachmentHTML}
                ${textHTML}
                ${sourcesHTML}
                ${metaHTML}
            </div>
        `;

        elements.messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    /**
     * Minimal, safe markdown renderer: headings, bullet & numbered lists,
     * bold/italic, inline code and [KB1]-style citations. HTML is escaped first.
     */
    function formatMessageText(text) {
        if (!text) return '';

        const inline = (s) => s
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/(^|[^*])\*(?!\s)([^*]+?)\*(?!\*)/g, '$1<em>$2</em>')
            .replace(/\[(KB\d+)\]/g, '<span class="cite">$1</span>');

        const lines = escapeHtml(text).split('\n');
        const out = [];
        let listType = null;
        let para = [];

        const flushPara = () => {
            if (para.length) {
                out.push(`<p>${inline(para.join('<br>'))}</p>`);
                para = [];
            }
        };
        const closeList = () => {
            if (listType) {
                out.push(`</${listType}>`);
                listType = null;
            }
        };

        for (const raw of lines) {
            const line = raw.trimEnd();
            let m;
            if (!line.trim()) {
                flushPara();
                closeList();
            } else if ((m = line.match(/^\s*#{1,4}\s+(.+)$/))) {
                flushPara();
                closeList();
                out.push(`<h4>${inline(m[1])}</h4>`);
            } else if ((m = line.match(/^\s*[-*•]\s+(.+)$/))) {
                flushPara();
                if (listType !== 'ul') { closeList(); out.push('<ul>'); listType = 'ul'; }
                out.push(`<li>${inline(m[1])}</li>`);
            } else if ((m = line.match(/^\s*\d+[.)]\s+(.+)$/))) {
                flushPara();
                if (listType !== 'ol') { closeList(); out.push('<ol>'); listType = 'ol'; }
                out.push(`<li>${inline(m[1])}</li>`);
            } else {
                closeList();
                para.push(line);
            }
        }
        flushPara();
        closeList();
        return out.join('');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ─── Typing Indicator ───

    function showTypingIndicator(label) {
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
                    ${label ? `<span class="typing-label">${escapeHtml(label)}</span>` : ''}
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

    // ─── Toast ───

    function showToast(message) {
        let toast = document.querySelector('.toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'toast';
            toast.setAttribute('role', 'status');
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.classList.add('visible');
        clearTimeout(showToast._t);
        showToast._t = setTimeout(() => toast.classList.remove('visible'), 4500);
    }

    // ─── History Management ───

    function saveSessions() {
        localStorage.setItem('medbot_sessions', JSON.stringify(state.sessions));
    }

    function renderSidebar() {
        elements.chatHistoryList.innerHTML = '';
        if (state.sessions.length === 0) {
            elements.chatHistoryList.innerHTML = `
                <div class="empty-history">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.4">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                    <p>No conversations yet</p>
                </div>`;
            return;
        }

        state.sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = 'history-item';
            if (state.currentSessionId === session.id) item.classList.add('active');
            
            item.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="min-width:14px;">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span>${escapeHtml(session.title)}</span>
                <button class="delete-chat-btn" aria-label="Delete chat">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            `;
            
            item.addEventListener('click', (e) => {
                if (e.target.closest('.delete-chat-btn')) {
                    e.stopPropagation();
                    deleteSession(session.id);
                } else {
                    loadSession(session.id);
                }
            });
            
            elements.chatHistoryList.appendChild(item);
        });
    }

    function deleteSession(id) {
        state.sessions = state.sessions.filter(s => s.id !== id);
        saveSessions();
        if (state.currentSessionId === id) {
            newConversation();
        } else {
            renderSidebar();
        }
    }

    async function loadSession(id) {
        const session = state.sessions.find(s => s.id === id);
        if (!session) return;
        
        state.currentSessionId = session.id;
        elements.messagesContainer.innerHTML = '';
        
        session.messages.forEach(msg => {
            addMessageToDOM(msg.role, msg.text, msg.opts);
        });
        
        if (window.innerWidth <= 768) toggleMobileSidebar();
        renderSidebar();
        
        try { await fetch('/clear', { method: 'POST' }); } catch (e) {}
    }

    function updateSession(role, text, opts = {}) {
        if (!state.currentSessionId) {
            state.currentSessionId = Date.now().toString();
            let preview = 'New Conversation';
            if (role === 'user') {
                const src = text || (opts.file ? opts.file.name : 'New Conversation');
                preview = src.length > 35 ? src.substring(0, 35) + '...' : src;
            }
            state.sessions.unshift({
                id: state.currentSessionId,
                title: preview,
                messages: []
            });
        }
        
        const session = state.sessions.find(s => s.id === state.currentSessionId);
        if (session) {
            const safeOpts = { ...opts };
            delete safeOpts.file; // File objects can't be JSON serialized
            if (opts.file) {
                safeOpts.fileName = opts.file.name;
                safeOpts.fileSize = opts.file.size;
                safeOpts.fileKind = detectKind(opts.file) || 'text';
            }
            session.messages.push({ role, text, opts: safeOpts });
            saveSessions();
            renderSidebar();
        }
    }

    function newConversation() {
        clearChat();
    }

    async function clearChat() {
        state.currentSessionId = null;
        elements.messagesContainer.innerHTML = welcomeTemplate;
        clearAttachment();
        renderSidebar();

        try {
            await fetch('/clear', { method: 'POST' });
        } catch (e) {
            console.warn('Could not clear server history:', e);
        }
    }

    // ─── Sidebar ───

    function toggleSidebar() {
        if (window.innerWidth <= 768) {
            toggleMobileSidebar();
        } else {
            elements.sidebar.classList.toggle('collapsed');
        }
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

    function setEngineBadge(data) {
        const badge = elements.engineBadge;
        if (!badge) return;
        if (data.engine === 'gemini') {
            badge.textContent = `✨ ${data.gemini && data.gemini.model_name ? data.gemini.model_name : 'Gemini'}`;
            badge.className = 'engine-badge engine-gemini';
            badge.title = 'Multimodal AI active: reads images, PDFs and data';
        } else {
            badge.textContent = '⚙️ Local model';
            badge.className = 'engine-badge engine-local';
            badge.title = 'Text-only local model. Add GEMINI_API_KEY to .env for image understanding and smarter answers.';
        }
    }

    async function checkStatus() {
        const dot = elements.statusIndicator.querySelector('.status-dot');
        const text = elements.statusIndicator.querySelector('.status-text');

        try {
            const response = await fetch('/status');
            if (response.ok) {
                const data = await response.json();
                state.multimodal = !!data.multimodal;
                setEngineBadge(data);
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
    elements.inputWrapper = document.getElementById('inputWrapper');
    document.addEventListener('DOMContentLoaded', init);
})();
