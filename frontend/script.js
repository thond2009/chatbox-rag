const API_BASE = '/api/v1';

let sessionId = generateUUID();
let isProcessing = false;

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const documentList = document.getElementById('documentList');
const newChatBtn = document.getElementById('newChatBtn');
const processingTimeEl = document.getElementById('processingTime');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
    checkHealth();
});

chatInput.addEventListener('input', () => {
    sendBtn.disabled = !chatInput.value.trim() || isProcessing;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
});

newChatBtn.addEventListener('click', () => {
    sessionId = generateUUID();
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">?</div>
            <h2>Xin chào!</h2>
            <p>Hãy nạp tài liệu và đặt câu hỏi. Tôi sẽ trả lời dựa trên nội dung tài liệu của bạn.</p>
            <div class="suggestions">
                <span class="suggestion-chip">Tài liệu này nói về gì?</span>
                <span class="suggestion-chip">Tóm tắt nội dung chính</span>
                <span class="suggestion-chip">Các điểm quan trọng</span>
            </div>
        </div>
    `;
    processingTimeEl.textContent = '';
});

chatMessages.addEventListener('click', (e) => {
    if (e.target.classList.contains('suggestion-chip')) {
        chatInput.value = e.target.textContent;
        sendMessage();
    }
});

async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    statusDot.classList.add('thinking');
    statusText.textContent = 'Đang xử lý...';
    processingTimeEl.textContent = '';

    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    addMessage('user', query);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    const loadingMsgEl = addLoadingMessage();

    try {
        const startTime = performance.now();

        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                query: query,
                top_k: 5,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        loadingMsgEl.remove();

        addMessage('assistant', data.answer, data.sources);

        if (data.processing_time_ms) {
            processingTimeEl.textContent = `${data.processing_time_ms}ms`;
        }

    } catch (error) {
        loadingMsgEl.remove();
        addMessage('assistant', `Lỗi: ${error.message}`, []);
    } finally {
        isProcessing = false;
        sendBtn.disabled = !chatInput.value.trim();
        statusDot.classList.remove('thinking');
        statusText.textContent = 'Sẵn sàng';
    }
}

function addMessage(role, content, sources = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const renderedContent = renderMarkdown(content);
    contentDiv.innerHTML = renderedContent;

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.innerHTML = '<strong>Nguồn tham khảo:</strong>';
        sources.forEach((s, i) => {
            const fileName = s.metadata?.file_name || `Nguồn ${i + 1}`;
            sourcesDiv.innerHTML += `<span class="source-item">[${i + 1}] ${escapeHtml(fileName)}</span>`;
        });
        contentDiv.appendChild(sourcesDiv);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoadingMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return msgDiv;
}

function renderMarkdown(text) {
    let html = text;
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/^\- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    html = html.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');
    html = html.replace(/\n{2,}/g, '</p><p>');
    html = '<p>' + html + '</p>';
    html = html.replace(/<p>\s*<\/p>/g, '');
    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) {
        await uploadFile(file);
        fileInput.value = '';
    }
}

async function uploadFile(file) {
    uploadStatus.textContent = 'Đang xử lý...';
    uploadStatus.className = 'upload-status loading';
    statusDot.classList.add('thinking');
    statusText.textContent = 'Đang nạp tài liệu...';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', 'auto');

    try {
        const response = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        uploadStatus.textContent = `${file.name}: ${data.message}`;
        uploadStatus.className = 'upload-status success';

        await loadDocuments();
    } catch (error) {
        uploadStatus.textContent = `Lỗi: ${error.message}`;
        uploadStatus.className = 'upload-status error';
    } finally {
        statusDot.classList.remove('thinking');
        statusText.textContent = 'Sẵn sàng';
    }
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/documents`);
        if (!response.ok) throw new Error('Failed to load documents');

        const data = await response.json();

        if (data.documents && data.documents.length > 0) {
            documentList.innerHTML = data.documents.map(doc => `
                <div class="document-item">
                    <div class="doc-info">
                        <span class="doc-name">${escapeHtml(doc.file_name || 'Unknown')}</span>
                        <span class="doc-meta">${doc.file_type || ''} - ${doc.chunks_count || 0} chunks</span>
                    </div>
                    <button class="doc-delete" onclick="deleteDocument('${doc.document_id}')" title="Xóa">x</button>
                </div>
            `).join('');
        } else {
            documentList.innerHTML = '<div class="empty-state">Chưa có tài liệu nào</div>';
        }
    } catch (error) {
        documentList.innerHTML = '<div class="empty-state">Không thể tải danh sách</div>';
    }
}

async function deleteDocument(documentId) {
    try {
        const response = await fetch(`${API_BASE}/documents/${documentId}`, {
            method: 'DELETE',
        });

        if (!response.ok) throw new Error('Failed to delete document');

        await loadDocuments();
    } catch (error) {
        alert('Không thể xóa tài liệu: ' + error.message);
    }
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            statusDot.style.background = 'var(--success)';
            statusText.textContent = 'Sẵn sàng';
        } else {
            statusDot.style.background = 'var(--warning)';
            statusText.textContent = 'Degraded';
        }
    } catch (error) {
        statusDot.style.background = 'var(--error)';
        statusText.textContent = 'Mất kết nối';
    }
}
