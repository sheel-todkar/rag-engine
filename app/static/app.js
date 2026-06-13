document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    lucide.createIcons();

    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const collectionInput = document.getElementById('collection-name');
    const uploadStatus = document.getElementById('upload-status');
    const progressFill = document.getElementById('progress-fill');
    const statusTitle = document.getElementById('status-title');
    const statusPercent = document.getElementById('status-percent');
    const statusDetails = document.getElementById('status-details');
    
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const chatForm = document.getElementById('chat-form');
    const chatViewport = document.getElementById('chat-viewport');
    const chatWelcome = document.getElementById('chat-welcome');
    
    const sourceInspector = document.getElementById('source-inspector');
    const inspectorContent = document.getElementById('inspector-content');
    const closeInspectorBtn = document.getElementById('close-inspector-btn');

    // Pipeline steps for visualizer
    const stepUpload = document.getElementById('step-upload');
    const stepEmbed = document.getElementById('step-embed');
    const stepRetrieve = document.getElementById('step-retrieve');
    const stepLlm = document.getElementById('step-llm');

    let isDocumentLoaded = false;
    let currentSources = [];

    // --- Ingestion Flow & UI states ---

    // Click to select file
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag-over styling states
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped file
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Handle selected file via dialogue
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            alert('Please upload a PDF file only.');
            return;
        }

        // Show status & reset pipeline indicators
        uploadStatus.classList.remove('hidden');
        progressFill.style.width = '0%';
        statusTitle.textContent = 'Uploading...';
        statusPercent.textContent = '0%';
        statusDetails.textContent = `File: ${file.name}`;
        
        // Active Chunking Visual Step
        resetPipelineSteps();
        stepUpload.classList.add('active');

        const formData = new FormData();
        formData.append('file', file);
        
        const collectionName = collectionInput.value.trim() || 'documents';

        try {
            // Fake progress animation since it's local and fast
            let progress = 0;
            const interval = setInterval(() => {
                if (progress < 85) {
                    progress += 15;
                    progressFill.style.width = `${progress}%`;
                    statusPercent.textContent = `${progress}%`;
                }
            }, 100);

            const response = await fetch(`/api/upload?collection_name=${encodeURIComponent(collectionName)}`, {
                method: 'POST',
                body: formData
            });

            clearInterval(interval);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Upload failed');
            }

            const data = await response.json();
            
            // Ingestion pipeline active step shifts to embedding complete
            progressFill.style.width = '100%';
            statusPercent.textContent = '100%';
            statusTitle.textContent = 'Ingestion Complete';
            statusDetails.textContent = `Successfully split into ${data.chunks_created} overlapping chunks and stored in ChromaDB!`;
            
            stepEmbed.classList.add('active');

            // Enable query inputs
            isDocumentLoaded = true;
            queryInput.disabled = false;
            sendBtn.disabled = false;
            queryInput.placeholder = 'Ask a question about the loaded PDF...';
            queryInput.focus();

        } catch (error) {
            console.error('Upload error:', error);
            statusTitle.textContent = 'Ingestion Failed';
            statusDetails.textContent = error.message;
            progressFill.style.background = 'var(--error)';
            resetPipelineSteps();
        }
    }

    // --- Query Flow & Chat Operations ---

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query || !isDocumentLoaded) return;

        // Add user query to UI
        appendMessage(query, 'user');
        queryInput.value = '';

        // Reset welcome message on first question
        if (chatWelcome) {
            chatWelcome.style.display = 'none';
        }

        // Active pipeline retrieval step
        resetPipelineSteps();
        stepUpload.classList.add('active');
        stepEmbed.classList.add('active');
        stepRetrieve.classList.add('active');

        // Add assistant thinking/loading UI bubble
        const loadingId = appendLoadingBubble();

        const collectionName = collectionInput.value.trim() || 'documents';

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: query,
                    collection_name: collectionName
                })
            });

            // Remove loading bubble
            document.getElementById(loadingId).remove();

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get answer');
            }

            const data = await response.json();

            // Active pipeline generation step
            stepLlm.classList.add('active');

            // Add final answer bubble
            appendMessage(data.answer, 'assistant', data.sources);

        } catch (error) {
            console.error('Query error:', error);
            document.getElementById(loadingId).remove();
            appendMessage(`Error generating response: ${error.message}`, 'assistant', [], true);
            resetPipelineSteps();
            stepUpload.classList.add('active');
            stepEmbed.classList.add('active');
        }
    });

    function appendMessage(text, sender, sources = [], isError = false) {
        const row = document.createElement('div');
        row.classList.add('message-row', sender);

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble');
        bubble.textContent = text;

        if (isError) {
            bubble.style.borderLeft = '4px solid var(--error)';
            bubble.style.color = 'var(--error)';
        }

        if (sender === 'assistant' && sources && sources.length > 0) {
            const meta = document.createElement('div');
            meta.classList.add('message-meta');
            meta.innerHTML = `<i data-lucide="database" style="width:12px; height:12px;"></i> Found ${sources.length} matching sources`;

            const button = document.createElement('button');
            button.classList.add('inspect-sources-btn');
            button.innerHTML = '<i data-lucide="eye" style="width:12px; height:12px;"></i> Inspect Sources';
            
            button.addEventListener('click', () => {
                showSources(sources);
            });

            meta.appendChild(button);
            bubble.appendChild(meta);
        }

        row.appendChild(bubble);
        chatViewport.appendChild(row);
        
        // Auto scroll to bottom
        chatViewport.scrollTop = chatViewport.scrollHeight;
        
        // Initialize dynamic Lucide icons inside bubble meta
        lucide.createIcons({
            attrs: {
                class: 'lucide-custom'
            }
        });
    }

    function appendLoadingBubble() {
        const id = 'loading-' + Date.now();
        const row = document.createElement('div');
        row.classList.add('message-row', 'assistant');
        row.id = id;

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble');

        const dots = document.createElement('div');
        dots.classList.add('loading-dots');
        dots.innerHTML = '<span></span><span></span><span></span>';

        bubble.appendChild(dots);
        row.appendChild(bubble);
        chatViewport.appendChild(row);
        chatViewport.scrollTop = chatViewport.scrollHeight;

        return id;
    }

    // --- Context Sources Inspector ---

    function showSources(sources) {
        inspectorContent.innerHTML = '';
        sources.forEach((source, index) => {
            const card = document.createElement('div');
            card.classList.add('source-card');
            
            const header = document.createElement('div');
            header.classList.add('source-card-header');
            header.innerHTML = `<span>Source Chunk #${index + 1}</span> <span>Cosine Sim</span>`;
            
            const body = document.createElement('p');
            body.textContent = source;

            card.appendChild(header);
            card.appendChild(body);
            inspectorContent.appendChild(card);
        });

        sourceInspector.classList.remove('hidden');
    }

    closeInspectorBtn.addEventListener('click', () => {
        sourceInspector.classList.add('hidden');
    });

    function resetPipelineSteps() {
        [stepUpload, stepEmbed, stepRetrieve, stepLlm].forEach(step => {
            step.classList.remove('active');
        });
    }
});
