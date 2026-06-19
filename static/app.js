document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatInput = document.getElementById('chat-input');
    const submitBtn = document.getElementById('submit-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const uploadStatusText = document.getElementById('upload-status-text');

    // Navigation Tabs
    const navDashboard = document.getElementById('nav-dashboard');
    const navCases = document.getElementById('nav-cases');
    const navDocs = document.getElementById('nav-docs');
    const navHistory = document.getElementById('nav-history');

    const viewDashboard = document.getElementById('view-dashboard');
    const viewCases = document.getElementById('view-cases');
    const viewDocs = document.getElementById('view-docs');
    const viewHistory = document.getElementById('view-history');

    const tabViews = [viewDashboard, viewCases, viewDocs, viewHistory];
    const navLinks = [navDashboard, navCases, navDocs, navHistory];

    // Domain Pills
    const domainPills = document.querySelectorAll('.domain-pill');
    let selectedDomain = 'tenant'; // Default domain

    // Active Trace & Results Views
    const resultText = document.getElementById('result-text');
    const riskBadge = document.getElementById('risk-badge');
    const citationsContainer = document.getElementById('citations-container');
    const sourcesList = document.getElementById('sources-list');

    // Right Panel Metrics
    const legalRiskVal = document.getElementById('legal-risk-val');
    const legalRiskBar = document.getElementById('legal-risk-bar');
    const financialRiskVal = document.getElementById('financial-risk-val');
    const financialRiskBar = document.getElementById('financial-risk-bar');
    const timeSensitivityVal = document.getElementById('time-sensitivity-val');
    const timeSensitivityBar = document.getElementById('time-sensitivity-bar');

    const immediateActionsList = document.getElementById('immediate-actions-list');
    const costInputTokens = document.getElementById('cost-input-tokens');
    const costOutputTokens = document.getElementById('cost-output-tokens');
    const costRagCalls = document.getElementById('cost-rag-calls');
    const costTotalCost = document.getElementById('cost-total-cost');
    const footerDisclaimer = document.getElementById('footer-disclaimer');

    // Sidebar Agent Node Elements mapping
    const agentNodes = {
        orchestrate: document.getElementById('agent-orchestrate'),
        classify: document.getElementById('agent-classify'),
        research: document.getElementById('agent-research'),
        analyze_docs: document.getElementById('agent-analyze-docs'),
        assess_risk: document.getElementById('agent-assess-risk'),
        synthesize: document.getElementById('agent-synthesize'),
        safety_check: document.getElementById('agent-safety-check')
    };

    // Center Flow Agent Node Elements mapping
    const flowNodes = {
        orchestrate: document.getElementById('flow-orchestrate'),
        classify: document.getElementById('flow-classify'),
        research: document.getElementById('flow-research'),
        analyze_docs: document.getElementById('flow-analyze-docs'),
        assess_risk: document.getElementById('flow-assess-risk'),
        synthesize: document.getElementById('flow-synthesize'),
        safety_check: document.getElementById('flow-safety-check')
    };

    // Agent execution order for pipeline animation
    const agentOrder = ['orchestrate', 'classify', 'analyze_docs', 'research', 'assess_risk', 'synthesize', 'safety_check'];

    // Default labels for each agent node
    const defaultLabels = {
        orchestrate: 'Orchestrator — idle',
        classify: 'Classifier — idle',
        research: 'Researcher — idle',
        analyze_docs: 'Doc Analyzer — idle',
        assess_risk: 'Risk Assessor',
        synthesize: 'Synthesizer',
        safety_check: 'Safety Guard'
    };

    // Running descriptions for each agent step (matches mockup style)
    const runningDescriptions = {
        orchestrate: 'Orchestrator — routing user query...',
        classify: 'Classifier — identifying legal domains & concepts...',
        analyze_docs: 'Doc Analyzer — reading uploaded documents...',
        research: 'Researcher — querying case law database...',
        assess_risk: 'Risk Assessor — scoring exposure...',
        synthesize: 'Synthesizer — framing findings...',
        safety_check: 'Safety Guard — applying disclaimers...'
    };

    // Done descriptions for each agent step (matches mockup style)
    const doneDescriptions = {
        orchestrate: 'Orchestrator — routed to tenant rights',
        classify: 'Classifier — domain identified',
        analyze_docs: 'Doc Analyzer — documents reviewed',
        research: 'Researcher — sources retrieved',
        assess_risk: 'Risk Assessor — risk calculated',
        synthesize: 'Synthesizer — findings compiled',
        safety_check: 'Safety Guard — verified & disclaimers applied'
    };

    // Configure marked options for secure rendering
    marked.setOptions({
        sanitize: true,
        breaks: true
    });

    // Domain pill click handlers
    domainPills.forEach(pill => {
        pill.addEventListener('click', () => {
            domainPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            selectedDomain = pill.getAttribute('data-domain');

            // Populate textbox with helper templates to make testing easy
            if (selectedDomain === 'tenant') {
                chatInput.value = "My landlord hasn't returned my security deposit after 52 days. California. I have receipts for all payments.";
            } else if (selectedDomain === 'employment') {
                chatInput.value = "I was fired immediately after requesting family medical leave (FMLA) to care for my sick child.";
            } else if (selectedDomain === 'contract') {
                chatInput.value = "Analyzing lease covenants details. What are my obligations and liabilities under this contract?";
            }
        });
    });

    // Set initial text template
    chatInput.value = "My landlord hasn't returned my security deposit after 52 days. California. I have receipts for all payments.";

    // Helper: Reset Agent pipeline states
    const resetAgentPipeline = () => {
        Object.keys(agentNodes).forEach(key => {
            agentNodes[key].className = 'sidebar-item pending';
        });
        Object.keys(flowNodes).forEach(key => {
            flowNodes[key].className = 'agent-node pending';
            const flowTextEl = flowNodes[key].querySelector('.flow-text');
            if (flowTextEl) flowTextEl.textContent = defaultLabels[key];
        });
    };

    // Helper: Set a specific agent node state (sidebar + flow)
    const setAgentState = (key, state) => {
        // Update sidebar
        agentNodes[key].className = `sidebar-item ${state}`;

        // Update flow node
        flowNodes[key].className = `agent-node ${state}`;

        // Update flow text based on state
        const flowTextEl = flowNodes[key].querySelector('.flow-text');
        if (flowTextEl) {
            if (state === 'running') {
                flowTextEl.textContent = runningDescriptions[key];
            } else if (state === 'done') {
                flowTextEl.textContent = doneDescriptions[key];
            } else {
                flowTextEl.textContent = defaultLabels[key];
            }
        }
    };

    // Helper: Animate Pipeline run steps in sequence (UX micro-animation)
    // This function marks all agents before `currentStep` as done, the current as running, and the rest as pending
    const animatePipelineFlow = (currentStep) => {
        const currentIdx = agentOrder.indexOf(currentStep);

        agentOrder.forEach((key, idx) => {
            if (idx < currentIdx) {
                setAgentState(key, 'done');
            } else if (idx === currentIdx) {
                setAgentState(key, 'running');
            } else {
                setAgentState(key, 'pending');
            }
        });
    };

    // Helper: Update flow node descriptions with actual API response data
    const updateFlowWithResponseData = (data) => {
        // Orchestrator — show routed domain
        const domain = data.legal_domain || 'other';
        const domainDisplay = domain.replace(/_/g, ' ');
        flowNodes.orchestrate.querySelector('.flow-text').textContent = `Orchestrator — routed to ${domainDisplay}`;

        // Classifier — show domain, jurisdiction hint, confidence
        const conf = data.confidence ? data.confidence.toFixed(2) : '0.00';
        const jurisdiction = data.likely_jurisdiction_matters ? 'jurisdiction matters' : '';
        flowNodes.classify.querySelector('.flow-text').textContent = `Classifier — ${domainDisplay} · ${jurisdiction ? jurisdiction + ' · ' : ''}confidence ${conf}`;

        // Researcher — show number of sources retrieved
        const sourceCount = data.sources ? data.sources.length : 0;
        flowNodes.research.querySelector('.flow-text').textContent = `Researcher — ${sourceCount} source${sourceCount !== 1 ? 's' : ''} retrieved`;

        // Doc Analyzer — show if documents were analyzed or skipped
        const hasDocAnalysis = data.document_analysis && Object.keys(data.document_analysis).length > 0;
        if (hasDocAnalysis) {
            flowNodes.analyze_docs.querySelector('.flow-text').textContent = 'Doc Analyzer — uploaded documents reviewed';
            flowNodes.analyze_docs.className = 'agent-node done';
        } else {
            flowNodes.analyze_docs.querySelector('.flow-text').textContent = 'Doc Analyzer — skipped (no documents)';
            flowNodes.analyze_docs.className = 'agent-node pending';
        }

        // Risk Assessor — show risk score
        flowNodes.assess_risk.querySelector('.flow-text').textContent = `Risk Assessor — score ${data.risk_score}/10`;

        // Synthesizer — findings compiled
        flowNodes.synthesize.querySelector('.flow-text').textContent = 'Synthesizer — findings compiled';

        // Safety Guard — verified
        flowNodes.safety_check.querySelector('.flow-text').textContent = 'Safety Guard — verified & disclaimers applied';
    };

    // Document Ingest status list update
    const fetchIngestedDocuments = async () => {
        try {
            const resp = await fetch('/api/documents');
            if (resp.ok) {
                const docs = await resp.json();
                if (docs.length > 0) {
                    sourcesList.innerHTML = '';
                    docs.forEach(doc => {
                        const div = document.createElement('div');
                        div.className = 'sidebar-source-item';
                        div.innerHTML = `<i class="ti ti-file-text" aria-hidden="true"></i> ${doc.filename}`;
                        sourcesList.appendChild(div);
                    });
                }
            }
        } catch (err) {
            console.error('Error listing uploaded docs:', err);
        }
    };

    // File Upload handling
    const uploadPDF = async (file) => {
        if (!file || !file.name.endsWith('.pdf')) {
            alert('Please select a valid PDF file.');
            return;
        }

        uploadStatus.style.display = 'flex';
        uploadStatusText.textContent = `Uploading "${file.name}"...`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) throw new Error('Document upload failed');

            uploadStatusText.textContent = 'Ingesting and vectorizing...';
            setTimeout(async () => {
                uploadStatus.style.display = 'none';
                await fetchIngestedDocuments();
            }, 3000);

        } catch (err) {
            alert(`Upload failed: ${err.message}`);
            uploadStatus.style.display = 'none';
        }
    };

    // Estimate token count from character count (rough: ~4 chars per token)
    const estimateTokens = (text) => {
        if (!text) return 0;
        return Math.round(text.length / 4);
    };

    // Main Analyze Query Form submission
    const analyzeQuery = async () => {
        const queryText = chatInput.value.trim();
        if (!queryText) return;

        submitBtn.disabled = true;
        submitBtn.innerHTML = 'Analyzing <i class="ti ti-loader spin"></i>';

        // Dynamic pipeline UI trigger animations (simulating server steps)
        // These provide visual feedback while the backend processes the query
        animatePipelineFlow('orchestrate');

        const animationTimers = [];
        const stepDelay = 1500;
        agentOrder.slice(1).forEach((step, idx) => {
            const timer = setTimeout(() => animatePipelineFlow(step), stepDelay * (idx + 1));
            animationTimers.push(timer);
        });

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: queryText })
            });

            if (!resp.ok) throw new Error('Server returned pipeline error');
            const data = await resp.json();

            // Clear any remaining animation timers since we have real data
            animationTimers.forEach(t => clearTimeout(t));

            // Mark all agents as done in sidebar
            Object.values(agentNodes).forEach(node => node.className = 'sidebar-item done');

            // Mark all flow nodes as done (except doc analyzer if no docs)
            Object.values(flowNodes).forEach(node => node.className = 'agent-node done');

            // Update flow node descriptions with actual response data
            updateFlowWithResponseData(data);

            // Save this inquiry and response to local storage history
            saveToHistory(queryText, data);

            // 1. Findings Text (Markdown rendered)
            resultText.innerHTML = marked.parse(data.answer);

            // 2. Risk Badge (Color & text)
            let riskClass = 'risk-low';
            let riskText = 'Low';
            if (data.risk_score >= 8) {
                riskClass = 'risk-high';
                riskText = 'High · Act now';
            } else if (data.risk_score >= 4) {
                riskClass = 'risk-medium';
                riskText = 'Medium';
            }
            riskBadge.className = `risk-badge ${riskClass}`;
            riskBadge.textContent = `Risk: ${riskText}`;

            // 3. Citations Rendering
            citationsContainer.innerHTML = '';
            if (data.sources && data.sources.length > 0) {
                data.sources.forEach(src => {
                    const citeDiv = document.createElement('div');
                    citeDiv.className = 'citation';
                    citeDiv.innerHTML = `
                        <div class="citation-source">${src.source}</div>
                        <div class="citation-text">"${src.text}"</div>
                    `;
                    citationsContainer.appendChild(citeDiv);
                });
            } else {
                citationsContainer.innerHTML = '<div class="citation"><div class="citation-source">Citations</div><div class="citation-text">No citations found.</div></div>';
            }

            // 4. Update Risk Dimensions in right panel
            const legalScore = data.risk_dimensions.legal_risk || 1;
            const financialScore = data.risk_dimensions.financial_risk || 1;
            const timeScore = data.risk_dimensions.time_sensitivity || 1;

            legalRiskVal.textContent = `${legalScore} / 10`;
            legalRiskBar.style.width = `${legalScore * 10}%`;

            financialRiskVal.textContent = `${financialScore} / 10`;
            financialRiskBar.style.width = `${financialScore * 10}%`;

            timeSensitivityVal.textContent = `${timeScore} / 10`;
            timeSensitivityBar.style.width = `${timeScore * 10}%`;

            // 5. Update Immediate Actions List
            immediateActionsList.innerHTML = '';
            if (data.immediate_actions && data.immediate_actions.length > 0) {
                data.immediate_actions.forEach(act => {
                    const item = document.createElement('div');
                    item.className = 'action-item';
                    item.innerHTML = `<i class="ti ti-circle-check" aria-hidden="true"></i> ${act}`;
                    immediateActionsList.appendChild(item);
                });
            } else {
                immediateActionsList.innerHTML = '<div class="action-item"><i class="ti ti-circle-check" aria-hidden="true"></i> No immediate actions required.</div>';
            }

            // 6. Update cost info with token estimation
            const inputTokens = estimateTokens(queryText);
            const outputTokens = estimateTokens(data.answer);
            const ragCalls = data.sources ? data.sources.length : 0;
            // Local LLM cost estimation (rough: ~$0.002/1K tokens for llama3.2)
            const estimatedCost = ((inputTokens + outputTokens) * 0.000002).toFixed(3);

            costInputTokens.textContent = inputTokens.toLocaleString();
            costOutputTokens.textContent = outputTokens.toLocaleString();
            costRagCalls.textContent = ragCalls;
            costTotalCost.textContent = `$${estimatedCost} (Local)`;

            // 7. Update Footer Disclaimer
            const jurisdiction = data.likely_jurisdiction_matters ? 'your jurisdiction' : 'California';
            footerDisclaimer.innerHTML = `<strong>Not legal advice.</strong> This is general legal information only. For your specific situation, consult a licensed attorney in ${jurisdiction}. ${data.lawyer_recommended ? '<strong>Lawyer recommended.</strong>' : ''}`;

            // Reload sidebar document list to sync ingested files
            await fetchIngestedDocuments();

        } catch (err) {
            console.error('Workflow error:', err);

            // Clear any remaining animation timers
            animationTimers.forEach(t => clearTimeout(t));

            // Reset pipeline and show error state
            resetAgentPipeline();
            flowNodes.orchestrate.querySelector('.flow-text').textContent = 'Pipeline failed — API may be rate-limited or offline';
            flowNodes.orchestrate.className = 'agent-node running';

            resultText.textContent = "Error executing multi-agent legal reasoning pipeline. Please verify your internet connection and make sure your API credentials are configured correctly.";
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Analyze <i class="ti ti-arrow-up-right"></i>';
        }
    };

    // Helper: Save query analysis results to client-side localStorage history
    const saveToHistory = (query, data) => {
        try {
            const history = JSON.parse(localStorage.getItem('lex_query_history')) || [];
            const newItem = {
                id: Date.now().toString(),
                query: query,
                date: new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }),
                domain: data.legal_domain || 'other',
                riskScore: data.risk_score || 1,
                answer: data.answer || '',
                trace: data.agent_trace || [],
                actions: data.immediate_actions || [],
                lawyerRecommended: data.lawyer_recommended || false,
                lawyerUrgency: data.lawyer_urgency || 'optional'
            };
            history.unshift(newItem);
            localStorage.setItem('lex_query_history', JSON.stringify(history));
        } catch (err) {
            console.error('Failed to save inquiry to history:', err);
        }
    };

    // Tab view switching logic
    const switchTab = (activeNav, activeView) => {
        navLinks.forEach(link => link.classList.remove('active'));
        tabViews.forEach(view => view.classList.remove('active'));
        
        activeNav.classList.add('active');
        activeView.classList.add('active');

        // Trigger updates when entering specific tabs
        if (activeNav === navDocs) {
            fetchDocumentsTab();
        } else if (activeNav === navHistory) {
            renderHistoryList();
        } else if (activeNav === navCases) {
            renderCasesGrid();
        }
    };

    // --- Tab-specific implementations ---

    // History Tab rendering and interactions
    const historyItemsList = document.getElementById('history-items-list');
    const historyDetailEmpty = document.getElementById('history-detail-empty');
    const historyDetailView = document.getElementById('history-detail-view');
    const historyDetailQuery = document.getElementById('history-detail-query');
    const historyDetailBadge = document.getElementById('history-detail-badge');
    const historyDetailAnswer = document.getElementById('history-detail-answer');
    const historyDetailTrace = document.getElementById('history-detail-trace');

    const renderHistoryList = () => {
        const history = JSON.parse(localStorage.getItem('lex_query_history')) || [];
        historyItemsList.innerHTML = '';
        
        if (history.length === 0) {
            historyItemsList.innerHTML = '<div class="history-item-placeholder">No past inquiries found.</div>';
            historyDetailEmpty.style.display = 'flex';
            historyDetailView.style.display = 'none';
            return;
        }

        history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.dataset.id = item.id;
            
            const domainText = item.domain ? item.domain.replace(/_/g, ' ') : 'other';
            div.innerHTML = `
                <div class="history-item-header">
                    <span class="history-item-title">${domainText}</span>
                    <span class="history-item-date">${item.date}</span>
                </div>
                <div class="history-item-preview">${item.query}</div>
            `;
            
            div.addEventListener('click', () => {
                document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
                div.classList.add('active');
                showHistoryDetail(item);
            });
            historyItemsList.appendChild(div);
        });
    };

    const showHistoryDetail = (item) => {
        historyDetailEmpty.style.display = 'none';
        historyDetailView.style.display = 'block';

        historyDetailQuery.textContent = item.query;
        
        let riskClass = 'risk-low';
        let riskText = 'Low';
        if (item.riskScore >= 8) {
            riskClass = 'risk-high';
            riskText = `Risk Score: ${item.riskScore}/10 (High)`;
        } else if (item.riskScore >= 4) {
            riskClass = 'risk-medium';
            riskText = `Risk Score: ${item.riskScore}/10 (Medium)`;
        } else {
            riskText = `Risk Score: ${item.riskScore}/10 (Low)`;
        }
        historyDetailBadge.className = `risk-badge ${riskClass}`;
        historyDetailBadge.textContent = riskText;

        historyDetailAnswer.innerHTML = marked.parse(item.answer);

        historyDetailTrace.innerHTML = '';
        if (item.trace && item.trace.length > 0) {
            item.trace.forEach(log => {
                const li = document.createElement('li');
                li.textContent = log;
                historyDetailTrace.appendChild(li);
            });
        } else {
            historyDetailTrace.innerHTML = '<li>No decision logs saved.</li>';
        }
    };

    // Cases Tab rendering
    const casesList = document.getElementById('cases-list');

    const renderCasesGrid = () => {
        const history = JSON.parse(localStorage.getItem('lex_query_history')) || [];
        casesList.innerHTML = '';
        
        if (history.length === 0) {
            casesList.innerHTML = `
                <div class="no-cases-card">
                    <i class="ti ti-folder-open"></i>
                    <h3>No cases analyzed yet</h3>
                    <p>Run a query in the Dashboard to analyze your legal situation and build a case file.</p>
                </div>
            `;
            return;
        }

        history.forEach(item => {
            const card = document.createElement('div');
            card.className = 'case-card';

            const lawyerRec = item.lawyerRecommended 
                ? `<span class="status-badge pending" style="margin-top:2px;">Attorney: ${item.lawyerUrgency}</span>`
                : `<span class="status-badge ingested" style="margin-top:2px;">Info Only</span>`;

            let riskLabel = 'Low';
            if (item.riskScore >= 8) riskLabel = 'High';
            else if (item.riskScore >= 4) riskLabel = 'Medium';

            const domainText = item.domain ? item.domain.replace(/_/g, ' ') : 'other';
            card.innerHTML = `
                <div class="case-card-header">
                    <span class="case-card-title">${domainText} Case</span>
                    <span class="case-card-date">${item.date}</span>
                </div>
                <div class="case-card-metrics">
                    <span>Overall Risk: <strong>${item.riskScore}/10 (${riskLabel})</strong></span>
                    ${lawyerRec}
                </div>
                <p style="font-size:12px; color:var(--text-secondary); line-height:1.4; display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical; overflow:hidden; margin: 4px 0;">
                    <strong>Situation:</strong> "${item.query}"
                </p>
                <div class="case-card-actions">
                    <h4>Next Steps</h4>
                    <div class="case-actions-list">
                        ${item.actions.slice(0, 2).map(act => `
                            <div class="case-action-bullet">
                                <i class="ti ti-circle-check"></i>
                                <span>${act}</span>
                            </div>
                        `).join('') || '<div class="case-action-bullet"><span>No actions required.</span></div>'}
                    </div>
                </div>
            `;
            casesList.appendChild(card);
        });
    };

    // Documents Tab rendering and drag-and-drop ingestion
    const docsTabFileInput = document.getElementById('docs-tab-file-input');
    const dragDropZone = document.getElementById('drag-drop-zone');
    const docsTableBody = document.getElementById('docs-table-body');

    const fetchDocumentsTab = async () => {
        try {
            const resp = await fetch('/api/documents');
            if (resp.ok) {
                const docs = await resp.json();
                docsTableBody.innerHTML = '';
                
                if (docs.length === 0) {
                    docsTableBody.innerHTML = `
                        <tr>
                            <td colspan="3" class="empty-table-cell">No documents uploaded yet. Ingest a PDF to get started.</td>
                        </tr>
                    `;
                    return;
                }

                docs.forEach(doc => {
                    const tr = document.createElement('tr');
                    const sizeFormatted = doc.size_kb ? `${doc.size_kb.toFixed(1)} KB` : '0 KB';
                    
                    let statusClass = 'pending';
                    if (doc.status === 'Ingested') statusClass = 'ingested';

                    tr.innerHTML = `
                        <td><strong>${doc.filename}</strong></td>
                        <td>${sizeFormatted}</td>
                        <td><span class="status-badge ${statusClass}">${doc.status}</span></td>
                    `;
                    docsTableBody.appendChild(tr);
                });
            }
        } catch (err) {
            console.error('Error fetching docs for table:', err);
        }
    };

    const handleUploadTab = async (file) => {
        if (!file || !file.name.endsWith('.pdf')) {
            alert('Please select a valid PDF file.');
            return;
        }

        docsTableBody.innerHTML = `
            <tr>
                <td colspan="3" class="empty-table-cell" style="color:var(--success)">
                    <div class="spinner" style="margin:0 auto 10px;"></div>
                    Ingesting "${file.name}"...
                </td>
            </tr>
        `;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) throw new Error('Document upload failed');

            setTimeout(async () => {
                await fetchDocumentsTab();
                await fetchIngestedDocuments();
            }, 3000);

        } catch (err) {
            alert(`Upload failed: ${err.message}`);
            fetchDocumentsTab();
        }
    };

    // Event Listeners
    submitBtn.addEventListener('click', analyzeQuery);

    // Tab Navigation switching event bindings
    navDashboard.addEventListener('click', (e) => { e.preventDefault(); switchTab(navDashboard, viewDashboard); });
    navCases.addEventListener('click', (e) => { e.preventDefault(); switchTab(navCases, viewCases); });
    navDocs.addEventListener('click', (e) => { e.preventDefault(); switchTab(navDocs, viewDocs); });
    navHistory.addEventListener('click', (e) => { e.preventDefault(); switchTab(navHistory, viewHistory); });

    // Drag & Drop Documents upload tab handlers
    dragDropZone.addEventListener('click', () => docsTabFileInput.click());
    docsTabFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadTab(e.target.files[0]);
        }
    });

    dragDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dragDropZone.classList.add('dragover');
    });

    dragDropZone.addEventListener('dragleave', () => {
        dragDropZone.classList.remove('dragover');
    });

    dragDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dragDropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleUploadTab(e.dataTransfer.files[0]);
        }
    });

    // Allow Enter key to submit (Shift+Enter for newline)
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeQuery();
        }
    });

    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadPDF(e.target.files[0]);
        }
    });

    // Initial setups
    fetchIngestedDocuments();
});