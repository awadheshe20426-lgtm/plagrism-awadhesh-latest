import { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { downloadReport } from './BatchList';

interface Match {
    source_chunk: string;
    target_chunk: string;
    score?: number;
    type?: string;
    note?: string;
}
interface PlagItem {
    similar_document: string;
    similarity: number;
    matches: Match[];
    source_owner?: string | null;
    cross_batch?: boolean;
    source_uploaded_at?: string | null;
}
interface DocResult {
    document_id: string;
    filename: string;
    status: string;
    ocr?: { used: boolean; confidence: number | null; low_confidence: boolean } | null;
    ai_analysis: { score: number | null; is_ai: boolean; confidence: number | null; provider: string; advisory?: boolean; note?: string };
    plagiarism_analysis: PlagItem[];
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const riskColor = (v: number) => (v >= 0.7 ? '#ef4444' : v >= 0.3 ? '#f59e0b' : '#10b981');

const BatchResultsPage = () => {
    const { batchId } = useParams();
    const [results, setResults] = useState<DocResult[]>([]);
    const [progress, setProgress] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [expanded, setExpanded] = useState<Record<string, boolean>>({});
    const pollRef = useRef<any>(null);

    const token = () => localStorage.getItem('token');

    const loadResults = async () => {
        try {
            const r = await fetch(`/api/v1/batches/${batchId}/results`, { headers: { Authorization: `Bearer ${token()}` } });
            if (!r.ok) throw new Error('Failed to fetch results');
            const d = await r.json();
            setResults(d.data);
        } catch (e: any) { setError(e.message); }
    };

    useEffect(() => {
        if (!batchId) return;
        // Poll status until done, then load full results; keeps the page live while processing.
        const poll = async () => {
            try {
                const r = await fetch(`/api/v1/batches/${batchId}/status`, { headers: { Authorization: `Bearer ${token()}` } });
                if (!r.ok) throw new Error('Failed to fetch status');
                const s = await r.json();
                setProgress(s);
                if (s.done) {
                    if (pollRef.current) clearInterval(pollRef.current);
                    await loadResults();
                }
            } catch (e: any) { setError(e.message); if (pollRef.current) clearInterval(pollRef.current); }
        };
        poll();
        pollRef.current = setInterval(poll, 2500);
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, [batchId]);

    const processing = progress && !progress.done;

    return (
        <div className="container" style={{ padding: '40px 0' }}>
            <h1 style={{ fontSize: '32px', fontWeight: 700, marginBottom: '8px' }}>Analysis Results</h1>
            <p style={{ color: 'var(--text-secondary)', fontFamily: 'monospace', fontSize: '13px', marginBottom: '24px' }}>Batch {batchId}</p>

            {batchId && (
                <div style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
                    <button onClick={() => downloadReport(batchId, 'pdf')} className="btn-secondary" style={{ padding: '12px 24px', fontSize: '14px', cursor: 'pointer' }}>📄 Export PDF</button>
                    <button onClick={() => downloadReport(batchId, 'csv')} className="btn-secondary" style={{ padding: '12px 24px', fontSize: '14px', cursor: 'pointer' }}>📊 Export CSV</button>
                </div>
            )}

            {error && <p style={{ color: 'var(--error)' }}>⚠️ {error}</p>}

            {processing && (
                <div className="glass" style={{ padding: '24px', marginBottom: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <span style={{ fontWeight: 600 }}>Analyzing… {progress.completed}/{progress.total} documents</span>
                        <span style={{ color: 'var(--text-secondary)' }}>{progress.percent}%</span>
                    </div>
                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '999px', overflow: 'hidden' }}>
                        <div style={{ width: `${progress.percent}%`, height: '100%', background: 'var(--primary)', transition: 'width .4s' }} />
                    </div>
                </div>
            )}

            {!processing && results.length === 0 && !error && <p style={{ color: 'var(--text-secondary)' }}>No results found for this batch.</p>}

            <div style={{ display: 'grid', gap: '20px' }}>
                {results.map(doc => {
                    const top = doc.plagiarism_analysis[0];
                    const topSim = top ? top.similarity : 0;
                    return (
                        <div key={doc.document_id} className="glass" style={{ padding: '24px', borderRadius: '16px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
                                <div>
                                    <h3 style={{ fontSize: '18px', fontWeight: 700 }}>{doc.filename}</h3>
                                    {doc.ocr?.used && (
                                        <span style={{ fontSize: '12px', color: doc.ocr.low_confidence ? '#ef4444' : 'var(--text-muted)' }}>
                                            📷 OCR {doc.ocr.confidence ?? '?'}% confidence{doc.ocr.low_confidence ? ' — low, verify manually' : ''}
                                        </span>
                                    )}
                                </div>
                                <div style={{ display: 'flex', gap: '24px' }}>
                                    <div style={{ textAlign: 'center' }}>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Plagiarism</div>
                                        <div style={{ fontSize: '22px', fontWeight: 800, color: riskColor(topSim) }}>{pct(topSim)}</div>
                                    </div>
                                    <div style={{ textAlign: 'center' }}>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>AI (advisory)</div>
                                        <div style={{ fontSize: '22px', fontWeight: 800, color: riskColor(doc.ai_analysis.score ?? 0) }}>
                                            {doc.ai_analysis.score != null ? pct(doc.ai_analysis.score) : '—'}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {doc.ai_analysis.advisory && (doc.ai_analysis.score ?? 0) >= 0.5 && (
                                <p style={{ marginTop: '8px', fontSize: '12px', color: '#f59e0b' }}>⚠️ {doc.ai_analysis.note}</p>
                            )}

                            {doc.plagiarism_analysis.length === 0 ? (
                                <p style={{ marginTop: '16px', color: '#10b981', fontSize: '14px' }}>✅ No significant similarity found.</p>
                            ) : (
                                <div style={{ marginTop: '16px' }}>
                                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '10px' }}>
                                        Matched against {doc.plagiarism_analysis.length} document(s):
                                    </p>
                                    {doc.plagiarism_analysis.map((p, i) => {
                                        const key = `${doc.document_id}-${i}`;
                                        const isExact = p.matches.some(m => m.type === 'exact_duplicate');
                                        return (
                                            <div key={key} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '14px', marginBottom: '10px' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                                                    onClick={() => setExpanded(e => ({ ...e, [key]: !e[key] }))}>
                                                    <div style={{ fontSize: '14px' }}>
                                                        <span style={{ fontWeight: 700, color: riskColor(p.similarity) }}>{pct(p.similarity)}</span>
                                                        {' '}similar to <strong>{p.similar_document}</strong>
                                                        {p.source_owner && <span style={{ color: 'var(--text-muted)' }}> · by {p.source_owner}</span>}
                                                        {isExact && <span style={{ marginLeft: '8px', fontSize: '11px', background: 'rgba(239,68,68,0.2)', color: '#ef4444', padding: '2px 8px', borderRadius: '6px' }}>EXACT COPY</span>}
                                                        {p.cross_batch && <span style={{ marginLeft: '8px', fontSize: '11px', background: 'rgba(245,158,11,0.2)', color: '#f59e0b', padding: '2px 8px', borderRadius: '6px' }}>OTHER BATCH</span>}
                                                    </div>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>{expanded[key] ? '▼ hide' : `▶ ${p.matches.length} passage(s)`}</span>
                                                </div>

                                                {expanded[key] && (
                                                    <div style={{ marginTop: '12px', display: 'grid', gap: '10px' }}>
                                                        {p.matches.map((m, j) => (
                                                            <div key={j} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12.5px' }}>
                                                                <div style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: '8px', padding: '10px' }}>
                                                                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>This submission {m.score ? `· ${pct(m.score)}` : ''}</div>
                                                                    {m.source_chunk}
                                                                </div>
                                                                <div style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', borderRadius: '8px', padding: '10px' }}>
                                                                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>{p.similar_document}</div>
                                                                    {m.target_chunk}
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default BatchResultsPage;
