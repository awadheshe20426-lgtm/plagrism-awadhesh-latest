import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';

export const downloadReport = async (batchId: string, format: 'pdf' | 'csv') => {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`/api/v1/batches/${batchId}/report.${format}`, {
            headers: { 'Authorization': `Bearer ${token}` },
        });
        if (!response.ok) throw new Error(`Export failed (${response.status})`);
        const blob = await response.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `batch_${batchId.slice(0, 8)}_report.${format}`;
        a.click();
        URL.revokeObjectURL(a.href);
    } catch (e: any) {
        alert(e.message || 'Export failed');
    }
};

interface BatchRow {
    batch_id: string;
    owner: string | null;
    status: string;
    analysis_type: string;
    total_docs: number;
    transcript_count?: number;
    created_at: string | null;
}

interface TranscriptDoc {
    document_id: string;
    filename: string;
    ocr_used: boolean;
    transcript: string | null;
}

const TranscriptModal = ({ batchId, onClose }: { batchId: string; onClose: () => void }) => {
    const [docs, setDocs] = useState<TranscriptDoc[] | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const load = async () => {
            try {
                const token = localStorage.getItem('token');
                const response = await fetch(`/api/v1/batches/${batchId}/transcripts`, {
                    headers: { 'Authorization': `Bearer ${token}` },
                });
                if (!response.ok) throw new Error(`Failed to load transcripts (${response.status})`);
                const data = await response.json();
                setDocs(data.documents);
            } catch (e: any) {
                setError(e.message);
            }
        };
        load();
        const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [batchId, onClose]);

    return (
        <div onClick={onClose} style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 1000,
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px',
        }}>
            <div className="glass" onClick={e => e.stopPropagation()} style={{
                width: '100%', maxWidth: '760px', maxHeight: '85vh', overflowY: 'auto',
                padding: '24px', background: 'rgba(15, 18, 35, 0.97)',
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                    <h3 style={{ fontSize: '20px', fontWeight: 600 }}>
                        Transcripts <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{batchId.slice(0, 8)}…</span>
                    </h3>
                    <button onClick={onClose} className="btn-secondary"
                        style={{ padding: '6px 12px', fontSize: '12px', borderRadius: '8px', cursor: 'pointer' }}>
                        ✕ Close
                    </button>
                </div>
                {error && <p style={{ color: 'var(--error)', fontSize: '14px' }}>⚠️ {error}</p>}
                {!docs && !error && <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Loading…</p>}
                {docs?.map(doc => (
                    <div key={doc.document_id} style={{ marginBottom: '20px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', gap: '8px' }}>
                            <strong style={{ fontSize: '14px', wordBreak: 'break-all' }}>{doc.filename}</strong>
                            {doc.transcript && (
                                <button onClick={() => navigator.clipboard?.writeText(doc.transcript || '')} className="btn-secondary"
                                    style={{ padding: '4px 10px', fontSize: '11px', borderRadius: '6px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                                    📋 Copy
                                </button>
                            )}
                        </div>
                        {doc.transcript ? (
                            <pre style={{
                                whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '13px', lineHeight: 1.6,
                                padding: '12px', borderRadius: '8px', background: 'rgba(255,255,255,0.04)',
                                border: '1px solid rgba(255,255,255,0.08)', fontFamily: 'inherit', margin: 0,
                            }}>{doc.transcript}</pre>
                        ) : (
                            <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                                No transcript (only made for images, scanned PDFs and documents containing images).
                            </p>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

const statusColor = (status: string) =>
    status === 'completed' ? '#10b981' : status === 'failed' ? '#ef4444' : '#f59e0b';

const BatchList = ({ title, showOwner = false }: { title: string; showOwner?: boolean }) => {
    const [batches, setBatches] = useState<BatchRow[]>([]);
    const [total, setTotal] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [transcriptBatch, setTranscriptBatch] = useState<string | null>(null);
    const closeTranscript = useCallback(() => setTranscriptBatch(null), []);

    useEffect(() => {
        const fetchBatches = async () => {
            try {
                const token = localStorage.getItem('token');
                const response = await fetch('/api/v1/batches', {
                    headers: { 'Authorization': `Bearer ${token}` },
                });
                if (!response.ok) throw new Error('Failed to load batches');
                const data = await response.json();
                setBatches(data.data);
                setTotal(data.total);
            } catch (e: any) {
                setError(e.message);
            }
        };
        fetchBatches();
    }, []);

    return (
        <div className="glass" style={{ padding: '24px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ fontSize: '24px', fontWeight: 600 }}>{title}</h2>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{total} total</span>
            </div>

            {error && <p style={{ color: 'var(--error)', fontSize: '14px' }}>⚠️ {error}</p>}

            {batches.length === 0 && !error ? (
                <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>No batches yet. Upload documents to get started.</p>
            ) : (
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.1)' }}>
                                <th style={{ textAlign: 'left', padding: '12px', fontSize: '14px' }}>Date</th>
                                {showOwner && <th style={{ textAlign: 'left', padding: '12px', fontSize: '14px' }}>Owner</th>}
                                <th style={{ textAlign: 'left', padding: '12px', fontSize: '14px' }}>Batch</th>
                                <th style={{ textAlign: 'left', padding: '12px', fontSize: '14px' }}>Docs</th>
                                <th style={{ textAlign: 'left', padding: '12px', fontSize: '14px' }}>Status</th>
                                <th style={{ textAlign: 'left', padding: '12px', fontSize: '14px' }}>Transcript</th>
                                <th style={{ textAlign: 'left', padding: '12px', fontSize: '14px' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {batches.map(batch => (
                                <tr key={batch.batch_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                    <td style={{ padding: '12px', fontSize: '14px', whiteSpace: 'nowrap' }}>
                                        {batch.created_at ? new Date(batch.created_at).toLocaleString() : '-'}
                                    </td>
                                    {showOwner && <td style={{ padding: '12px', fontSize: '14px' }}>{batch.owner || '-'}</td>}
                                    <td style={{ padding: '12px', fontSize: '13px', fontFamily: 'monospace' }}>{batch.batch_id.slice(0, 8)}…</td>
                                    <td style={{ padding: '12px', fontSize: '14px' }}>{batch.total_docs}</td>
                                    <td style={{ padding: '12px', fontSize: '14px' }}>
                                        <span style={{
                                            padding: '4px 8px', borderRadius: '6px', fontSize: '12px',
                                            backgroundColor: `${statusColor(batch.status)}33`,
                                            color: statusColor(batch.status),
                                        }}>
                                            {batch.status}
                                        </span>
                                    </td>
                                    <td style={{ padding: '12px', fontSize: '14px', whiteSpace: 'nowrap' }}>
                                        {batch.transcript_count ? (
                                            <button onClick={() => setTranscriptBatch(batch.batch_id)} className="btn-secondary"
                                                style={{ padding: '6px 12px', fontSize: '12px', borderRadius: '8px', cursor: 'pointer' }}>
                                                📝 View ({batch.transcript_count})
                                            </button>
                                        ) : (
                                            <span style={{ color: 'var(--text-secondary)' }}>—</span>
                                        )}
                                    </td>
                                    <td style={{ padding: '12px', display: 'flex', gap: '8px', whiteSpace: 'nowrap' }}>
                                        <Link to={`/batch/${batch.batch_id}`} className="btn-secondary"
                                            style={{ padding: '6px 12px', fontSize: '12px', textDecoration: 'none', borderRadius: '8px' }}>
                                            View Results
                                        </Link>
                                        <button onClick={() => downloadReport(batch.batch_id, 'pdf')} className="btn-secondary"
                                            style={{ padding: '6px 12px', fontSize: '12px', borderRadius: '8px', cursor: 'pointer' }}>
                                            📄 PDF
                                        </button>
                                        <button onClick={() => downloadReport(batch.batch_id, 'csv')} className="btn-secondary"
                                            style={{ padding: '6px 12px', fontSize: '12px', borderRadius: '8px', cursor: 'pointer' }}>
                                            📊 CSV
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
            {transcriptBatch && <TranscriptModal batchId={transcriptBatch} onClose={closeTranscript} />}
        </div>
    );
};

export default BatchList;
