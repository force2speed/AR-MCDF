import React, { useState, useEffect, useRef } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8050';


// ─── Heatmap ─────────────────────────────────────────────────────────────────
const Heatmap = ({ data, scheme, animated }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!data || !data.length || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const rows = data.length, cols = data[0].length;
    const cw = canvasRef.current.width / cols;
    const ch = canvasRef.current.height / rows;
    const color = (v) => {
      const c = (r, g, b) => `rgb(${~~r},${~~g},${~~b})`;
      if (scheme === 'normal') return c(20 + v * 30, 120 + v * 80, 180 + v * 60);
      if (scheme === 'attack') return c(80 + v * 160, v * 40, v * 30);
      if (scheme === 'adv')    return c(180 + v * 60, 80 + v * 90, 20 + v * 40);
      if (scheme === 'san')    return c(v * 30, 130 + v * 110, 150 + v * 90);
      return c(v * 200, v * 200, v * 200);
    };
    if (animated) {
      let r = 0;
      const draw = () => {
        if (r >= rows) return;
        for (let c = 0; c < cols; c++) { ctx.fillStyle = color(data[r][c]); ctx.fillRect(c * cw, r * ch, cw, ch); }
        r++; requestAnimationFrame(draw);
      };
      draw();
    } else {
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) { ctx.fillStyle = color(data[r][c]); ctx.fillRect(c * cw, r * ch, cw, ch); }
    }
  }, [data, scheme, animated]);
  return (
    <div style={{ position: 'relative', overflow: 'hidden', borderRadius: '10px' }}>
      <canvas ref={canvasRef} width={400} height={160}
        style={{ width: '100%', display: 'block', border: '1px solid rgba(255,255,255,0.06)' }} />
      {animated && (
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: '100%',
          background: 'linear-gradient(to bottom, transparent, rgba(56,189,248,0.35) 50%, transparent)',
          animation: 'scanline 2s infinite linear', pointerEvents: 'none'
        }} />
      )}
    </div>
  );
};

// ─── Animated number ─────────────────────────────────────────────────────────
const AnimNum = ({ value, decimals = 3, color = '#f8fafc', size = '2rem' }) => {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const target = parseFloat(value);
    const steps = 40, interval = 16;
    let step = 0;
    const t = setInterval(() => {
      step++;
      setDisplay(+(target * (step / steps)).toFixed(decimals));
      if (step >= steps) clearInterval(t);
    }, interval);
    return () => clearInterval(t);
  }, [value, decimals]);
  return <span style={{ fontSize: size, fontWeight: 800, color, fontFamily: 'monospace' }}>{display.toFixed(decimals)}</span>;
};

// ─── Progress stepper ────────────────────────────────────────────────────────
const STEPS = ['Ingestion', 'Threat Synth', 'Sanitization', 'ST-GCN', 'Q-Bouncer'];
const STEP_COLORS = ['#38bdf8', '#fbbf24', '#a78bfa', '#f87171', '#34d399'];

const PipelineProgress = ({ active, loading }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0, marginBottom: 48, flexWrap: 'wrap', rowGap: 16 }}>
    {STEPS.map((s, i) => {
      const done = active > i + 1;
      const current = active === i + 1;
      const col = STEP_COLORS[i];
      return (
        <React.Fragment key={s}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 44, height: 44, borderRadius: '50%',
              background: done || current ? col : 'rgba(255,255,255,0.06)',
              border: `2px solid ${done || current ? col : 'rgba(255,255,255,0.1)'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.85rem', fontWeight: 700,
              color: done || current ? '#09090b' : '#475569',
              boxShadow: current ? `0 0 20px ${col}80` : 'none',
              transition: 'all 0.5s ease',
              animation: current && loading ? 'pulse 1s infinite' : 'none',
            }}>
              {done ? '✓' : i + 1}
            </div>
            <span style={{ fontSize: '0.72rem', color: done || current ? col : '#475569', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase', transition: 'color 0.5s' }}>{s}</span>
          </div>
          {i < STEPS.length - 1 && (
            <div style={{ width: 48, height: 2, background: done ? STEP_COLORS[i] : 'rgba(255,255,255,0.08)', transition: 'background 0.5s', margin: '0 4px', marginBottom: 24 }} />
          )}
        </React.Fragment>
      );
    })}
  </div>
);

// ─── Card wrapper ─────────────────────────────────────────────────────────────
const Card = ({ children, accent, style = {} }) => (
  <div style={{
    background: 'linear-gradient(145deg, rgba(30,41,59,0.85), rgba(15,23,42,0.85))',
    backdropFilter: 'blur(20px)',
    border: `1px solid ${accent ? accent + '30' : 'rgba(255,255,255,0.06)'}`,
    borderRadius: 24, padding: 32,
    boxShadow: accent ? `0 0 0 1px ${accent}20, 0 25px 50px -12px rgba(0,0,0,0.5)` : '0 25px 50px -12px rgba(0,0,0,0.5)',
    animation: 'fadeInUp 0.7s cubic-bezier(0.2,0.8,0.2,1) forwards',
    ...style,
  }}>
    {children}
  </div>
);

// ─── Metric pill ──────────────────────────────────────────────────────────────
const Metric = ({ label, value, color = '#f8fafc', decimals, pct = false }) => (
  <div style={{ background: 'rgba(0,0,0,0.35)', borderRadius: 12, padding: '14px 18px', textAlign: 'center', border: '1px solid rgba(255,255,255,0.04)' }}>
    <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 6 }}>{label}</div>
    {decimals != null
      ? <AnimNum value={value} decimals={decimals} color={color} size="1.6rem" />
      : <div style={{ fontSize: '1.6rem', fontWeight: 800, color, fontFamily: 'monospace' }}>{pct ? `${value}%` : value}</div>}
  </div>
);

// ─── Status badge ─────────────────────────────────────────────────────────────
const Badge = ({ children, color }) => (
  <span style={{
    display: 'inline-block', padding: '5px 14px', borderRadius: 999,
    background: color + '20', border: `1px solid ${color}50`,
    color, fontSize: '0.78rem', fontWeight: 700, letterSpacing: '1px',
  }}>{children}</span>
);

// ─── Prediction block ─────────────────────────────────────────────────────────
const PredBlock = ({ title, data, accentColor }) => (
  <div style={{ background: accentColor + '08', borderRadius: 16, padding: 24, border: `1px solid ${accentColor}25`, flex: 1 }}>
    <Badge color={accentColor}>{title}</Badge>
    <div style={{ marginTop: 20, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      <Metric label="Attack Prob" value={data.prob_attack} decimals={4} color={accentColor} />
      <Metric label="Normal Prob" value={data.prob_normal} decimals={4} color="#94a3b8" />
    </div>
    <div style={{
      marginTop: 16, padding: 18, borderRadius: 12,
      background: data.prediction === 'ATTACK' ? 'rgba(239,68,68,0.12)' : 'rgba(52,211,153,0.12)',
      border: `1px solid ${data.prediction === 'ATTACK' ? 'rgba(239,68,68,0.4)' : 'rgba(52,211,153,0.4)'}`,
      textAlign: 'center',
    }}>
      <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 6 }}>ST-GCN Prediction</div>
      <div style={{ fontSize: '1.8rem', fontWeight: 900, color: data.prediction === 'ATTACK' ? '#ef4444' : '#34d399', letterSpacing: '2px' }}>
        {data.prediction}
      </div>
      {data.fooled && (
        <div style={{ marginTop: 8, fontSize: '0.78rem', color: '#ef4444', fontWeight: 600 }}>
          ⚠ Adversarial noise evaded the model
        </div>
      )}
      {data.caught_by_sanitizer && (
        <div style={{ marginTop: 8, fontSize: '0.78rem', color: '#34d399', fontWeight: 600 }}>
          ✓ Flagged by Stage 3 Sanitizer before ST-GCN
        </div>
      )}
    </div>
  </div>
);

// ─── Q-Action card ────────────────────────────────────────────────────────────
const ACTION_COLOR = { 0: '#38bdf8', 1: '#fbbf24', 2: '#f97316', 3: '#ef4444' };
const QAction = ({ label, data }) => (
  <div style={{ flex: 1, background: 'rgba(0,0,0,0.3)', borderRadius: 16, padding: 24, border: '1px solid rgba(255,255,255,0.05)' }}>
    <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 12 }}>{label}</div>
    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: ACTION_COLOR[data.action], marginBottom: 16 }}>
      {data.action_name}
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
      <Metric label="Threat Level" value={data.threat_level} decimals={4} color="#fca5a5" />
      <Metric label="Isolation Score" value={data.isolation_score} decimals={4} color="#c4b5fd" />
    </div>
  </div>
);

// ─── Main Demo ────────────────────────────────────────────────────────────────
export default function Demo() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [stage, setStage] = useState(0);

  const run = async () => {
    setLoading(true); setError(null); setResult(null); setStage(0);
    try {
      const res = await fetch(`${API_URL}/api/run-demo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_stage: 5 }),
      });
      const data = await res.json();
      if (!data.ok) { setError(data.error || 'Simulation failed'); }
      else {
        setResult(data);
        setTimeout(() => setStage(1), 400);
        setTimeout(() => setStage(2), 2200);
        setTimeout(() => setStage(3), 4000);
        setTimeout(() => setStage(4), 5800);
        setTimeout(() => setStage(5), 7600);
      }
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; }
        @keyframes scanline { 0%{transform:translateY(-100%)} 100%{transform:translateY(100%)} }
        @keyframes fadeInUp { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.45} }
        @keyframes glow { 0%,100%{box-shadow:0 0 20px #3b82f640} 50%{box-shadow:0 0 40px #3b82f690} }
        body { margin: 0; background: #09090b; }
      `}</style>

      <div style={{ minHeight: '100vh', background: '#09090b', backgroundImage: 'radial-gradient(ellipse 80% 50% at 50% -10%, #1e1b4b88 0%, transparent 70%)', color: '#f8fafc', padding: '56px 24px', fontFamily: '"Inter", system-ui, sans-serif' }}>
        <div style={{ maxWidth: 1320, margin: '0 auto' }}>

          {/* Header */}
          <header style={{ textAlign: 'center', marginBottom: 52 }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', background: 'rgba(56,189,248,0.08)', padding: '6px 18px', borderRadius: 999, border: '1px solid rgba(56,189,248,0.2)', marginBottom: 20 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#38bdf8', marginRight: 10, boxShadow: '0 0 10px #38bdf8', animation: loading ? 'pulse 1s infinite' : 'none' }} />
              <span style={{ fontSize: '0.82rem', color: '#38bdf8', fontWeight: 600, letterSpacing: '0.8px' }}>
                {loading ? 'PIPELINE EXECUTING...' : result ? 'ANALYSIS COMPLETE' : 'SYSTEM READY'}
              </span>
            </div>
            <h1 style={{ fontSize: 'clamp(2.2rem, 5vw, 3.8rem)', fontWeight: 900, background: 'linear-gradient(135deg, #fff 30%, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', margin: '0 0 12px', letterSpacing: '-1.5px' }}>
              AR-MCDF Defense Matrix
            </h1>
            <p style={{ color: '#475569', fontSize: '1.05rem', margin: 0 }}>
              Adversarial-Resilient Multi-Cloud Defense Framework · Live Pipeline Demo
            </p>
          </header>

          {/* Pipeline stepper */}
          <PipelineProgress active={stage} loading={loading} />

          {/* Run button */}
          <div style={{ textAlign: 'center', marginBottom: 56 }}>
            <button onClick={run} disabled={loading} style={{
              padding: '15px 44px', fontSize: '1rem', fontWeight: 700,
              background: loading ? '#1e293b' : 'linear-gradient(135deg, #3b82f6, #6366f1)',
              color: loading ? '#475569' : '#fff', border: 'none', borderRadius: 999,
              cursor: loading ? 'wait' : 'pointer',
              boxShadow: loading ? 'none' : '0 10px 30px -8px #3b82f680',
              transition: 'all 0.25s', animation: !loading && !result ? 'glow 2s infinite' : 'none',
              letterSpacing: '0.3px',
            }}>
              {loading ? '⏳  Executing Full Pipeline...' : result ? '🔄  Run Again' : '▶  Run Analysis Sequence'}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div style={{ padding: 20, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5', borderRadius: 14, textAlign: 'center', maxWidth: 640, margin: '0 auto 40px' }}>
              <strong>Pipeline Error:</strong> {error}
            </div>
          )}

          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>

              {/* ── Row 1: Module 1 + Module 2 + Module 3 ── */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 28 }}>

                {/* Module 1 */}
                {stage >= 1 && (
                  <Card accent="#38bdf8">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
                      <Badge color="#38bdf8">Module 1</Badge>
                    </div>
                    <h2 style={{ color: '#fff', fontSize: '1.45rem', margin: '10px 0 4px', fontWeight: 700 }}>Network Ingestion</h2>
                    <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: 24 }}>UNSW-NB15 traffic windowing · 50-packet frames</p>

                    <div style={{ marginBottom: 18 }}>
                      <p style={{ color: '#38bdf8', fontSize: '0.82rem', fontWeight: 600, margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: '0.6px' }}>Normal Baseline</p>
                      <Heatmap data={result.stage1.normal_heatmap} scheme="normal" animated />
                    </div>
                    <div style={{ marginBottom: 22 }}>
                      <p style={{ color: '#fca5a5', fontSize: '0.82rem', fontWeight: 600, margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: '0.6px' }}>Malicious Threat Profile</p>
                      <Heatmap data={result.stage1.attack_heatmap} scheme="attack" animated />
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                      <Metric label="Total Windows" value={result.stage1.total_windows} />
                      <Metric label="Attack Ratio" value={result.stage1.attack_ratio.toFixed(1)} color="#fca5a5" pct />
                    </div>
                  </Card>
                )}

                {/* Module 2 */}
                {stage >= 2 && (
                  <Card accent="#fbbf24">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
                      <Badge color="#fbbf24">Module 2</Badge>
                    </div>
                    <h2 style={{ color: '#fff', fontSize: '1.45rem', margin: '10px 0 4px', fontWeight: 700 }}>Threat Synthesizer</h2>
                    <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: 24 }}>FGSM adversarial perturbation · ε = 0.10</p>

                    <div style={{ marginBottom: 22 }}>
                      <p style={{ color: '#fbbf24', fontSize: '0.82rem', fontWeight: 600, margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: '0.6px' }}>Adversarial Signature — Evades AI</p>
                      <Heatmap data={result.stage2.adv_heatmap} scheme="adv" animated />
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                      <Metric label="L2 Norm Dist" value={result.stage2.l2_norm} decimals={3} color="#fbbf24" />
                      <Metric label="L-inf Bound"  value={result.stage2.linf}    decimals={3} color="#f8fafc" />
                    </div>
                    <Metric label="Features Altered" value={result.stage2.pct_altered.toFixed(1)} color="#fca5a5" pct />
                  </Card>
                )}

                {/* Module 3 */}
                {stage >= 3 && (
                  <Card accent="#a78bfa">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
                      <Badge color="#a78bfa">Module 3</Badge>
                    </div>
                    <h2 style={{ color: '#fff', fontSize: '1.45rem', margin: '10px 0 4px', fontWeight: 700 }}>Noise Sanitizer</h2>
                    <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: 24 }}>Gaussian smoothing filter · σ = 0.9 · τ = 0.85</p>

                    <div style={{ marginBottom: 22 }}>
                      <p style={{ color: '#a78bfa', fontSize: '0.82rem', fontWeight: 600, margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: '0.6px' }}>Sanitized Output</p>
                      <Heatmap data={result.stage3.san_heatmap} scheme="san" animated />
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                      <Metric label="L2 Before" value={result.stage3.l2_before} decimals={3} color="#94a3b8" />
                      <Metric label="L2 After"  value={result.stage3.l2_after}  decimals={3} color="#a78bfa" />
                    </div>
                    <div style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: 18, borderRadius: 14,
                      background: result.stage3.flagged ? 'rgba(239,68,68,0.1)' : 'rgba(52,211,153,0.1)',
                      border: `1px solid ${result.stage3.flagged ? 'rgba(239,68,68,0.35)' : 'rgba(52,211,153,0.35)'}`,
                    }}>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 6 }}>Sanitizer Verdict</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 800, color: result.stage3.flagged ? '#ef4444' : '#34d399' }}>
                          {result.stage3.flagged ? '⚠ ADVERSARIAL DETECTED' : '✓ CLEAN TRAFFIC'}
                        </div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: 4 }}>Confidence</div>
                        <AnimNum value={result.stage3.confidence} decimals={4} color="#a78bfa" size="1.5rem" />
                      </div>
                    </div>
                  </Card>
                )}
              </div>

              {/* ── Row 2: Module 4 ST-GCN Detection ── */}
              {stage >= 4 && (
                <Card accent="#f87171">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
                    <Badge color="#f87171">Module 4</Badge>
                  </div>
                  <h2 style={{ color: '#fff', fontSize: '1.6rem', margin: '10px 0 4px', fontWeight: 700 }}>ST-GCN Threat Detection</h2>
                  <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: 32 }}>
                    Spatio-Temporal Graph Convolutional Network · Cosine-similarity graph edges · side-by-side pipeline comparison
                  </p>
                  <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                    <PredBlock title="UNPROTECTED PATH — NO SANITIZER" data={result.stage4.without_defense} accentColor="#ef4444" />
                    <PredBlock title="PROTECTED PATH — AR-MCDF ACTIVE" data={result.stage4.with_defense}    accentColor="#34d399" />
                  </div>
                  {result.stage4.without_defense.prediction === 'NORMAL' && result.stage4.with_defense.prediction === 'ATTACK' && (
                    <div style={{ marginTop: 24, padding: 18, background: 'rgba(52,211,153,0.06)', borderRadius: 12, border: '1px solid rgba(52,211,153,0.2)', textAlign: 'center' }}>
                      <span style={{ color: '#34d399', fontWeight: 700 }}>✓ Defense Effective — </span>
                      <span style={{ color: '#94a3b8' }}>Sanitizer stripped adversarial camouflage, restoring correct ST-GCN classification.</span>
                    </div>
                  )}
                </Card>
              )}

              {/* ── Row 3: Module 5 Q-Bouncer ── */}
              {stage >= 5 && (
                <Card accent="#34d399">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
                    <Badge color="#34d399">Module 5</Badge>
                  </div>
                  <h2 style={{ color: '#fff', fontSize: '1.6rem', margin: '10px 0 4px', fontWeight: 700 }}>Q-Learning Bouncer</h2>
                  <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: 32 }}>
                    Tabular RL agent · Isolation Score Is = 0.65·Threat + 0.35·Resource · Bellman Q-update
                  </p>
                  <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                    <QAction label="UNPROTECTED PATH — NO SANITIZER" data={result.stage5.without_defense} />
                    <QAction label="PROTECTED PATH — AR-MCDF ACTIVE"  data={result.stage5.with_defense}   />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 24 }}>
                    {[['THROTTLE', '#38bdf8'], ['REROUTE', '#fbbf24'], ['ISOLATE', '#f97316'], ['BLACKLIST', '#ef4444']].map(([a, c]) => (
                      <div key={a} style={{ background: c + '15', borderRadius: 10, padding: '10px 6px', textAlign: 'center', border: `1px solid ${c}30` }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: 700, color: c, letterSpacing: '0.8px' }}>{a}</div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

            </div>
          )}
        </div>
      </div>
    </>
  );
}
