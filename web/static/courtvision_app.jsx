const { useState, useEffect, useMemo } = React;

const C = {
  bg: "#08090e",
  bg2: "#0c0e16",
  surface: "#111520",
  card: "#151a27",
  cardHv: "#1a2030",
  border: "rgba(255,255,255,0.06)",
  borderMd: "rgba(255,255,255,0.14)",
  orange: "#f07a28",
  orangeFaint: "rgba(240,122,40,0.12)",
  blue: "#38bdf8",
  blueFaint: "rgba(56,189,248,0.12)",
  green: "#4ade80",
  red: "#f87171",
  yellow: "#fbbf24",
  text: "#e6e1d6",
  muted: "#8792a2",
  faint: "rgba(255,255,255,0.04)",
};

const FONTS = `
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: ${C.bg}; color: ${C.text}; font-family: 'Barlow', 'Segoe UI', sans-serif; }
  select { appearance: none; }
  button:focus { outline: none; }
  input:focus, select:focus { outline: none; border-color: rgba(240,122,40,0.6) !important; }
`;

const BOOTSTRAP = window.COURTVISION_BOOTSTRAP || {
  error: null,
  form: {
    session_name: "",
    mode: "game",
    analysis_profile: "preview",
    player_id: "",
    account_id: "",
    input: "",
  },
  workout_accounts: [],
  analysis_profiles: [
    { id: "preview", label: "Fast Preview" },
    { id: "full", label: "Full Run" },
  ],
};

// ─── MOCK DATA ────────────────────────────────────────────────────────────────

const PLAYERS = [
  { id: 4,  label: "#4  Williams", team: 1, touches: 28, posS: 187, att: 11, made: 7  },
  { id: 12, label: "#12 Johnson",  team: 1, touches: 19, posS: 124, att:  7, made: 3  },
  { id: 23, label: "#23 Davis",    team: 2, touches: 24, posS: 156, att:  9, made: 5  },
  { id: 5,  label: "#5  Miller",   team: 2, touches: 15, posS:  98, att:  5, made: 2  },
  { id: 11, label: "#11 Brown",    team: 1, touches: 21, posS: 142, att:  8, made: 4  },
];

const ALL_SHOTS = [
  // Paint
  { x:248, y:390, made:true,  pid:4  }, { x:268, y:382, made:true,  pid:4  },
  { x:232, y:378, made:false, pid:12 }, { x:272, y:371, made:true,  pid:4  },
  { x:243, y:363, made:false, pid:23 }, { x:280, y:387, made:true,  pid:11 },
  { x:218, y:389, made:true,  pid:12 }, { x:255, y:356, made:false, pid:5  },
  // Mid-range
  { x:155, y:338, made:false, pid:4  }, { x:345, y:330, made:true,  pid:23 },
  { x:195, y:313, made:true,  pid:11 }, { x:305, y:318, made:false, pid:5  },
  { x:250, y:302, made:true,  pid:4  }, { x:178, y:354, made:false, pid:23 },
  { x:322, y:350, made:true,  pid:12 }, { x:130, y:316, made:false, pid:11 },
  { x:370, y:322, made:true,  pid:4  },
  // Corner 3s
  { x:48,  y:398, made:true,  pid:11 }, { x:52,  y:380, made:false, pid:5  },
  { x:452, y:393, made:true,  pid:4  }, { x:448, y:408, made:false, pid:23 },
  // Above-break 3s
  { x:98,  y:286, made:false, pid:4  }, { x:142, y:258, made:true,  pid:11 },
  { x:208, y:232, made:false, pid:12 }, { x:250, y:224, made:true,  pid:23 },
  { x:292, y:238, made:false, pid:4  }, { x:358, y:263, made:true,  pid:5  },
  { x:402, y:282, made:false, pid:11 }, { x:345, y:234, made:true,  pid:4  },
  { x:168, y:245, made:false, pid:23 },
];

const HOT_ZONES = [
  { x:170, y:280, w:160, h:140, made:9,  att:14, lbl:"Paint"        },
  { x:0,   y:350, w:170, h:105, made:3,  att:7,  lbl:"Left Corner"  },
  { x:330, y:350, w:170, h:105, made:4,  att:8,  lbl:"Right Corner" },
  { x:0,   y:228, w:170, h:122, made:2,  att:6,  lbl:"Left Mid"     },
  { x:330, y:228, w:170, h:122, made:4,  att:9,  lbl:"Right Mid"    },
  { x:170, y:228, w:160, h:52,  made:2,  att:5,  lbl:"Top Mid"      },
  { x:0,   y:123, w:163, h:105, made:3,  att:8,  lbl:"Left Wing ×3" },
  { x:337, y:123, w:163, h:105, made:4,  att:9,  lbl:"Right Wing ×3"},
  { x:163, y:123, w:174, h:105, made:4,  att:11, lbl:"Top ×3"       },
];

// ─── BASKETBALL COURT SVG ─────────────────────────────────────────────────────
const BX = 250, BY = 415, R3 = 235;
const ARC_Y = Math.round(BY - Math.sqrt(R3 * R3 - (470 - BX) * (470 - BX)));

function CourtSVG({ shots = [], filterShot = "all", showZones = false }) {
  const lc = "rgba(255,255,255,0.18)";
  const [hov, setHov] = useState(null);

  const visible = shots.filter(s =>
    filterShot === "all" || (filterShot === "made" && s.made) || (filterShot === "missed" && !s.made)
  );

  return (
    <svg viewBox="0 0 500 460" style={{ width:"100%", display:"block", borderRadius:10 }}>
      <rect x={0} y={0} width={500} height={460} fill="#0d1220" rx={10} />

      {/* Hot zones */}
      {showZones && HOT_ZONES.map((z, i) => {
        const p = z.made / z.att;
        const hot  = p >= 0.5;
        const cold = p < 0.35;
        const alpha = 0.12 + Math.abs(p - 0.42) * 0.7;
        const fill = hot ? `rgba(239,68,68,${alpha})` : cold ? `rgba(59,130,246,${alpha})` : `rgba(251,191,36,${alpha * 0.7})`;
        return (
          <g key={i}>
            <rect x={z.x} y={z.y} width={z.w} height={z.h} fill={fill} />
            <text x={z.x+z.w/2} y={z.y+z.h/2-7} textAnchor="middle" dominantBaseline="middle"
              fill="white" fontSize={13} fontWeight="700" fontFamily="Barlow Condensed,sans-serif" opacity={0.95}>
              {Math.round(p*100)}%
            </text>
            <text x={z.x+z.w/2} y={z.y+z.h/2+9} textAnchor="middle" dominantBaseline="middle"
              fill="rgba(255,255,255,0.55)" fontSize={9} fontFamily="Barlow,sans-serif">
              {z.made}/{z.att}
            </text>
          </g>
        );
      })}

      {/* Key (paint) */}
      <rect x={170} y={280} width={160} height={175} fill="rgba(255,255,255,0.025)" stroke={lc} strokeWidth={1} />

      {/* Free throw line */}
      <line x1={170} y1={280} x2={330} y2={280} stroke={lc} strokeWidth={1} />
      {/* FT circle upper */}
      <path d="M 190 280 A 60 60 0 0 1 310 280" fill="none" stroke={lc} strokeWidth={1} />
      {/* FT circle lower (dashed) */}
      <path d="M 190 280 A 60 60 0 0 0 310 280" fill="none" stroke={lc} strokeWidth={1} strokeDasharray="5 4" />

      {/* Restricted area */}
      <path d={`M ${BX-40} ${BY} A 40 40 0 0 0 ${BX+40} ${BY}`} fill="none" stroke={lc} strokeWidth={1} />

      {/* Corner 3-point lines */}
      <line x1={30}  y1={455} x2={30}  y2={ARC_Y} stroke={lc} strokeWidth={1} />
      <line x1={470} y1={455} x2={470} y2={ARC_Y} stroke={lc} strokeWidth={1} />

      {/* 3-point arc */}
      <path d={`M 30 ${ARC_Y} A ${R3} ${R3} 0 0 1 470 ${ARC_Y}`} fill="none" stroke={lc} strokeWidth={1} />

      {/* Backboard + rim */}
      <line x1={222} y1={382} x2={278} y2={382} stroke="rgba(255,255,255,0.38)" strokeWidth={2.5} />
      <circle cx={BX} cy={BY} r={14} fill="none" stroke={C.orange} strokeWidth={2} />

      {/* Court outline */}
      <rect x={1} y={1} width={498} height={458} fill="none" stroke={lc} strokeWidth={1} rx={8} />

      {/* Shot markers */}
      {visible.map((s, i) => (
        <g key={i} style={{ cursor:"pointer" }}
          onMouseEnter={() => setHov(i)}
          onMouseLeave={() => setHov(null)}>
          <circle cx={s.x} cy={s.y} r={hov===i ? 9 : 7}
            fill={s.made ? "rgba(74,222,128,0.92)" : "rgba(248,113,113,0.92)"}
            stroke={s.made ? "#16a34a" : "#dc2626"}
            strokeWidth={1.5}
            style={{ transition:"r 0.12s" }} />
          {!s.made && (
            <>
              <line x1={s.x-3.5} y1={s.y-3.5} x2={s.x+3.5} y2={s.y+3.5} stroke="#b91c1c" strokeWidth={1.5} />
              <line x1={s.x+3.5} y1={s.y-3.5} x2={s.x-3.5} y2={s.y+3.5} stroke="#b91c1c" strokeWidth={1.5} />
            </>
          )}
          {hov===i && (
            <g>
              <rect x={s.x+10} y={s.y-18} width={72} height={22} rx={4}
                fill="#1a2030" stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
              <text x={s.x+46} y={s.y-4} textAnchor="middle"
                fill="white" fontSize={9.5} fontFamily="Barlow,sans-serif">
                #{PLAYERS.find(p=>p.id===s.pid)?.id ?? s.pid} · {s.made?"Made":"Miss"}
              </text>
            </g>
          )}
        </g>
      ))}
    </svg>
  );
}

// ─── REUSABLE UI ──────────────────────────────────────────────────────────────

function Tag({ label, color }) {
  const bg = color === "orange" ? C.orangeFaint : C.blueFaint;
  const fg = color === "orange" ? C.orange : C.blue;
  const bd = color === "orange" ? "rgba(240,122,40,0.3)" : "rgba(56,189,248,0.3)";
  return (
    <span style={{ background:bg, color:fg, border:`1px solid ${bd}`, borderRadius:20,
      fontSize:"0.7rem", letterSpacing:"0.08em", padding:"2px 10px", fontWeight:600,
      textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif" }}>
      {label}
    </span>
  );
}

function PillToggle({ options, value, onChange }) {
  return (
    <div style={{ display:"flex", background:C.surface, borderRadius:8,
      border:`1px solid ${C.border}`, overflow:"hidden" }}>
      {options.map(([v, lab, acc]) => (
        <button key={v} onClick={() => onChange(v)}
          style={{ padding:"0.38rem 0.85rem", border:"none",
            background: value===v ? (acc === "green" ? "rgba(74,222,128,0.12)" : acc === "red" ? "rgba(248,113,113,0.12)" : C.orangeFaint) : "transparent",
            color: value===v ? (acc === "green" ? C.green : acc === "red" ? C.red : C.orange) : C.muted,
            cursor:"pointer", fontFamily:"Barlow,sans-serif", fontSize:"0.8rem",
            fontWeight: value===v ? 600 : 400, transition:"all 0.12s", textTransform:"capitalize" }}>
          {lab}
        </button>
      ))}
    </div>
  );
}

function BarStat({ label, value, max, color }) {
  const pct = Math.round(value / max * 100);
  return (
    <div>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
        <span style={{ fontSize:"0.8rem", color:C.muted }}>{label}</span>
        <span style={{ fontSize:"0.8rem", fontWeight:600, color }}>{value}/{max}</span>
      </div>
      <div style={{ height:4, borderRadius:2, background:"rgba(255,255,255,0.07)" }}>
        <div style={{ height:"100%", borderRadius:2, width:`${pct}%`, background:color, transition:"width 0.4s" }} />
      </div>
    </div>
  );
}

function MetCard({ label, value, accent, sub }) {
  return (
    <div style={{ background:C.card, borderRadius:14, padding:"1.1rem 1.25rem",
      border:`1px solid ${C.border}` }}>
      <div style={{ fontSize:"0.7rem", color:C.muted, letterSpacing:"0.08em",
        textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
        {label}
      </div>
      <div style={{ fontFamily:"Bebas Neue,'Impact',sans-serif", fontSize:"2rem",
        letterSpacing:"0.02em", color:accent || C.text, lineHeight:1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize:"0.72rem", color:C.muted, marginTop:3 }}>{sub}</div>}
    </div>
  );
}

// ─── SCREENS ─────────────────────────────────────────────────────────────────

function WelcomeScreen({ onPick }) {
  const [hov, setHov] = useState(null);
  return (
    <div style={{ minHeight:"100vh", background:C.bg, display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center", padding:"2rem" }}>
      {/* Wordmark */}
      <div style={{ marginBottom:"3.5rem", textAlign:"center" }}>
        <div style={{ display:"flex", alignItems:"center", gap:12, justifyContent:"center", marginBottom:"0.75rem" }}>
          <svg width={38} height={38} viewBox="0 0 38 38">
            <circle cx={19} cy={19} r={18} fill={C.orange} />
            <circle cx={19} cy={19} r={18} fill="none" stroke="rgba(0,0,0,0.15)" strokeWidth={1} />
            <line x1={1} y1={19} x2={37} y2={19} stroke="rgba(0,0,0,0.3)" strokeWidth={1.2} />
            <path d="M 19 1 Q 26 10 26 19 Q 26 28 19 37" fill="none" stroke="rgba(0,0,0,0.3)" strokeWidth={1.2} />
            <path d="M 19 1 Q 12 10 12 19 Q 12 28 19 37" fill="none" stroke="rgba(0,0,0,0.3)" strokeWidth={1.2} />
          </svg>
          <h1 style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"2.6rem",
            letterSpacing:"0.06em", color:C.text, lineHeight:1 }}>
            COURT<span style={{ color:C.orange }}>VISION</span>
          </h1>
        </div>
        <p style={{ color:C.muted, fontSize:"0.82rem", letterSpacing:"0.14em",
          textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif" }}>
          No-wearable basketball analytics
        </p>
      </div>

      {/* Mode cards */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1.25rem",
        width:"100%", maxWidth:540 }}>
        {[
          {
            id:"workout",
            title:"Workout",
            desc:"Track a single player or small group. Enter player IDs manually before the session.",
            accent:C.orange,
            icon:<svg width={32} height={32} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}><circle cx={12} cy={7} r={4}/><path d="M6 21v-2a6 6 0 0 1 12 0v2"/></svg>
          },
          {
            id:"game",
            title:"Game",
            desc:"Full game tracking. Computer vision auto-detects all players and both teams.",
            accent:C.blue,
            icon:<svg width={32} height={32} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}><circle cx={12} cy={12} r={10}/><path d="M12 2Q18 7 18 12Q18 17 12 22M12 2Q6 7 6 12Q6 17 12 22M2 12H22"/></svg>
          },
        ].map(m => (
          <button key={m.id} onClick={() => onPick(m.id)}
            onMouseEnter={() => setHov(m.id)}
            onMouseLeave={() => setHov(null)}
            style={{ background: hov===m.id ? C.cardHv : C.card,
              border:`1px solid ${hov===m.id ? m.accent : C.border}`,
              borderRadius:18, padding:"2rem 1.5rem", cursor:"pointer", textAlign:"left",
              transition:"all 0.18s", transform: hov===m.id ? "translateY(-3px)" : "none" }}>
            <div style={{ color:m.accent, marginBottom:"1rem" }}>{m.icon}</div>
            <div style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.65rem",
              letterSpacing:"0.04em", color:C.text, marginBottom:"0.5rem" }}>
              {m.title}
            </div>
            <div style={{ color:C.muted, fontSize:"0.83rem", lineHeight:1.55 }}>{m.desc}</div>
          </button>
        ))}
      </div>

      <p style={{ color:"rgba(255,255,255,0.16)", fontSize:"0.72rem", marginTop:"3.5rem",
        letterSpacing:"0.1em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif" }}>
        iPad · NFHS Feed · No wearables required
      </p>
    </div>
  );
}

function SetupScreen({
  mode,
  onBack,
  onStart,
  initialValues,
  workoutAccounts,
  analysisProfiles,
  errorMessage,
}) {
  const [name, setName] = useState(initialValues.session_name || "");
  const [source, setSource] = useState(initialValues.input || "");
  const [profile, setProfile] = useState(initialValues.analysis_profile || "preview");
  const [accountId, setAccountId] = useState(initialValues.account_id || "");
  const [playerId, setPlayerId] = useState(initialValues.player_id || "");
  const [t1, setT1] = useState("Home");
  const [t2, setT2] = useState("Away");
  const [error, setError] = useState(errorMessage || "");
  const acc = mode === "workout" ? C.orange : C.blue;

  const inp = {
    background:C.surface, border:`1px solid ${C.border}`, borderRadius:10,
    padding:"0.7rem 1rem", color:C.text, fontSize:"0.9rem",
    fontFamily:"Barlow,sans-serif", width:"100%",
  };

  useEffect(() => {
    setError(errorMessage || "");
  }, [errorMessage]);

  function handleSubmit(event) {
    event.preventDefault();

    const trimmedSource = source.trim();
    const trimmedPlayerId = playerId.trim();
    const trimmedT1 = t1.trim();
    const trimmedT2 = t2.trim();
    const fallbackName = mode === "workout"
      ? "Workout Session"
      : trimmedT1 && trimmedT2
        ? `${trimmedT1} vs ${trimmedT2}`
        : "Game Session";
    const trimmedName = name.trim() || fallbackName;

    if (!trimmedSource) {
      setError("Enter a video path or URL.");
      return;
    }

    if (mode === "workout" && !accountId) {
      setError("Select a workout account.");
      return;
    }

    if (mode === "workout" && !trimmedPlayerId) {
      setError("Enter a player number.");
      return;
    }

    setError("");
    onStart({
      input: trimmedSource,
      session_name: trimmedName,
      mode,
      analysis_profile: profile,
      player_id: mode === "workout" ? trimmedPlayerId : "",
      account_id: mode === "workout" ? accountId : "",
    });
  }

  return (
    <div style={{ minHeight:"100vh", background:C.bg, display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center", padding:"2rem" }}>
      <form onSubmit={handleSubmit} style={{ width:"100%", maxWidth:480, background:C.card,
        borderRadius:22, border:`1px solid ${C.border}`, padding:"2.25rem" }}>
        <button type="button" onClick={onBack}
          style={{ background:"none", border:"none", color:C.muted, cursor:"pointer",
            fontSize:"0.82rem", marginBottom:"1.5rem", padding:0,
            fontFamily:"Barlow,sans-serif", display:"flex", alignItems:"center", gap:5 }}>
          ← Back
        </button>

        <div style={{ marginBottom:"2rem" }}>
          <Tag label={mode} color={mode==="workout" ? "orange" : "blue"} />
          <h2 style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"2.2rem",
            letterSpacing:"0.04em", color:C.text, marginTop:"0.5rem" }}>
            {mode === "workout" ? "Setup Workout" : "Setup Game"}
          </h2>
        </div>

        <div style={{ marginBottom:"1.25rem" }}>
          <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
            letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
            Session Name
          </label>
          <input value={name} onChange={e=>setName(e.target.value)}
            placeholder={mode==="workout" ? "Morning Shooting Drill" : "Varsity Practice — Fri"}
            style={inp} />
        </div>

        {mode === "workout" ? (
          <>
            <div style={{ marginBottom:"1.25rem" }}>
              <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
                letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
                Workout Account
              </label>
              <select value={accountId} onChange={e=>setAccountId(e.target.value)} style={inp}>
                <option value="">Select account</option>
                {workoutAccounts.map(account => (
                  <option key={account.id} value={account.id}>{account.label}</option>
                ))}
              </select>
            </div>

            <div style={{ marginBottom:"1.25rem" }}>
              <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
                letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
                Player Number
              </label>
              <input value={playerId} onChange={e=>setPlayerId(e.target.value)}
                placeholder="12"
                style={inp} />
            </div>
          </>
        ) : (
          <div style={{ marginBottom:"1.25rem" }}>
            <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
              letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
              Teams
            </label>
            <div style={{ display:"flex", gap:8 }}>
              <input value={t1} onChange={e=>setT1(e.target.value)} placeholder="Home" style={inp} />
              <input value={t2} onChange={e=>setT2(e.target.value)} placeholder="Away" style={inp} />
            </div>
          </div>
        )}

        <div style={{ marginBottom:"1.25rem" }}>
          <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
            letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
            Analysis Profile
          </label>
          <div style={{ display:"grid", gridTemplateColumns:`repeat(${analysisProfiles.length}, minmax(0, 1fr))`, gap:8 }}>
            {analysisProfiles.map(option => (
              <button
                key={option.id}
                type="button"
                onClick={() => setProfile(option.id)}
                style={{
                  background: profile === option.id ? C.orangeFaint : C.surface,
                  border:`1px solid ${profile === option.id ? "rgba(240,122,40,0.38)" : C.border}`,
                  borderRadius:10,
                  color: profile === option.id ? C.orange : C.text,
                  cursor:"pointer",
                  fontFamily:"Barlow,sans-serif",
                  fontSize:"0.82rem",
                  fontWeight:600,
                  padding:"0.8rem 0.9rem",
                  textAlign:"left",
                }}>
                <div style={{ fontFamily:"Barlow Condensed,sans-serif", fontSize:"0.74rem",
                  letterSpacing:"0.08em", marginBottom:4, textTransform:"uppercase" }}>
                  {option.id}
                </div>
                <div>{option.label}</div>
              </button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom:"1.75rem" }}>
          <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
            letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
            Video Source
          </label>
          <input
            value={source}
            onChange={e=>setSource(e.target.value)}
            placeholder="Input_vids/video_1.mp4 or https://..."
            style={inp}
          />
          <div style={{ color:C.muted, fontSize:"0.75rem", lineHeight:1.5, marginTop:"0.55rem" }}>
            Use a local path already available to the server or a supported video URL.
          </div>
        </div>

        {error && (
          <div style={{ marginBottom:"1.25rem" }}>
            <div style={{ background:"rgba(248,113,113,0.1)", border:"1px solid rgba(248,113,113,0.24)",
              borderRadius:10, color:"#fecaca", fontSize:"0.82rem", lineHeight:1.5, padding:"0.75rem 0.9rem" }}>
              {error}
            </div>
          </div>
        )}

        <button type="submit"
          style={{ width:"100%", padding:"0.95rem", background:acc, border:"none",
            borderRadius:12, color: mode==="workout" ? "#1a0800" : "#00111f",
            fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.15rem",
            letterSpacing:"0.06em", cursor:"pointer", fontWeight:700 }}>
          START SESSION →
        </button>
      </form>
    </div>
  );
}

// ─── DASHBOARD TABS ───────────────────────────────────────────────────────────

function LiveFeedTab() {
  return (
    <div style={{ background:C.card, borderRadius:18, border:`1px solid ${C.border}`,
      padding:"5rem 2rem", display:"flex", flexDirection:"column",
      alignItems:"center", gap:"1.25rem" }}>
      <div style={{ width:72, height:72, borderRadius:"50%", background:C.orangeFaint,
        border:`2px dashed rgba(240,122,40,0.3)`, display:"flex", alignItems:"center",
        justifyContent:"center" }}>
        <svg width={30} height={30} viewBox="0 0 24 24" fill="none" stroke={C.orange} strokeWidth={1.5}>
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
      </div>
      <div style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.3rem",
        letterSpacing:"0.06em", color:C.muted }}>
        Live Feed
      </div>
      <p style={{ color:C.muted, fontSize:"0.87rem", textAlign:"center", maxWidth:320, lineHeight:1.6 }}>
        Connect a video source from the setup screen. The processed feed with player tracking, 
        shot detection, and tactical overlays will appear here.
      </p>
      <div style={{ display:"flex", gap:10, marginTop:"0.5rem" }}>
        {["iPad Stream","NFHS Upload","File Import"].map(s => (
          <div key={s} style={{ background:C.surface, border:`1px solid ${C.border}`,
            borderRadius:10, padding:"0.55rem 1.1rem", fontSize:"0.8rem", color:C.muted }}>
            {s}
          </div>
        ))}
      </div>
    </div>
  );
}

function ShotChartTab() {
  const [pid, setPid]  = useState("all");
  const [sht, setSht]  = useState("all");

  const shots = useMemo(() =>
    pid === "all" ? ALL_SHOTS : ALL_SHOTS.filter(s => String(s.pid) === String(pid))
  , [pid]);

  const made   = shots.filter(s=>s.made).length;
  const missed = shots.length - made;
  const fgPct  = shots.length > 0 ? Math.round(made/shots.length*100) : 0;

  // zone calc
  const paint  = shots.filter(s => s.x>=170&&s.x<=330&&s.y>=280);
  const isThree = s => {
    const d = Math.sqrt((s.x-BX)**2+(s.y-BY)**2);
    return d>235 || s.y < ARC_Y;
  };
  const three  = shots.filter(s=>isThree(s));
  const mid    = shots.filter(s=>!paint.includes(s)&&!three.includes(s));

  return (
    <div style={{ display:"grid", gridTemplateColumns:"minmax(0,1fr) 300px", gap:"1.25rem" }}>
      {/* Court panel */}
      <div style={{ background:C.card, borderRadius:18, border:`1px solid ${C.border}`, padding:"1.25rem" }}>
        <div style={{ display:"flex", gap:10, marginBottom:"1rem", flexWrap:"wrap", alignItems:"center" }}>
          <select value={pid} onChange={e=>setPid(e.target.value)}
            style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8,
              color:C.text, padding:"0.42rem 0.75rem", fontFamily:"Barlow,sans-serif",
              fontSize:"0.82rem", cursor:"pointer" }}>
            <option value="all">All Players</option>
            {PLAYERS.map(p=><option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
          <PillToggle
            options={[["all","All","orange"],["made","Made","green"],["missed","Missed","red"]]}
            value={sht} onChange={setSht} />
        </div>
        <CourtSVG shots={shots} filterShot={sht} />
        <div style={{ display:"flex", gap:18, marginTop:"0.75rem",
          paddingTop:"0.75rem", borderTop:`1px solid ${C.border}` }}>
          <LegDot color={C.green} label="Made" />
          <LegDot color={C.red}   label="Missed" />
        </div>
      </div>

      {/* Stats sidebar */}
      <div style={{ display:"flex", flexDirection:"column", gap:"1rem" }}>
        {/* Shooting summary */}
        <div style={{ background:C.card, borderRadius:18, border:`1px solid ${C.border}`, padding:"1.25rem" }}>
          <SectionHead>Shot Summary</SectionHead>
          <div style={{ display:"flex", flexDirection:"column", gap:0 }}>
            <SRow label="Attempts" val={shots.length} />
            <SRow label="Made" val={made} color={C.green} />
            <SRow label="Missed" val={missed} color={C.red} />
          </div>
          <div style={{ marginTop:"1rem" }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
              <span style={{ fontSize:"0.72rem", color:C.muted, textTransform:"uppercase",
                letterSpacing:"0.07em", fontFamily:"Barlow Condensed,sans-serif" }}>FG%</span>
              <span style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.4rem",
                color: fgPct>=50 ? C.green : fgPct>=40 ? C.yellow : C.red }}>
                {fgPct}%
              </span>
            </div>
            <div style={{ height:6, borderRadius:3, background:"rgba(255,255,255,0.07)" }}>
              <div style={{ height:"100%", borderRadius:3, width:`${fgPct}%`,
                background: fgPct>=50 ? C.green : fgPct>=40 ? C.yellow : C.red,
                transition:"width 0.4s" }} />
            </div>
          </div>
        </div>

        {/* Zone breakdown */}
        <div style={{ background:C.card, borderRadius:18, border:`1px solid ${C.border}`, padding:"1.25rem" }}>
          <SectionHead>By Zone</SectionHead>
          <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
            {[["Paint",paint,C.blue],["Mid-Range",mid,C.yellow],["3-Point",three,C.orange]].map(([lbl,arr,col]) => {
              const m = arr.filter(s=>s.made).length;
              const a = arr.length;
              const p = a>0 ? Math.round(m/a*100) : 0;
              return (
                <div key={lbl}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                    <span style={{ fontSize:"0.8rem", color:C.muted }}>{lbl}</span>
                    <span style={{ fontSize:"0.8rem", fontWeight:600, color:col }}>{p}% ({m}/{a})</span>
                  </div>
                  <div style={{ height:4, borderRadius:2, background:"rgba(255,255,255,0.07)" }}>
                    <div style={{ height:"100%", borderRadius:2, width:`${p}%`, background:col, transition:"width 0.4s" }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function HotZonesTab() {
  return (
    <div style={{ display:"grid", gridTemplateColumns:"minmax(0,1fr) 280px", gap:"1.25rem" }}>
      <div style={{ background:C.card, borderRadius:18, border:`1px solid ${C.border}`, padding:"1.25rem" }}>
        <div style={{ marginBottom:"1rem" }}>
          <span style={{ fontSize:"0.75rem", color:C.muted, textTransform:"uppercase",
            letterSpacing:"0.08em", fontFamily:"Barlow Condensed,sans-serif" }}>
            Field goal percentage by court zone
          </span>
        </div>
        <CourtSVG shots={[]} showZones={true} />
        <div style={{ display:"flex", gap:16, marginTop:"1rem",
          paddingTop:"0.75rem", borderTop:`1px solid ${C.border}`, flexWrap:"wrap" }}>
          {[["rgba(59,130,246,0.5)","Cold (<35%)"],[`rgba(251,191,36,0.4)`,"Avg (35–50%)"],["rgba(239,68,68,0.5)","Hot (>50%)"]].map(([bg,lbl])=>(
            <div key={lbl} style={{ display:"flex", alignItems:"center", gap:6,
              fontSize:"0.78rem", color:C.muted }}>
              <div style={{ width:12, height:12, borderRadius:2, background:bg }} />
              {lbl}
            </div>
          ))}
        </div>
      </div>

      <div style={{ background:C.card, borderRadius:18, border:`1px solid ${C.border}`, padding:"1.25rem" }}>
        <SectionHead>Zone Breakdown</SectionHead>
        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
          {[...HOT_ZONES].sort((a,b)=>(b.made/b.att)-(a.made/a.att)).map((z,i)=>{
            const p = Math.round(z.made/z.att*100);
            const hot=p>=50, cold=p<35;
            return (
              <div key={i} style={{ display:"flex", justifyContent:"space-between",
                alignItems:"center", padding:"0.55rem 0.85rem", borderRadius:10,
                background:C.surface, border:`1px solid ${C.border}` }}>
                <div>
                  <div style={{ fontSize:"0.82rem", fontWeight:600, color:C.text }}>{z.lbl}</div>
                  <div style={{ fontSize:"0.72rem", color:C.muted }}>{z.att} att.</div>
                </div>
                <div style={{ textAlign:"right" }}>
                  <div style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.15rem",
                    color: hot?C.green:cold?C.red:C.yellow }}>{p}%</div>
                  <div style={{ fontSize:"0.72rem", color:C.muted }}>{z.made}/{z.att}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function PlayerStatsTab() {
  const [sort, setSort] = useState("touches");
  const sorted = useMemo(() => [...PLAYERS].sort((a,b) => {
    if (sort==="touches") return b.touches - a.touches;
    if (sort==="fg")      return (b.made/b.att) - (a.made/a.att);
    if (sort==="time")    return b.posS - a.posS;
    return 0;
  }), [sort]);

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:"1rem" }}>
      <div style={{ display:"flex", gap:8, alignItems:"center" }}>
        <span style={{ fontSize:"0.78rem", color:C.muted, fontFamily:"Barlow Condensed,sans-serif",
          textTransform:"uppercase", letterSpacing:"0.06em" }}>Sort:</span>
        {[["touches","Touches"],["fg","FG%"],["time","Possession"]].map(([v,l])=>(
          <button key={v} onClick={()=>setSort(v)}
            style={{ padding:"0.33rem 0.85rem",
              border:`1px solid ${sort===v ? C.orange : C.border}`,
              borderRadius:8, background: sort===v ? C.orangeFaint : "transparent",
              color: sort===v ? C.orange : C.muted, cursor:"pointer",
              fontFamily:"Barlow,sans-serif", fontSize:"0.8rem" }}>
            {l}
          </button>
        ))}
      </div>

      {sorted.map((p, rank) => {
        const fg  = Math.round(p.made/p.att*100);
        const avg = (p.posS/p.touches).toFixed(1);
        const col = fg>=55?C.green:fg>=45?C.yellow:C.red;
        const tcol = p.team===1?C.blue:C.orange;
        return (
          <div key={p.id} style={{ background:C.card, borderRadius:16,
            border:`1px solid ${rank===0 ? "rgba(240,122,40,0.22)" : C.border}`,
            padding:"1.1rem 1.5rem" }}>
            <div style={{ display:"grid", gridTemplateColumns:"28px 1fr repeat(5,auto)",
              gap:"0.75rem 1.5rem", alignItems:"center" }}>
              {/* Rank */}
              <div style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.4rem",
                color: rank===0?C.orange:C.muted, lineHeight:1 }}>
                {rank+1}
              </div>
              {/* Name */}
              <div>
                <div style={{ fontWeight:600, fontSize:"0.95rem" }}>{p.label}</div>
                <div style={{ fontSize:"0.75rem", color:tcol, marginTop:2,
                  fontFamily:"Barlow Condensed,sans-serif" }}>
                  Team {p.team}
                </div>
              </div>
              {/* Stats */}
              <PStat label="Touches"  val={p.touches}       />
              <PStat label="Poss."    val={`${p.posS}s`}    />
              <PStat label="Avg"      val={`${avg}s`}       />
              <PStat label="FGA/FGM"  val={`${p.made}/${p.att}`} />
              <PStat label="FG%"      val={`${fg}%`} color={col} />
            </div>

            {/* Mini FG bar */}
            <div style={{ marginTop:"0.85rem", height:3, borderRadius:2,
              background:"rgba(255,255,255,0.07)" }}>
              <div style={{ height:"100%", borderRadius:2, width:`${fg}%`,
                background:col, transition:"width 0.4s" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── DASHBOARD SHELL ─────────────────────────────────────────────────────────

function Dashboard({ session, onBack }) {
  const [tab, setTab] = useState("shot-chart");
  const allMade = ALL_SHOTS.filter(s=>s.made).length;
  const fgAll   = Math.round(allMade/ALL_SHOTS.length*100);
  const totalT  = PLAYERS.reduce((s,p)=>s+p.touches,0);

  const TABS = [
    ["live",       "Live Feed"   ],
    ["shot-chart", "Shot Chart"  ],
    ["hot-zones",  "Hot Zones"   ],
    ["players",    "Player Stats"],
  ];

  return (
    <div style={{ minHeight:"100vh", background:C.bg }}>
      {/* Header */}
      <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`,
        position:"sticky", top:0, zIndex:10 }}>
        <div style={{ maxWidth:1200, margin:"0 auto", padding:"0 1.5rem",
          display:"flex", alignItems:"center", justifyContent:"space-between", height:60 }}>
          <div style={{ display:"flex", alignItems:"center", gap:14 }}>
            <button onClick={onBack}
              style={{ background:"none", border:"none", color:C.muted,
                cursor:"pointer", fontSize:"0.82rem", fontFamily:"Barlow,sans-serif",
                padding:0 }}>
              ← Back
            </button>
            <div style={{ width:1, height:20, background:C.border }} />
            <span style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.25rem",
              letterSpacing:"0.05em", color:C.text }}>
              COURT<span style={{ color:C.orange }}>VISION</span>
            </span>
            <Tag label={session.mode} color={session.mode==="workout"?"orange":"blue"} />
          </div>
          <span style={{ fontWeight:600, fontSize:"0.9rem", color:C.text }}>
            {session.name}
          </span>
        </div>
      </div>

      <div style={{ maxWidth:1200, margin:"0 auto", padding:"1.5rem" }}>
        {/* Metric row */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:"1rem", marginBottom:"1.5rem" }}>
          <MetCard label="Shot Attempts"   value={ALL_SHOTS.length}   accent={C.orange}  sub="This session" />
          <MetCard label="Field Goal %"    value={`${fgAll}%`}        accent={fgAll>=50?C.green:C.yellow} />
          <MetCard label="Total Touches"   value={totalT}             accent={C.blue}    sub="Confirmed possessions" />
          <MetCard label="Players Tracked" value={PLAYERS.length}     accent={C.muted}   />
        </div>

        {/* Tab bar */}
        <div style={{ display:"flex", gap:4, background:C.surface, padding:4,
          borderRadius:12, border:`1px solid ${C.border}`, marginBottom:"1.5rem",
          width:"fit-content" }}>
          {TABS.map(([id,lbl])=>(
            <button key={id} onClick={()=>setTab(id)}
              style={{ padding:"0.45rem 1.15rem", borderRadius:8, border:"none",
                background: tab===id ? C.card : "transparent",
                color: tab===id ? C.text : C.muted, cursor:"pointer",
                fontFamily:"Barlow,sans-serif", fontSize:"0.85rem",
                fontWeight: tab===id ? 600 : 400, transition:"all 0.12s",
                whiteSpace:"nowrap",
                boxShadow: tab===id ? `0 0 0 1px ${C.border}` : "none" }}>
              {lbl}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "live"       && <LiveFeedTab />}
        {tab === "shot-chart" && <ShotChartTab />}
        {tab === "hot-zones"  && <HotZonesTab />}
        {tab === "players"    && <PlayerStatsTab />}
      </div>
    </div>
  );
}

// ─── MICRO COMPONENTS ────────────────────────────────────────────────────────

function SectionHead({ children }) {
  return (
    <div style={{ fontFamily:"Bebas Neue,Impact,sans-serif", letterSpacing:"0.06em",
      fontSize:"0.9rem", color:C.muted, marginBottom:"0.85rem", textTransform:"uppercase" }}>
      {children}
    </div>
  );
}
function LegDot({ color, label }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:6,
      fontSize:"0.8rem", color:C.muted }}>
      <svg width={12} height={12}><circle cx={6} cy={6} r={5} fill={color} /></svg>
      {label}
    </div>
  );
}
function SRow({ label, val, color }) {
  return (
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
      padding:"0.4rem 0", borderBottom:`1px solid ${C.border}` }}>
      <span style={{ fontSize:"0.82rem", color:C.muted }}>{label}</span>
      <span style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.1rem",
        color:color||C.text }}>{val}</span>
    </div>
  );
}
function PStat({ label, val, color }) {
  return (
    <div style={{ textAlign:"right" }}>
      <div style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.05rem",
        color:color||C.text, lineHeight:1 }}>{val}</div>
      <div style={{ fontSize:"0.68rem", color:C.muted, textTransform:"uppercase",
        letterSpacing:"0.06em", fontFamily:"Barlow Condensed,sans-serif", marginTop:2 }}>
        {label}
      </div>
    </div>
  );
}

// ─── ROOT APP ─────────────────────────────────────────────────────────────────

function App() {
  const initialMode = BOOTSTRAP.form?.mode || "game";
  const hasBootstrapError = Boolean(BOOTSTRAP.error);
  const [screen, setScreen] = useState(hasBootstrapError ? "setup" : "welcome");
  const [mode, setMode] = useState(hasBootstrapError ? initialMode : null);
  const [form, setForm] = useState(BOOTSTRAP.form || {});
  const [error, setError] = useState(BOOTSTRAP.error || "");

  useEffect(() => {
    const el = document.createElement("style");
    const previousBodyStyle = document.body.getAttribute("style") || "";
    el.textContent = FONTS;
    document.head.appendChild(el);
    document.body.style.cssText = `background:${C.bg};margin:0;padding:0;`;
    return () => {
      el.remove();
      document.body.setAttribute("style", previousBodyStyle);
    };
  }, []);

  if (screen === "welcome") return (
    <WelcomeScreen onPick={m => {
      setMode(m);
      setError("");
      setForm(current => ({
        ...current,
        mode: m,
      }));
      setScreen("setup");
    }} />
  );
  return (
    <SetupScreen mode={mode}
      onBack={() => {
        setError("");
        setScreen("welcome");
      }}
      initialValues={form}
      workoutAccounts={BOOTSTRAP.workout_accounts || []}
      analysisProfiles={BOOTSTRAP.analysis_profiles || []}
      errorMessage={error}
      onStart={params => {
        setForm(params);
        setError("");
        const search = new URLSearchParams();
        Object.entries(params).forEach(([key, value]) => {
          if (value) {
            search.set(key, value);
          }
        });
        window.location.assign(`/analyze?${search.toString()}`);
      }} />
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
