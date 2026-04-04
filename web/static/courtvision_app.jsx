const { useState, useEffect } = React;

const C = {
  bg: "#08090e",
  card: "#151a27",
  cardHv: "#1a2030",
  surface: "#111520",
  border: "rgba(255,255,255,0.06)",
  orange: "#f07a28",
  orangeFaint: "rgba(240,122,40,0.12)",
  blue: "#38bdf8",
  blueFaint: "rgba(56,189,248,0.12)",
  text: "#e6e1d6",
  muted: "#8792a2",
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

function WelcomeScreen({ onPick }) {
  const [hov, setHov] = useState(null);

  return (
    <div style={{ minHeight:"100vh", background:C.bg, display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center", padding:"2rem" }}>
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

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1.25rem",
        width:"100%", maxWidth:540 }}>
        {[
          {
            id:"workout",
            title:"Workout",
            desc:"Track a single player or small group. Enter player IDs manually before the session.",
            accent:C.orange,
            icon:<svg width={32} height={32} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}><circle cx={12} cy={7} r={4}/><path d="M6 21v-2a6 6 0 0 1 12 0v2"/></svg>,
          },
          {
            id:"game",
            title:"Game",
            desc:"Full game tracking. Computer vision auto-detects all players and both teams.",
            accent:C.blue,
            icon:<svg width={32} height={32} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}><circle cx={12} cy={12} r={10}/><path d="M12 2Q18 7 18 12Q18 17 12 22M12 2Q6 7 6 12Q6 17 12 22M2 12H22"/></svg>,
          },
        ].map((modeOption) => (
          <button key={modeOption.id} onClick={() => onPick(modeOption.id)}
            onMouseEnter={() => setHov(modeOption.id)}
            onMouseLeave={() => setHov(null)}
            style={{ background: hov===modeOption.id ? C.cardHv : C.card,
              border:`1px solid ${hov===modeOption.id ? modeOption.accent : C.border}`,
              borderRadius:18, padding:"2rem 1.5rem", cursor:"pointer", textAlign:"left",
              transition:"all 0.18s", transform: hov===modeOption.id ? "translateY(-3px)" : "none" }}>
            <div style={{ color:modeOption.accent, marginBottom:"1rem" }}>{modeOption.icon}</div>
            <div style={{ fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.65rem",
              letterSpacing:"0.04em", color:C.text, marginBottom:"0.5rem" }}>
              {modeOption.title}
            </div>
            <div style={{ color:C.muted, fontSize:"0.83rem", lineHeight:1.55 }}>{modeOption.desc}</div>
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
  const accent = mode === "workout" ? C.orange : C.blue;

  const inputStyle = {
    background:C.surface,
    border:`1px solid ${C.border}`,
    borderRadius:10,
    padding:"0.7rem 1rem",
    color:C.text,
    fontSize:"0.9rem",
    fontFamily:"Barlow,sans-serif",
    width:"100%",
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
          <input value={name} onChange={event => setName(event.target.value)}
            placeholder={mode==="workout" ? "Morning Shooting Drill" : "Varsity Practice — Fri"}
            style={inputStyle} />
        </div>

        {mode === "workout" ? (
          <>
            <div style={{ marginBottom:"1.25rem" }}>
              <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
                letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
                Workout Account
              </label>
              <select value={accountId} onChange={event => setAccountId(event.target.value)} style={inputStyle}>
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
              <input value={playerId} onChange={event => setPlayerId(event.target.value)}
                placeholder="12"
                style={inputStyle} />
            </div>
          </>
        ) : (
          <div style={{ marginBottom:"1.25rem" }}>
            <label style={{ display:"block", fontSize:"0.7rem", color:C.muted,
              letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"Barlow Condensed,sans-serif", marginBottom:"0.5rem" }}>
              Teams
            </label>
            <div style={{ display:"flex", gap:8 }}>
              <input value={t1} onChange={event => setT1(event.target.value)} placeholder="Home" style={inputStyle} />
              <input value={t2} onChange={event => setT2(event.target.value)} placeholder="Away" style={inputStyle} />
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
            onChange={event => setSource(event.target.value)}
            placeholder="Input_vids/video_1.mp4 or https://..."
            style={inputStyle}
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
          style={{ width:"100%", padding:"0.95rem", background:accent, border:"none",
            borderRadius:12, color: mode==="workout" ? "#1a0800" : "#00111f",
            fontFamily:"Bebas Neue,Impact,sans-serif", fontSize:"1.15rem",
            letterSpacing:"0.06em", cursor:"pointer", fontWeight:700 }}>
          START SESSION →
        </button>
      </form>
    </div>
  );
}

function App() {
  const initialMode = BOOTSTRAP.form?.mode || "game";
  const hasBootstrapError = Boolean(BOOTSTRAP.error);
  const [screen, setScreen] = useState(hasBootstrapError ? "setup" : "welcome");
  const [mode, setMode] = useState(hasBootstrapError ? initialMode : null);
  const [form, setForm] = useState(BOOTSTRAP.form || {});
  const [error, setError] = useState(BOOTSTRAP.error || "");

  useEffect(() => {
    const styleElement = document.createElement("style");
    const previousBodyStyle = document.body.getAttribute("style") || "";
    styleElement.textContent = FONTS;
    document.head.appendChild(styleElement);
    document.body.style.cssText = `background:${C.bg};margin:0;padding:0;`;

    return () => {
      styleElement.remove();
      document.body.setAttribute("style", previousBodyStyle);
    };
  }, []);

  if (screen === "welcome") {
    return (
      <WelcomeScreen onPick={(selectedMode) => {
        setMode(selectedMode);
        setError("");
        setForm(current => ({
          ...current,
          mode: selectedMode,
        }));
        setScreen("setup");
      }} />
    );
  }

  return (
    <SetupScreen
      mode={mode}
      onBack={() => {
        setError("");
        setScreen("welcome");
      }}
      initialValues={form}
      workoutAccounts={BOOTSTRAP.workout_accounts || []}
      analysisProfiles={BOOTSTRAP.analysis_profiles || []}
      errorMessage={error}
      onStart={(params) => {
        setForm(params);
        setError("");
        const search = new URLSearchParams();
        Object.entries(params).forEach(([key, value]) => {
          if (value) {
            search.set(key, value);
          }
        });
        window.location.assign(`/analyze?${search.toString()}`);
      }}
    />
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
