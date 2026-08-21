use chrono::{DateTime, Duration, FixedOffset, NaiveDate, TimeZone, Utc};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use regex::Regex;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

const HKT_OFFSET: i32 = 8 * 3600;

fn hkt() -> FixedOffset {
    FixedOffset::east_opt(HKT_OFFSET).unwrap()
}

// --- Data structures ---

#[derive(Deserialize)]
struct HistoryEntry {
    timestamp: Option<serde_json::Value>,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
    prompt: Option<String>,
    display: Option<String>,
}

#[derive(Deserialize)]
struct TranscriptEntry {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    timestamp: Option<String>,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
    message: Option<TranscriptMessage>,
}

#[derive(Deserialize)]
struct TranscriptMessage {
    content: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OpenCodeSession {
    id: Option<String>,
    time: Option<OpenCodeTime>,
}

#[derive(Deserialize)]
struct OpenCodeTime {
    created: Option<i64>,
    updated: Option<i64>,
}

#[derive(Deserialize)]
struct OpenCodeMessage {
    role: Option<String>,
    time: Option<OpenCodeTime>,
    id: Option<String>,
}

#[derive(Deserialize)]
struct OpenCodePart {
    text: Option<String>,
}

#[derive(Clone)]
struct Prompt {
    time_str: String,
    timestamp_ms: i64,
    session: String,
    session_full: String,
    prompt: String,
    tool: String,
}

#[derive(Clone)]
struct SearchMatch {
    date: String,
    time_str: String,
    timestamp_ms: i64,
    session: String,
    role: String,
    snippet: String,
    tool: String,
}

struct SessionInfo {
    count: usize,
    first: DateTime<FixedOffset>,
    last: DateTime<FixedOffset>,
    tool: String,
    id_short: String,
}

// --- CLI ---

#[derive(Parser)]
#[command(name = "anam", about = "Search AI coding chat history", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Date to scan (YYYY-MM-DD, "today", "yesterday")
    #[arg(default_value = "today")]
    date: String,

    /// Show all prompts (not just last 50)
    #[arg(long)]
    full: bool,

    /// Output as JSON
    #[arg(long)]
    json: bool,

    /// Filter by tool (Claude, Codex, OpenCode)
    #[arg(long)]
    tool: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Search prompts or transcripts
    Search {
        /// Search pattern (regex)
        pattern: String,

        /// Number of days to search
        #[arg(long, default_value = "7")]
        days: u32,

        /// Search user prompts only (default: search full transcripts)
        #[arg(long)]
        prompts_only: bool,

        /// Filter by tool
        #[arg(long)]
        tool: Option<String>,

        /// Filter by role. Each name is tool-specific: you (aliases: user, me),
        /// claude (aliases: assistant, ai), opencode, codex. assistant/ai match
        /// Claude turns only, not OpenCode or Codex.
        #[arg(long)]
        role: Option<String>,

        /// Filter by session ID (prefix match)
        #[arg(long)]
        session: Option<String>,

        /// Emit full untruncated turn content instead of ~100-char snippets
        #[arg(long)]
        full: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Dump a session's turns in order, untruncated
    Dump {
        /// Session ID prefix (e.g. first 8 chars)
        session: String,

        /// Include tool_use/tool_result blocks (skipped by default)
        #[arg(long)]
        include_tools: bool,

        /// Filter by tool (Claude, Codex, OpenCode)
        #[arg(long)]
        tool: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

// --- Path helpers ---

fn home_dir() -> PathBuf {
    dirs::home_dir().expect("Cannot determine home directory")
}

fn history_files() -> Vec<(String, PathBuf)> {
    let home = home_dir();
    // Codex moved from ~/.codex/history.jsonl to rollout transcripts (see scan_codex).
    vec![("Claude".into(), home.join(".claude/history.jsonl"))]
}

fn codex_session_roots() -> Vec<PathBuf> {
    let home = home_dir();
    vec![
        home.join(".codex/sessions"),
        home.join(".codex/archived_sessions"),
    ]
}

fn projects_dir() -> PathBuf {
    home_dir().join(".claude/projects")
}

fn opencode_storage() -> PathBuf {
    home_dir().join(".local/share/opencode/storage")
}

// --- Time helpers ---

fn date_to_range_ms(date_str: &str) -> (i64, i64) {
    let hkt_tz = hkt();
    let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .unwrap_or_else(|_| panic!("Invalid date: {}", date_str));
    let start = hkt_tz
        .from_local_datetime(&date.and_hms_opt(0, 0, 0).unwrap())
        .single()
        .unwrap();
    let end = start + Duration::days(1);
    (start.timestamp() * 1000, end.timestamp() * 1000)
}

fn resolve_date(input: &str) -> String {
    let now = Utc::now().with_timezone(&hkt());
    match input {
        "today" => now.format("%Y-%m-%d").to_string(),
        "yesterday" => (now - Duration::days(1)).format("%Y-%m-%d").to_string(),
        other => {
            NaiveDate::parse_from_str(other, "%Y-%m-%d").unwrap_or_else(|_| {
                eprintln!("Invalid date format: {}. Use YYYY-MM-DD.", other);
                std::process::exit(1);
            });
            other.to_string()
        }
    }
}

/// Convert epoch milliseconds to HKT.
///
/// Valid timestamps (including pre-epoch negatives) convert directly.
/// Values chrono cannot represent are clamped to the Unix epoch
/// (1970-01-01 08:00:00 HKT) rather than `Utc::now()`, so a corrupt
/// record cannot date itself today.
fn ms_to_hkt(ms: i64) -> DateTime<FixedOffset> {
    DateTime::from_timestamp_millis(ms)
        .unwrap_or(DateTime::UNIX_EPOCH)
        .with_timezone(&hkt())
}

// --- Content extraction ---

fn extract_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            let mut parts = Vec::new();
            for block in arr {
                if let Some(obj) = block.as_object() {
                    match obj.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                                parts.push(text.to_string());
                            }
                        }
                        Some("tool_use") => {
                            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                                parts.push(format!("[tool: {}]", name));
                            }
                        }
                        _ => {}
                    }
                }
            }
            parts.join(" ")
        }
        _ => String::new(),
    }
}

// --- Scan history (date mode) ---

fn scan_history(date_str: &str, tool_filter: Option<&str>) -> Vec<Prompt> {
    let (start_ms, end_ms) = date_to_range_ms(date_str);
    let mut prompts = Vec::new();

    let files = history_files();
    for (label, path) in &files {
        if let Some(filter) = tool_filter {
            if !label.eq_ignore_ascii_case(filter) {
                continue;
            }
        }
        if !path.exists() {
            continue;
        }
        if let Ok(file) = fs::File::open(path) {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let entry: HistoryEntry = match serde_json::from_str(&line) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let ts = match &entry.timestamp {
                    Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(0),
                    _ => continue,
                };
                if ts < start_ms || ts >= end_ms {
                    continue;
                }
                let prompt_text = entry.display.or(entry.prompt).unwrap_or_default();
                let session = entry.session_id.unwrap_or_else(|| "unknown".into());
                let dt = ms_to_hkt(ts);

                prompts.push(Prompt {
                    time_str: dt.format("%H:%M").to_string(),
                    timestamp_ms: ts,
                    session: session[..session.len().min(8)].to_string(),
                    session_full: session,
                    prompt: prompt_text,
                    tool: label.clone(),
                });
            }
        }
    }

    // OpenCode
    if tool_filter.is_none()
        || tool_filter
            .map(|t| t.eq_ignore_ascii_case("opencode"))
            .unwrap_or(false)
    {
        prompts.extend(scan_opencode(start_ms, end_ms));
    }

    // Codex (rollout transcripts)
    if tool_filter.is_none()
        || tool_filter
            .map(|t| t.eq_ignore_ascii_case("codex"))
            .unwrap_or(false)
    {
        prompts.extend(scan_codex(start_ms, end_ms));
    }

    prompts.sort_by_key(|p| p.timestamp_ms);
    prompts
}

// --- OpenCode scanning ---

fn scan_opencode(start_ms: i64, end_ms: i64) -> Vec<Prompt> {
    let storage = opencode_storage();
    let session_dir = storage.join("session");
    if !session_dir.exists() {
        return Vec::new();
    }

    let mut prompts = Vec::new();

    let session_dirs: Vec<_> = match fs::read_dir(&session_dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(_) => return Vec::new(),
    };

    for sess_entry in session_dirs {
        let sess_path = sess_entry.path();
        if !sess_path.is_dir() {
            continue;
        }
        let json_files: Vec<_> = match fs::read_dir(&sess_path) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                .collect(),
            Err(_) => continue,
        };

        for jf in json_files {
            let content = match fs::read_to_string(jf.path()) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let sess: OpenCodeSession = match serde_json::from_str(&content) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let created = sess.time.as_ref().and_then(|t| t.created).unwrap_or(0);
            let updated = sess.time.as_ref().and_then(|t| t.updated).unwrap_or(0);
            if !((start_ms <= created && created < end_ms)
                || (start_ms <= updated && updated < end_ms))
            {
                continue;
            }

            let sess_id = match sess.id {
                Some(id) => id,
                None => continue,
            };

            let msg_dir = storage.join("message").join(&sess_id);
            if !msg_dir.exists() {
                continue;
            }

            let msg_files: Vec<_> = match fs::read_dir(&msg_dir) {
                Ok(rd) => rd
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.file_name()
                            .to_str()
                            .map(|n| n.starts_with("msg_") && n.ends_with(".json"))
                            .unwrap_or(false)
                    })
                    .collect(),
                Err(_) => continue,
            };

            for mf in msg_files {
                let mc = match fs::read_to_string(mf.path()) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let msg: OpenCodeMessage = match serde_json::from_str(&mc) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                if msg.role.as_deref() != Some("user") {
                    continue;
                }

                let ts_ms = msg.time.as_ref().and_then(|t| t.created).unwrap_or(0);
                if ts_ms < start_ms || ts_ms >= end_ms {
                    continue;
                }

                let msg_id = match msg.id {
                    Some(id) => id,
                    None => continue,
                };

                let part_dir = storage.join("part").join(&msg_id);
                let mut prompt_text = String::new();
                if part_dir.exists() {
                    if let Ok(rd) = fs::read_dir(&part_dir) {
                        let mut parts: Vec<_> = rd.filter_map(|e| e.ok()).collect();
                        parts.sort_by_key(|e| e.file_name());
                        for pf in parts {
                            if let Ok(pc) = fs::read_to_string(pf.path()) {
                                if let Ok(part) = serde_json::from_str::<OpenCodePart>(&pc) {
                                    if let Some(text) = part.text {
                                        prompt_text.push_str(&text);
                                    }
                                }
                            }
                        }
                    }
                }

                if !prompt_text.is_empty() {
                    let dt = ms_to_hkt(ts_ms);
                    prompts.push(Prompt {
                        time_str: dt.format("%H:%M").to_string(),
                        timestamp_ms: ts_ms,
                        session: sess_id[..sess_id.len().min(8)].to_string(),
                        session_full: sess_id.clone(),
                        prompt: prompt_text,
                        tool: "OpenCode".into(),
                    });
                }
            }
        }
    }

    prompts
}

// --- Codex scanning (rollout transcripts) ---

struct CodexMessage {
    timestamp_ms: i64,
    role: String,
    text: String,
}

fn collect_codex_rollouts(dir: &Path, start_epoch: i64, end_epoch: i64, out: &mut Vec<PathBuf>) {
    let rd = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in rd.filter_map(|e| e.ok()) {
        let path = entry.path();
        let meta = match path.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            collect_codex_rollouts(&path, start_epoch, end_epoch, out);
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.starts_with("rollout-") || !name.ends_with(".jsonl") {
            continue;
        }
        if let Ok(mtime) = meta.modified() {
            let epoch = mtime
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            if epoch >= start_epoch && epoch <= end_epoch {
                out.push(path);
            }
        }
    }
}

fn codex_rollout_files(start_ms: i64, end_ms: i64) -> Vec<PathBuf> {
    let start_epoch = (start_ms / 1000) - 86400;
    let end_epoch = (end_ms / 1000) + 86400;
    let mut files = Vec::new();
    for root in codex_session_roots() {
        if root.exists() {
            collect_codex_rollouts(&root, start_epoch, end_epoch, &mut files);
        }
    }
    files
}

fn codex_session_id(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    if stem.len() > 36 {
        let mut start = stem.len() - 36;
        while start < stem.len() && !stem.is_char_boundary(start) {
            start += 1;
        }
        stem[start..].to_string()
    } else {
        stem
    }
}

fn codex_messages(path: &Path) -> Vec<CodexMessage> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let mut messages = Vec::new();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let val: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if val.get("type").and_then(|t| t.as_str()) != Some("response_item") {
            continue;
        }
        let payload = match val.get("payload") {
            Some(p) => p,
            None => continue,
        };
        if payload.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        let role = match payload.get("role").and_then(|r| r.as_str()) {
            Some("user") => "user",
            Some("assistant") => "assistant",
            _ => continue,
        };
        let ts_str = match val.get("timestamp").and_then(|t| t.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let ts_dt = match DateTime::parse_from_rfc3339(&ts_str.replace('Z', "+00:00")) {
            Ok(dt) => dt,
            Err(_) => continue,
        };
        let timestamp_ms = ts_dt.timestamp() * 1000;
        let mut parts: Vec<String> = Vec::new();
        if let Some(content) = payload.get("content").and_then(|c| c.as_array()) {
            for block in content {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    if !text.is_empty() {
                        parts.push(text.to_string());
                    }
                }
            }
        }
        let text = parts.join(" ");
        if text.is_empty() {
            continue;
        }
        messages.push(CodexMessage {
            timestamp_ms,
            role: role.to_string(),
            text,
        });
    }
    messages
}

fn scan_codex(start_ms: i64, end_ms: i64) -> Vec<Prompt> {
    let files = codex_rollout_files(start_ms, end_ms);
    files
        .par_iter()
        .flat_map(|path| {
            let session_full = codex_session_id(path);
            let session = session_full[..session_full.len().min(8)].to_string();
            let mut out = Vec::new();
            for msg in codex_messages(path) {
                if msg.role != "user" {
                    continue;
                }
                if msg.timestamp_ms < start_ms || msg.timestamp_ms >= end_ms {
                    continue;
                }
                let dt = ms_to_hkt(msg.timestamp_ms);
                out.push(Prompt {
                    time_str: dt.format("%H:%M").to_string(),
                    timestamp_ms: msg.timestamp_ms,
                    session: session.clone(),
                    session_full: session_full.clone(),
                    prompt: msg.text,
                    tool: "Codex".into(),
                });
            }
            out
        })
        .collect()
}

// --- Search prompts (fast) ---

fn search_prompts(
    pattern: &str,
    start_ms: i64,
    end_ms: i64,
    tool_filter: Option<&str>,
    role_filter: Option<&str>,
    session_filter: Option<&str>,
    full: bool,
) -> Vec<SearchMatch> {
    // Prompts are always role "you" — if filtering for assistant roles, skip entirely
    if let Some(rf) = role_filter {
        if !matches_role("you", rf) {
            return Vec::new();
        }
    }

    let regex = Regex::new(&format!("(?i){}", pattern)).unwrap_or_else(|_| {
        eprintln!("Invalid regex pattern: {}", pattern);
        std::process::exit(1);
    });

    let mut matches = Vec::new();

    let files = history_files();
    for (label, path) in &files {
        if let Some(filter) = tool_filter {
            if !label.eq_ignore_ascii_case(filter) {
                continue;
            }
        }
        if !path.exists() {
            continue;
        }
        if let Ok(file) = fs::File::open(path) {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let entry: HistoryEntry = match serde_json::from_str(&line) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let ts = match &entry.timestamp {
                    Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(0),
                    _ => continue,
                };
                if ts < start_ms || ts >= end_ms {
                    continue;
                }
                let session = entry.session_id.unwrap_or_else(|| "unknown".into());
                if let Some(sf) = session_filter {
                    if !session.starts_with(sf) && !session[..session.len().min(8)].starts_with(sf)
                    {
                        continue;
                    }
                }
                let prompt_text = entry.display.or(entry.prompt).unwrap_or_default();
                if let Some(m) = regex.find(&prompt_text) {
                    let dt = ms_to_hkt(ts);
                    let snippet = if full {
                        prompt_text.clone()
                    } else {
                        make_snippet(&prompt_text, m.start(), m.end())
                    };

                    matches.push(SearchMatch {
                        date: dt.format("%Y-%m-%d").to_string(),
                        time_str: dt.format("%H:%M").to_string(),
                        timestamp_ms: ts,
                        session: session[..session.len().min(8)].to_string(),
                        role: "you".into(),
                        snippet,
                        tool: label.clone(),
                    });
                }
            }
        }
    }

    // OpenCode
    if tool_filter.is_none()
        || tool_filter
            .map(|t| t.eq_ignore_ascii_case("opencode"))
            .unwrap_or(false)
    {
        let oc_prompts = scan_opencode(start_ms, end_ms);
        for p in &oc_prompts {
            if let Some(sf) = session_filter {
                if !p.session_full.starts_with(sf) && !p.session.starts_with(sf) {
                    continue;
                }
            }
            if let Some(m) = regex.find(&p.prompt) {
                let snippet = if full {
                    p.prompt.clone()
                } else {
                    make_snippet(&p.prompt, m.start(), m.end())
                };
                matches.push(SearchMatch {
                    date: ms_to_hkt(p.timestamp_ms).format("%Y-%m-%d").to_string(),
                    time_str: p.time_str.clone(),
                    timestamp_ms: p.timestamp_ms,
                    session: p.session.clone(),
                    role: "you".into(),
                    snippet,
                    tool: "OpenCode".into(),
                });
            }
        }
    }

    // Codex (rollout transcripts)
    if tool_filter.is_none()
        || tool_filter
            .map(|t| t.eq_ignore_ascii_case("codex"))
            .unwrap_or(false)
    {
        let codex_prompts = scan_codex(start_ms, end_ms);
        for p in &codex_prompts {
            if let Some(sf) = session_filter {
                if !p.session_full.starts_with(sf) && !p.session.starts_with(sf) {
                    continue;
                }
            }
            if let Some(m) = regex.find(&p.prompt) {
                let snippet = if full {
                    p.prompt.clone()
                } else {
                    make_snippet(&p.prompt, m.start(), m.end())
                };
                matches.push(SearchMatch {
                    date: ms_to_hkt(p.timestamp_ms).format("%Y-%m-%d").to_string(),
                    time_str: p.time_str.clone(),
                    timestamp_ms: p.timestamp_ms,
                    session: p.session.clone(),
                    role: "you".into(),
                    snippet,
                    tool: "Codex".into(),
                });
            }
        }
    }

    matches.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
    matches
}

// --- Search transcripts (deep) ---

fn search_transcripts(
    pattern: &str,
    start_ms: i64,
    end_ms: i64,
    tool_filter: Option<&str>,
    role_filter: Option<&str>,
    session_filter: Option<&str>,
    full: bool,
) -> Vec<SearchMatch> {
    let regex = Regex::new(&format!("(?i){}", pattern)).unwrap_or_else(|_| {
        eprintln!("Invalid regex pattern: {}", pattern);
        std::process::exit(1);
    });

    let mut matches = Vec::new();

    // Claude transcripts
    if tool_filter.is_none()
        || tool_filter
            .map(|t| t.eq_ignore_ascii_case("claude"))
            .unwrap_or(false)
    {
        let proj_dir = projects_dir();
        if proj_dir.exists() {
            // Collect session files with mtime in range (with 1-day buffer)
            let start_epoch = (start_ms / 1000) - 86400;
            let end_epoch = (end_ms / 1000) + 86400;

            let mut session_files: Vec<PathBuf> = Vec::new();
            if let Ok(projects) = fs::read_dir(&proj_dir) {
                for proj in projects.filter_map(|e| e.ok()) {
                    if !proj.path().is_dir() {
                        continue;
                    }
                    if let Ok(files) = fs::read_dir(proj.path()) {
                        for f in files.filter_map(|e| e.ok()) {
                            let path = f.path();
                            if path.extension().map(|x| x == "jsonl").unwrap_or(false) {
                                if let Ok(meta) = path.metadata() {
                                    if let Ok(mtime) = meta.modified() {
                                        let epoch = mtime
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_secs()
                                            as i64;
                                        if epoch >= start_epoch && epoch <= end_epoch {
                                            session_files.push(path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Parallel scan with rayon
            let rf = role_filter;
            let sf = session_filter;
            let claude_matches: Vec<SearchMatch> = session_files
                .par_iter()
                .flat_map(|path| {
                    let mut file_matches = Vec::new();
                    let file = match fs::File::open(path) {
                        Ok(f) => f,
                        Err(_) => return file_matches,
                    };
                    let reader = BufReader::new(file);
                    for line in reader.lines() {
                        let line = match line {
                            Ok(l) => l,
                            Err(_) => continue,
                        };
                        let entry: TranscriptEntry = match serde_json::from_str(&line) {
                            Ok(e) => e,
                            Err(_) => continue,
                        };
                        let entry_type = match entry.entry_type.as_deref() {
                            Some("user") | Some("assistant") => {
                                entry.entry_type.as_deref().unwrap()
                            }
                            _ => continue,
                        };

                        let role = if entry_type == "user" {
                            "you"
                        } else {
                            "claude"
                        };

                        // Role filter
                        if let Some(rfilt) = rf {
                            if !matches_role(role, rfilt) {
                                continue;
                            }
                        }

                        let ts_str = match &entry.timestamp {
                            Some(s) => s.clone(),
                            None => continue,
                        };
                        let ts_dt =
                            match DateTime::parse_from_rfc3339(&ts_str.replace('Z', "+00:00")) {
                                Ok(dt) => dt,
                                Err(_) => match ts_str.parse::<DateTime<Utc>>() {
                                    Ok(dt) => dt.fixed_offset(),
                                    Err(_) => continue,
                                },
                            };
                        let ts_ms = ts_dt.timestamp() * 1000;
                        if ts_ms < start_ms || ts_ms >= end_ms {
                            continue;
                        }

                        // Session filter: check entry sessionId, fall back to file stem
                        if let Some(sfilt) = sf {
                            let sid = entry.session_id.as_deref().unwrap_or_else(|| {
                                // No sessionId in entry — use filename as session
                                ""
                            });
                            let effective_sid = if sid.is_empty() {
                                path.file_stem().unwrap_or_default().to_string_lossy()
                            } else {
                                std::borrow::Cow::Borrowed(sid)
                            };
                            if !effective_sid.starts_with(sfilt) {
                                continue;
                            }
                        }

                        let content = match &entry.message {
                            Some(msg) => match &msg.content {
                                Some(c) => c,
                                None => continue,
                            },
                            None => continue,
                        };

                        let text = extract_text(content);
                        if text.is_empty() {
                            continue;
                        }

                        if let Some(m) = regex.find(&text) {
                            let hkt_dt = ts_dt.with_timezone(&hkt());
                            let session = entry.session_id.unwrap_or_else(|| {
                                path.file_stem()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string()
                            });
                            let snippet = if full {
                                text.clone()
                            } else {
                                make_snippet(&text, m.start(), m.end())
                            };

                            file_matches.push(SearchMatch {
                                date: hkt_dt.format("%Y-%m-%d").to_string(),
                                time_str: hkt_dt.format("%H:%M").to_string(),
                                timestamp_ms: ts_ms,
                                session: session[..session.len().min(8)].to_string(),
                                role: role.into(),
                                snippet,
                                tool: "Claude".into(),
                            });
                        }
                    }
                    file_matches
                })
                .collect();

            matches.extend(claude_matches);
        }
    }

    // OpenCode transcripts
    if tool_filter.is_none()
        || tool_filter
            .map(|t| t.eq_ignore_ascii_case("opencode"))
            .unwrap_or(false)
    {
        let storage = opencode_storage();
        let session_dir = storage.join("session");
        if session_dir.exists() {
            let session_dirs: Vec<_> = match fs::read_dir(&session_dir) {
                Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
                Err(_) => Vec::new(),
            };

            for sess_entry in session_dirs {
                let sess_path = sess_entry.path();
                if !sess_path.is_dir() {
                    continue;
                }
                let json_files: Vec<_> = match fs::read_dir(&sess_path) {
                    Ok(rd) => rd
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                        .collect(),
                    Err(_) => continue,
                };

                for jf in json_files {
                    let content = match fs::read_to_string(jf.path()) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    let sess: OpenCodeSession = match serde_json::from_str(&content) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };

                    let created = sess.time.as_ref().and_then(|t| t.created).unwrap_or(0);
                    let updated = sess.time.as_ref().and_then(|t| t.updated).unwrap_or(0);
                    if !((start_ms <= created && created < end_ms)
                        || (start_ms <= updated && updated < end_ms))
                    {
                        continue;
                    }

                    let sess_id = match sess.id {
                        Some(id) => id,
                        None => continue,
                    };

                    let msg_dir = storage.join("message").join(&sess_id);
                    if !msg_dir.exists() {
                        continue;
                    }

                    let msg_files: Vec<_> = match fs::read_dir(&msg_dir) {
                        Ok(rd) => rd
                            .filter_map(|e| e.ok())
                            .filter(|e| {
                                e.file_name()
                                    .to_str()
                                    .map(|n| n.starts_with("msg_") && n.ends_with(".json"))
                                    .unwrap_or(false)
                            })
                            .collect(),
                        Err(_) => continue,
                    };

                    // Session filter for OpenCode
                    if let Some(sfilt) = session_filter {
                        if !sess_id.starts_with(sfilt)
                            && !sess_id[..sess_id.len().min(8)].starts_with(sfilt)
                        {
                            continue;
                        }
                    }

                    for mf in msg_files {
                        let mc = match fs::read_to_string(mf.path()) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };
                        let msg: OpenCodeMessage = match serde_json::from_str(&mc) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };

                        let role_str = msg.role.as_deref().unwrap_or("");
                        if role_str != "user" && role_str != "assistant" {
                            continue;
                        }

                        let role = if role_str == "user" {
                            "you"
                        } else {
                            "opencode"
                        };

                        // Role filter
                        if let Some(rfilt) = role_filter {
                            if !matches_role(role, rfilt) {
                                continue;
                            }
                        }

                        let ts_ms = msg.time.as_ref().and_then(|t| t.created).unwrap_or(0);
                        if ts_ms < start_ms || ts_ms >= end_ms {
                            continue;
                        }

                        let msg_id = match msg.id {
                            Some(id) => id,
                            None => continue,
                        };

                        let part_dir = storage.join("part").join(&msg_id);
                        let mut text = String::new();
                        if part_dir.exists() {
                            if let Ok(rd) = fs::read_dir(&part_dir) {
                                let mut parts: Vec<_> = rd.filter_map(|e| e.ok()).collect();
                                parts.sort_by_key(|e| e.file_name());
                                for pf in parts {
                                    if let Ok(pc) = fs::read_to_string(pf.path()) {
                                        if let Ok(part) = serde_json::from_str::<OpenCodePart>(&pc)
                                        {
                                            if let Some(t) = part.text {
                                                text.push_str(&t);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if !text.is_empty() {
                            if let Some(m) = regex.find(&text) {
                                let dt = ms_to_hkt(ts_ms);
                                let snippet = if full {
                                    text.clone()
                                } else {
                                    make_snippet(&text, m.start(), m.end())
                                };

                                matches.push(SearchMatch {
                                    date: dt.format("%Y-%m-%d").to_string(),
                                    time_str: dt.format("%H:%M").to_string(),
                                    timestamp_ms: ts_ms,
                                    session: sess_id[..sess_id.len().min(8)].to_string(),
                                    role: role.into(),
                                    snippet,
                                    tool: "OpenCode".into(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Codex transcripts (rollout files)
    if tool_filter.is_none()
        || tool_filter
            .map(|t| t.eq_ignore_ascii_case("codex"))
            .unwrap_or(false)
    {
        let codex_files = codex_rollout_files(start_ms, end_ms);
        let rf = role_filter;
        let sf = session_filter;
        let codex_matches: Vec<SearchMatch> = codex_files
            .par_iter()
            .flat_map(|path| {
                let mut file_matches = Vec::new();
                let session_full = codex_session_id(path);
                let session_short = session_full[..session_full.len().min(8)].to_string();
                if let Some(sfilt) = sf {
                    if !session_full.starts_with(sfilt) && !session_short.starts_with(sfilt) {
                        return file_matches;
                    }
                }
                for msg in codex_messages(path) {
                    let role = match msg.role.as_str() {
                        "user" => "you",
                        "assistant" => "codex",
                        _ => continue,
                    };
                    if let Some(rfilt) = rf {
                        if !matches_role(role, rfilt) {
                            continue;
                        }
                    }
                    let ts_ms = msg.timestamp_ms;
                    if ts_ms < start_ms || ts_ms >= end_ms {
                        continue;
                    }
                    if let Some(m) = regex.find(&msg.text) {
                        let dt = ms_to_hkt(ts_ms);
                        let snippet = if full {
                            msg.text.clone()
                        } else {
                            make_snippet(&msg.text, m.start(), m.end())
                        };
                        file_matches.push(SearchMatch {
                            date: dt.format("%Y-%m-%d").to_string(),
                            time_str: dt.format("%H:%M").to_string(),
                            timestamp_ms: ts_ms,
                            session: session_short.clone(),
                            role: role.into(),
                            snippet,
                            tool: "Codex".into(),
                        });
                    }
                }
                file_matches
            })
            .collect();
        matches.extend(codex_matches);
    }

    matches.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
    matches
}

// --- Helpers ---

/// Check if a stored role matches the `--role` filter.
/// Matching is case-insensitive. Each filter is tool-specific:
/// "you" / "user" / "me" → "you"
/// "claude" / "assistant" / "ai" → "claude" only (not OpenCode or Codex)
/// "opencode" → "opencode" only
/// "codex" → "codex" only
/// Unknown filters match the stored role by case-insensitive name.
fn matches_role(role: &str, filter: &str) -> bool {
    let r = role.to_ascii_lowercase();
    let f = filter.to_ascii_lowercase();
    match f.as_str() {
        "you" | "user" | "me" => r == "you",
        "claude" | "assistant" | "ai" => r == "claude",
        "opencode" => r == "opencode",
        "codex" => r == "codex",
        _ => r == f,
    }
}

fn make_snippet(text: &str, match_start: usize, match_end: usize) -> String {
    // Find char-safe boundaries
    let mut start = match_start.saturating_sub(40);
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (match_end + 60).min(text.len());
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }
    let mut snippet: String = text[start..end].replace('\n', " ");
    if start > 0 {
        snippet = format!("...{}", snippet);
    }
    if end < text.len() {
        snippet = format!("{}...", snippet);
    }
    snippet
}

// --- Session dump ---

struct Turn {
    timestamp_ms: i64,
    role: String,
    text: String,
}

/// Like extract_text, but keeps full content and optionally renders
/// tool_use/tool_result blocks. With include_tools=false, turns that are
/// pure tool noise extract to empty and are skipped by the caller.
fn extract_dump_text(content: &serde_json::Value, include_tools: bool) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            let mut parts = Vec::new();
            for block in arr {
                let obj = match block.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                match obj.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                            parts.push(text.to_string());
                        }
                    }
                    Some("tool_use") if include_tools => {
                        let name = obj.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                        let input = obj
                            .get("input")
                            .map(|i| serde_json::to_string(i).unwrap_or_default())
                            .unwrap_or_default();
                        parts.push(format!("[tool_use: {}] {}", name, input));
                    }
                    Some("tool_result") if include_tools => {
                        let inner = obj
                            .get("content")
                            .map(|c| extract_dump_text(c, include_tools))
                            .unwrap_or_default();
                        parts.push(format!("[tool_result] {}", inner));
                    }
                    _ => {}
                }
            }
            parts.join("\n")
        }
        _ => String::new(),
    }
}

fn dump_claude_sessions(prefix: &str, include_tools: bool) -> Vec<(String, String, Vec<Turn>)> {
    let proj_dir = projects_dir();
    let mut found: BTreeMap<String, Vec<Turn>> = BTreeMap::new();
    let projects = match fs::read_dir(&proj_dir) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    for proj in projects.filter_map(|e| e.ok()) {
        if !proj.path().is_dir() {
            continue;
        }
        let files = match fs::read_dir(proj.path()) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for f in files.filter_map(|e| e.ok()) {
            let path = f.path();
            if !path.extension().map(|x| x == "jsonl").unwrap_or(false) {
                continue;
            }
            let stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if !stem.starts_with(prefix) {
                continue;
            }
            let file = match fs::File::open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let turns = found.entry(stem).or_default();
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let entry: TranscriptEntry = match serde_json::from_str(&line) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let role = match entry.entry_type.as_deref() {
                    Some("user") => "you",
                    Some("assistant") => "claude",
                    _ => continue,
                };
                let ts_str = match &entry.timestamp {
                    Some(s) => s.clone(),
                    None => continue,
                };
                let ts_dt = match DateTime::parse_from_rfc3339(&ts_str.replace('Z', "+00:00")) {
                    Ok(dt) => dt,
                    Err(_) => match ts_str.parse::<DateTime<Utc>>() {
                        Ok(dt) => dt.fixed_offset(),
                        Err(_) => continue,
                    },
                };
                let content = match &entry.message {
                    Some(msg) => match &msg.content {
                        Some(c) => c,
                        None => continue,
                    },
                    None => continue,
                };
                let text = extract_dump_text(content, include_tools);
                if text.is_empty() {
                    continue;
                }
                turns.push(Turn {
                    timestamp_ms: ts_dt.timestamp() * 1000,
                    role: role.to_string(),
                    text,
                });
            }
        }
    }
    found
        .into_iter()
        .map(|(sid, turns)| ("Claude".to_string(), sid, turns))
        .collect()
}

fn dump_codex_sessions(prefix: &str) -> Vec<(String, String, Vec<Turn>)> {
    let files = codex_rollout_files(i64::MIN, i64::MAX);
    let mut found: BTreeMap<String, Vec<Turn>> = BTreeMap::new();
    for path in files {
        let session_full = codex_session_id(&path);
        if !session_full.starts_with(prefix) {
            continue;
        }
        let turns = found.entry(session_full).or_default();
        for msg in codex_messages(&path) {
            let role = match msg.role.as_str() {
                "user" => "you",
                "assistant" => "codex",
                _ => continue,
            };
            turns.push(Turn {
                timestamp_ms: msg.timestamp_ms,
                role: role.to_string(),
                text: msg.text,
            });
        }
    }
    found
        .into_iter()
        .map(|(sid, turns)| ("Codex".to_string(), sid, turns))
        .collect()
}

fn dump_opencode_sessions(prefix: &str) -> Vec<(String, String, Vec<Turn>)> {
    let storage = opencode_storage();
    let session_dir = storage.join("session");
    let mut found: BTreeMap<String, Vec<Turn>> = BTreeMap::new();
    let session_dirs = match fs::read_dir(&session_dir) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    for sess_entry in session_dirs.filter_map(|e| e.ok()) {
        let sess_path = sess_entry.path();
        if !sess_path.is_dir() {
            continue;
        }
        let json_files = match fs::read_dir(&sess_path) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for jf in json_files.filter_map(|e| e.ok()) {
            if !jf.path().extension().map(|x| x == "json").unwrap_or(false) {
                continue;
            }
            let content = match fs::read_to_string(jf.path()) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let sess: OpenCodeSession = match serde_json::from_str(&content) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let sess_id = match sess.id {
                Some(id) => id,
                None => continue,
            };
            if !sess_id.starts_with(prefix) {
                continue;
            }
            let msg_dir = storage.join("message").join(&sess_id);
            let msg_files = match fs::read_dir(&msg_dir) {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            let turns = found.entry(sess_id).or_default();
            for mf in msg_files.filter_map(|e| e.ok()) {
                let name_ok = mf
                    .file_name()
                    .to_str()
                    .map(|n| n.starts_with("msg_") && n.ends_with(".json"))
                    .unwrap_or(false);
                if !name_ok {
                    continue;
                }
                let mc = match fs::read_to_string(mf.path()) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let msg: OpenCodeMessage = match serde_json::from_str(&mc) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let role = match msg.role.as_deref() {
                    Some("user") => "you",
                    Some("assistant") => "opencode",
                    _ => continue,
                };
                let ts_ms = msg.time.as_ref().and_then(|t| t.created).unwrap_or(0);
                let msg_id = match msg.id {
                    Some(id) => id,
                    None => continue,
                };
                let part_dir = storage.join("part").join(&msg_id);
                let mut text = String::new();
                if let Ok(rd) = fs::read_dir(&part_dir) {
                    let mut parts: Vec<_> = rd.filter_map(|e| e.ok()).collect();
                    parts.sort_by_key(|e| e.file_name());
                    for pf in parts {
                        if let Ok(pc) = fs::read_to_string(pf.path()) {
                            if let Ok(part) = serde_json::from_str::<OpenCodePart>(&pc) {
                                if let Some(t) = part.text {
                                    text.push_str(&t);
                                }
                            }
                        }
                    }
                }
                if text.is_empty() {
                    continue;
                }
                turns.push(Turn {
                    timestamp_ms: ts_ms,
                    role: role.to_string(),
                    text,
                });
            }
        }
    }
    found
        .into_iter()
        .map(|(sid, turns)| ("OpenCode".to_string(), sid, turns))
        .collect()
}

fn run_dump(prefix: &str, include_tools: bool, tool_filter: Option<&str>, json: bool) {
    let want = |label: &str| {
        tool_filter
            .map(|t| t.eq_ignore_ascii_case(label))
            .unwrap_or(true)
    };
    let mut candidates: Vec<(String, String, Vec<Turn>)> = Vec::new();
    if want("claude") {
        candidates.extend(dump_claude_sessions(prefix, include_tools));
    }
    if want("codex") {
        candidates.extend(dump_codex_sessions(prefix));
    }
    if want("opencode") {
        candidates.extend(dump_opencode_sessions(prefix));
    }

    // Drop matches with no renderable turns (e.g. Claude "-edit-log" sidecar
    // files share the session-id stem but hold no transcript).
    candidates.retain(|(_, _, turns)| !turns.is_empty());

    if candidates.is_empty() {
        eprintln!("No session found matching prefix '{}'", prefix);
        std::process::exit(1);
    }
    if candidates.len() > 1 {
        eprintln!(
            "Prefix '{}' matches {} sessions; be more specific:",
            prefix,
            candidates.len()
        );
        for (tool, sid, turns) in &candidates {
            eprintln!("  {} ({}, {} turns)", sid, tool, turns.len());
        }
        std::process::exit(1);
    }

    let (tool, session_full, mut turns) = candidates.pop().unwrap();
    turns.sort_by_key(|t| t.timestamp_ms);

    if json {
        let output = serde_json::json!({
            "session": session_full,
            "tool": tool,
            "turns": turns.iter().map(|t| {
                let dt = ms_to_hkt(t.timestamp_ms);
                serde_json::json!({
                    "timestamp": t.timestamp_ms,
                    "date": dt.format("%Y-%m-%d").to_string(),
                    "time": dt.format("%H:%M").to_string(),
                    "role": t.role,
                    "content": t.text,
                })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("Session: {} ({})", session_full, tool);
        println!("Turns: {}", turns.len());
        println!();
        for t in &turns {
            let dt = ms_to_hkt(t.timestamp_ms);
            println!(
                "[{} {}] {}:",
                dt.format("%Y-%m-%d"),
                dt.format("%H:%M"),
                t.role
            );
            println!("{}", t.text);
            println!();
        }
    }
}

// --- Display ---

fn print_scan(prompts: &[Prompt], date_str: &str, full: bool) {
    let mut sessions: BTreeMap<String, SessionInfo> = BTreeMap::new();
    for p in prompts {
        let entry = sessions.entry(p.session_full.clone()).or_insert_with(|| {
            let dt = ms_to_hkt(p.timestamp_ms);
            SessionInfo {
                count: 0,
                first: dt,
                last: dt,
                tool: p.tool.clone(),
                id_short: p.session.clone(),
            }
        });
        entry.count += 1;
        let dt = ms_to_hkt(p.timestamp_ms);
        if dt < entry.first {
            entry.first = dt;
        }
        if dt > entry.last {
            entry.last = dt;
        }
    }

    let mut sorted_sessions: Vec<&SessionInfo> = sessions.values().collect();
    sorted_sessions.sort_by_key(|s| s.first);

    println!("Date: {} (HKT)", date_str);
    println!(
        "Total: {} prompts across {} sessions",
        prompts.len(),
        sessions.len()
    );
    println!();

    if let (Some(first), Some(last)) = (sorted_sessions.first(), sorted_sessions.last()) {
        println!(
            "Time range: {} - {}",
            first.first.format("%H:%M"),
            last.last.format("%H:%M")
        );
        println!();
    }

    println!("Sessions:");
    for s in &sorted_sessions {
        println!(
            "  [{}] {:3} prompts ({}-{}) - {}",
            s.id_short,
            s.count,
            s.first.format("%H:%M"),
            s.last.format("%H:%M"),
            s.tool
        );
    }
    println!();

    let display_prompts = if full {
        prompts.to_vec()
    } else {
        let start = if prompts.len() > 50 {
            prompts.len() - 50
        } else {
            0
        };
        prompts[start..].to_vec()
    };

    let label = if full {
        "All prompts:".to_string()
    } else {
        format!("Recent prompts (last {}):", display_prompts.len())
    };
    println!("{}", label);

    for p in &display_prompts {
        let preview: String = p
            .prompt
            .chars()
            .take(80)
            .collect::<String>()
            .replace('\n', " ");
        let ellipsis = if p.prompt.len() > 80 { "..." } else { "" };
        println!(
            "  {} [{}] ({}) {}{}",
            p.time_str, p.session, p.tool, preview, ellipsis
        );
    }
}

fn print_search(
    matches: &[SearchMatch],
    pattern: &str,
    days: u32,
    deep: bool,
    role_filter: Option<&str>,
    session_filter: Option<&str>,
    full: bool,
) {
    let mode = if deep {
        "full transcripts"
    } else {
        "prompts only"
    };
    let mut filters = String::new();
    if let Some(r) = role_filter {
        filters.push_str(&format!(", role={}", r));
    }
    if let Some(s) = session_filter {
        filters.push_str(&format!(", session={}", s));
    }
    println!(
        "Search: \"{}\" (last {} days, {}{})",
        pattern, days, mode, filters
    );

    if matches.is_empty() {
        println!("No matches found.");
        return;
    }

    let mut by_date: BTreeMap<String, Vec<&SearchMatch>> = BTreeMap::new();
    for m in matches {
        by_date.entry(m.date.clone()).or_default().push(m);
    }

    println!(
        "Found {} matches across {} days\n",
        matches.len(),
        by_date.len()
    );

    for (date, day_matches) in by_date.iter().rev() {
        println!("  {}:", date);
        let mut sorted = day_matches.clone();
        sorted.sort_by_key(|m| m.timestamp_ms);
        for m in sorted {
            let role_tag = if deep {
                format!("({})", m.role)
            } else {
                String::new()
            };
            if full {
                println!("    {} [{}] {:9}", m.time_str, m.session, role_tag);
                for line in m.snippet.lines() {
                    println!("      {}", line);
                }
                println!();
            } else {
                let snippet: String = m.snippet.chars().take(100).collect();
                println!(
                    "    {} [{}] {:9} {}",
                    m.time_str, m.session, role_tag, snippet
                );
            }
        }
        println!();
    }
}

fn print_json_scan(prompts: &[Prompt], date_str: &str) {
    let sessions: BTreeMap<String, usize> = {
        let mut map = BTreeMap::new();
        for p in prompts {
            *map.entry(p.session_full.clone()).or_insert(0) += 1;
        }
        map
    };

    let output = serde_json::json!({
        "date": date_str,
        "total": prompts.len(),
        "sessions": sessions.len(),
        "prompts": prompts.iter().map(|p| {
            serde_json::json!({
                "time": p.time_str,
                "timestamp": p.timestamp_ms,
                "session": p.session,
                "prompt": p.prompt,
                "tool": p.tool,
            })
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

fn print_json_search(matches: &[SearchMatch], full: bool) {
    let output: Vec<_> = matches
        .iter()
        .map(|m| {
            let mut obj = serde_json::json!({
                "date": m.date,
                "time": m.time_str,
                "timestamp": m.timestamp_ms,
                "session": m.session,
                "role": m.role,
                "tool": m.tool,
            });
            let key = if full { "content" } else { "snippet" };
            obj[key] = serde_json::Value::String(m.snippet.clone());
            obj
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

// --- Main ---

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Search {
            pattern,
            days,
            prompts_only,
            tool,
            role,
            session,
            full,
            json,
        }) => {
            let deep = !prompts_only;
            let now = Utc::now().with_timezone(&hkt());
            let end = (now + Duration::days(1))
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .unwrap();
            let end_dt = hkt().from_local_datetime(&end).single().unwrap();
            let start_dt = end_dt - Duration::days(days as i64);
            let start_ms = start_dt.timestamp() * 1000;
            let end_ms = end_dt.timestamp() * 1000;

            let t0 = Instant::now();

            let matches = if deep {
                search_transcripts(
                    &pattern,
                    start_ms,
                    end_ms,
                    tool.as_deref(),
                    role.as_deref(),
                    session.as_deref(),
                    full,
                )
            } else {
                search_prompts(
                    &pattern,
                    start_ms,
                    end_ms,
                    tool.as_deref(),
                    role.as_deref(),
                    session.as_deref(),
                    full,
                )
            };

            let elapsed = t0.elapsed();

            if json {
                print_json_search(&matches, full);
            } else {
                print_search(
                    &matches,
                    &pattern,
                    days,
                    deep,
                    role.as_deref(),
                    session.as_deref(),
                    full,
                );
                println!("({:.1}s)", elapsed.as_secs_f64());
            }
        }
        Some(Commands::Dump {
            session,
            include_tools,
            tool,
            json,
        }) => {
            run_dump(&session, include_tools, tool.as_deref(), json);
        }
        None => {
            let date_str = resolve_date(&cli.date);
            let prompts = scan_history(&date_str, cli.tool.as_deref());

            if cli.json {
                print_json_scan(&prompts, &date_str);
            } else {
                print_scan(&prompts, &date_str, cli.full);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempJsonl(PathBuf);

    impl Drop for TempJsonl {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn temp_jsonl(body: &str) -> TempJsonl {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "anam-unit-{}-{:?}-{}.jsonl",
            std::process::id(),
            std::thread::current().id(),
            nanos
        ));
        fs::write(&path, body).unwrap();
        TempJsonl(path)
    }

    // --- date_to_range_ms ---

    #[test]
    fn date_to_range_ms_hkt_midnight_exclusive_end() {
        let (start, end) = date_to_range_ms("2026-02-21");
        assert_eq!(start, 1_771_603_200_000);
        assert_eq!(end, 1_771_689_600_000);
        assert_eq!(end - start, 86_400_000);
        assert_eq!(
            ms_to_hkt(start).format("%Y-%m-%d %H:%M:%S %:z").to_string(),
            "2026-02-21 00:00:00 +08:00"
        );
        assert_eq!(
            ms_to_hkt(end).format("%Y-%m-%d %H:%M:%S %:z").to_string(),
            "2026-02-22 00:00:00 +08:00"
        );
    }

    #[test]
    fn date_to_range_ms_leap_day() {
        let (start, end) = date_to_range_ms("2024-02-29");
        assert_eq!(start, 1_709_136_000_000);
        assert_eq!(end - start, 86_400_000);
        assert_eq!(
            ms_to_hkt(start).format("%Y-%m-%d").to_string(),
            "2024-02-29"
        );
    }

    #[test]
    #[should_panic(expected = "Invalid date")]
    fn date_to_range_ms_rejects_non_iso() {
        let _ = date_to_range_ms("today");
    }

    // --- resolve_date ---

    #[test]
    fn resolve_date_literal_passthrough() {
        assert_eq!(resolve_date("2026-02-21"), "2026-02-21");
    }

    #[test]
    fn resolve_date_today_and_yesterday_are_hkt_calendar_days() {
        let today = resolve_date("today");
        let yesterday = resolve_date("yesterday");
        let today_d = NaiveDate::parse_from_str(&today, "%Y-%m-%d").unwrap();
        let yesterday_d = NaiveDate::parse_from_str(&yesterday, "%Y-%m-%d").unwrap();
        assert_eq!(today_d - yesterday_d, Duration::days(1));
        let now_hkt = Utc::now().with_timezone(&hkt()).date_naive();
        assert_eq!(today_d, now_hkt);
    }

    // --- ms_to_hkt ---

    #[test]
    fn ms_to_hkt_known_instant_and_millis() {
        let dt = ms_to_hkt(1_771_603_200_000);
        assert_eq!(dt.offset().local_minus_utc(), HKT_OFFSET);
        assert_eq!(
            dt.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-02-21 00:00:00"
        );
        let dt = ms_to_hkt(1_771_603_200_500);
        assert_eq!(dt.timestamp_subsec_millis(), 500);
    }

    #[test]
    fn ms_to_hkt_unix_epoch_is_eight_am_hkt() {
        let dt = ms_to_hkt(0);
        assert_eq!(
            dt.format("%Y-%m-%d %H:%M:%S %:z").to_string(),
            "1970-01-01 08:00:00 +08:00"
        );
    }

    #[test]
    fn ms_to_hkt_negative_millis_convert_not_now() {
        // -1500 ms is 1969-12-31 23:59:58.500 UTC = 1970-01-01 07:59:58.500 HKT.
        let got = ms_to_hkt(-1500);
        assert_eq!(
            got.format("%Y-%m-%d %H:%M:%S%.3f %:z").to_string(),
            "1970-01-01 07:59:58.500 +08:00"
        );
        let now_year = Utc::now().with_timezone(&hkt()).format("%Y").to_string();
        assert_ne!(got.format("%Y").to_string(), now_year);
    }

    #[test]
    fn ms_to_hkt_out_of_range_clamps_to_unix_epoch() {
        // Unrepresentable millis clamp to Unix epoch, not Utc::now().
        let got = ms_to_hkt(i64::MAX);
        assert_eq!(
            got.format("%Y-%m-%d %H:%M:%S %:z").to_string(),
            "1970-01-01 08:00:00 +08:00"
        );
        let now_year = Utc::now().with_timezone(&hkt()).format("%Y").to_string();
        assert_ne!(got.format("%Y").to_string(), now_year);
    }

    // --- matches_role ---

    #[test]
    fn matches_role_aliases() {
        assert!(matches_role("you", "you"));
        assert!(matches_role("you", "user"));
        assert!(matches_role("you", "me"));
        assert!(matches_role("you", "YOU"));
        assert!(!matches_role("claude", "you"));

        assert!(matches_role("claude", "claude"));
        assert!(matches_role("claude", "assistant"));
        assert!(matches_role("claude", "ai"));
        // claude/assistant/ai are Claude-specific, not a family matcher.
        assert!(!matches_role("opencode", "claude"));
        assert!(!matches_role("codex", "assistant"));
        assert!(!matches_role("opencode", "ai"));
        assert!(!matches_role("codex", "claude"));

        assert!(matches_role("opencode", "opencode"));
        assert!(!matches_role("claude", "opencode"));
        assert!(!matches_role("codex", "opencode"));

        assert!(matches_role("codex", "codex"));
        assert!(!matches_role("claude", "codex"));

        assert!(matches_role("tool", "TOOL"));
        assert!(!matches_role("you", "human"));

        // Filter and stored role are both case-insensitive.
        assert!(matches_role("You", "you"));
        assert!(matches_role("CLAUDE", "Assistant"));
        assert!(matches_role("OpenCode", "OPENCODE"));
    }

    // --- make_snippet ---

    #[test]
    fn make_snippet_short_string_is_unchanged() {
        assert_eq!(make_snippet("short", 0, 5), "short");
        assert_eq!(make_snippet("", 0, 0), "");
    }

    #[test]
    fn make_snippet_adds_ellipses_and_flattens_newlines() {
        let text = "x".repeat(200);
        let snippet = make_snippet(&text, 100, 105);
        assert!(snippet.starts_with("..."));
        assert!(snippet.ends_with("..."));

        let text = "line1\nline2 MATCH line3";
        let start = text.find("MATCH").unwrap();
        let snippet = make_snippet(text, start, start + 5);
        assert!(snippet.contains("line1 line2 MATCH line3"));
        assert!(!snippet.contains('\n'));
    }

    #[test]
    fn make_snippet_walks_off_cjk_char_boundaries() {
        let prefix = "字".repeat(14);
        let suffix = "字".repeat(30);
        let text = format!("{prefix}TARGET{suffix}");
        let start = text.find("TARGET").unwrap();
        assert!(!text.is_char_boundary(start.saturating_sub(40)));
        let snippet = make_snippet(&text, start, start + 6);
        assert!(snippet.contains("TARGET"));
    }

    // --- extract_text ---

    #[test]
    fn extract_text_string_blocks_and_skips() {
        assert_eq!(extract_text(&json!("hello")), "hello");
        assert_eq!(
            extract_text(&json!([
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"}
            ])),
            "a b"
        );
        assert_eq!(
            extract_text(&json!([{"type": "tool_use", "name": "Edit"}])),
            "[tool: Edit]"
        );
        assert_eq!(
            extract_text(&json!([
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "name": "Read"},
                {"type": "tool_result", "content": "ignored"}
            ])),
            "hi [tool: Read]"
        );
        assert_eq!(extract_text(&json!(null)), "");
        assert_eq!(extract_text(&json!({"type": "text", "text": "nope"})), "");
        assert_eq!(extract_text(&json!(["bare string"])), "");
        assert_eq!(
            extract_text(&json!([{"type": "thinking", "thinking": "secret"}])),
            ""
        );
    }

    // --- extract_dump_text ---

    #[test]
    fn extract_dump_text_omits_or_renders_tools() {
        let mixed = json!([
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "name": "Edit", "input": {"path": "a.rs"}},
            {"type": "tool_result", "content": "ok"}
        ]);
        assert_eq!(extract_dump_text(&mixed, false), "hello");
        let with_tools = extract_dump_text(&mixed, true);
        assert!(with_tools.contains("hello"));
        assert!(with_tools.contains("[tool_use: Edit]"));
        assert!(
            with_tools.contains("\"path\":\"a.rs\"") || with_tools.contains("\"path\": \"a.rs\"")
        );
        assert!(with_tools.contains("[tool_result] ok"));
        assert!(with_tools.contains('\n'));
        assert_eq!(extract_dump_text(&json!("plain"), false), "plain");
        assert_eq!(extract_dump_text(&json!(42), true), "");
    }

    #[test]
    fn extract_dump_text_recurses_into_tool_result_blocks() {
        let nested = json!([
            {"type": "tool_result", "content": [{"type": "text", "text": "inner"}]}
        ]);
        assert_eq!(extract_dump_text(&nested, true), "[tool_result] inner");
        assert_eq!(extract_dump_text(&nested, false), "");
    }

    // --- codex_session_id ---

    #[test]
    fn codex_session_id_takes_trailing_uuid() {
        let uuid = "12345678-1234-1234-1234-123456789abc";
        let path = PathBuf::from(format!("/tmp/rollout-2026-02-21T08-00-00-{uuid}.jsonl"));
        assert_eq!(codex_session_id(&path), uuid);
        assert_eq!(
            codex_session_id(Path::new("rollout-short.jsonl")),
            "rollout-short"
        );
        assert_eq!(codex_session_id(Path::new("")), "");
    }

    #[test]
    fn codex_session_id_walks_char_boundary_on_cut() {
        let stem = format!("é{}", "a".repeat(35));
        assert!(stem.len() > 36);
        let cut = stem.len() - 36;
        assert!(!stem.is_char_boundary(cut));
        let id = codex_session_id(&PathBuf::from(format!("{stem}.jsonl")));
        assert!(!id.is_empty());
        assert!(id.is_char_boundary(id.len()));
    }

    // --- synthetic JSONL fixtures per session format ---

    #[test]
    fn claude_history_jsonl_fixture() {
        let line = r#"{"timestamp":1771603200000,"sessionId":"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee","display":"shown","prompt":"raw"}"#;
        let entry: HistoryEntry = serde_json::from_str(line).unwrap();
        assert_eq!(
            entry.session_id.as_deref(),
            Some("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        );
        let ts = match &entry.timestamp {
            Some(serde_json::Value::Number(n)) => n.as_i64().unwrap(),
            other => panic!("expected numeric timestamp, got {other:?}"),
        };
        assert_eq!(ts, 1_771_603_200_000);
        assert_eq!(
            entry.display.clone().or(entry.prompt.clone()).as_deref(),
            Some("shown")
        );

        let prompt_only: HistoryEntry = serde_json::from_str(
            r#"{"timestamp":1771603200000,"sessionId":"s","prompt":"only prompt"}"#,
        )
        .unwrap();
        assert_eq!(
            prompt_only.display.or(prompt_only.prompt).as_deref(),
            Some("only prompt")
        );

        // String timestamps deserialize but scan_history only accepts Number.
        let iso: HistoryEntry = serde_json::from_str(
            r#"{"timestamp":"2026-02-21T00:00:00Z","sessionId":"s","prompt":"iso"}"#,
        )
        .unwrap();
        assert!(!matches!(iso.timestamp, Some(serde_json::Value::Number(_))));
    }

    #[test]
    fn claude_transcript_jsonl_fixture() {
        let user: TranscriptEntry = serde_json::from_str(
            r#"{"type":"user","timestamp":"2026-02-21T00:00:00.000Z","sessionId":"sess-1","message":{"content":"plain user"}}"#,
        )
        .unwrap();
        assert_eq!(user.entry_type.as_deref(), Some("user"));
        let text = extract_text(user.message.as_ref().unwrap().content.as_ref().unwrap());
        assert_eq!(text, "plain user");

        let assistant: TranscriptEntry = serde_json::from_str(
            r#"{"type":"assistant","timestamp":"2026-02-21T00:00:01.000Z","sessionId":"sess-1","message":{"content":[{"type":"text","text":"reply"},{"type":"tool_use","name":"Bash"}]}}"#,
        )
        .unwrap();
        let content = assistant.message.unwrap().content.unwrap();
        assert_eq!(extract_text(&content), "reply [tool: Bash]");
        assert_eq!(extract_dump_text(&content, false), "reply");
    }

    #[test]
    fn codex_rollout_jsonl_fixture() {
        let body = concat!(
            r#"{"type":"session_meta","timestamp":"2026-02-21T04:00:00.000Z"}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-02-21T04:00:00.500Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"auth middleware"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-02-21T04:00:01Z","payload":{"type":"message","role":"assistant","content":[{"text":"here is a patch"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-02-21T04:00:02Z","payload":{"type":"function_call","name":"exec"}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-02-21T04:00:03Z","payload":{"type":"message","role":"system","content":[{"text":"skip me"}]}}"#,
            "\n",
            r#"{"type":"response_item","timestamp":"2026-02-21T04:00:04Z","payload":{"type":"message","role":"user","content":[{"text":""}]}}"#,
            "\n",
            "not json\n",
        );
        let tmp = temp_jsonl(body);
        let msgs = codex_messages(&tmp.0);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].text, "auth middleware");
        let expected_ms = DateTime::parse_from_rfc3339("2026-02-21T04:00:00+00:00")
            .unwrap()
            .timestamp()
            * 1000;
        // `.timestamp() * 1000` drops the 500ms fraction on the source stamp.
        assert_eq!(msgs[0].timestamp_ms, expected_ms);
        assert_eq!(msgs[0].timestamp_ms % 1000, 0);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].text, "here is a patch");
    }

    #[test]
    fn opencode_json_fixtures() {
        let sess: OpenCodeSession = serde_json::from_str(
            r#"{"id":"ses_abc123","time":{"created":1771603200000,"updated":1771603300000},"extra":true}"#,
        )
        .unwrap();
        assert_eq!(sess.id.as_deref(), Some("ses_abc123"));
        assert_eq!(sess.time.as_ref().unwrap().created, Some(1_771_603_200_000));
        assert_eq!(sess.time.as_ref().unwrap().updated, Some(1_771_603_300_000));

        let msg: OpenCodeMessage = serde_json::from_str(
            r#"{"id":"msg_1","role":"user","time":{"created":1771603200000}}"#,
        )
        .unwrap();
        assert_eq!(msg.role.as_deref(), Some("user"));
        assert_eq!(msg.id.as_deref(), Some("msg_1"));

        let part: OpenCodePart =
            serde_json::from_str(r#"{"type":"text","text":"hello from opencode"}"#).unwrap();
        assert_eq!(part.text.as_deref(), Some("hello from opencode"));
    }
}
