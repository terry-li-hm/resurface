use chrono::{DateTime, Duration, FixedOffset, NaiveDate, TimeZone, Utc};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use regex::Regex;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
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
#[command(name = "resurface", about = "Search AI coding chat history")]
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

        /// Search full transcripts (user + assistant), not just prompts
        #[arg(long)]
        deep: bool,

        /// Filter by tool
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
    vec![
        ("Claude".into(), home.join(".claude/history.jsonl")),
        ("Codex".into(), home.join(".codex/history.jsonl")),
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

fn ms_to_hkt(ms: i64) -> DateTime<FixedOffset> {
    DateTime::from_timestamp(ms / 1000, ((ms % 1000) * 1_000_000) as u32)
        .unwrap_or_else(|| Utc::now())
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
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|x| x == "json")
                        .unwrap_or(false)
                })
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

// --- Search prompts (fast) ---

fn search_prompts(
    pattern: &str,
    start_ms: i64,
    end_ms: i64,
    tool_filter: Option<&str>,
) -> Vec<SearchMatch> {
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
                let prompt_text = entry.display.or(entry.prompt).unwrap_or_default();
                if let Some(m) = regex.find(&prompt_text) {
                    let dt = ms_to_hkt(ts);
                    let session = entry.session_id.unwrap_or_else(|| "unknown".into());
                    let snippet = make_snippet(&prompt_text, m.start(), m.end());

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
            if let Some(m) = regex.find(&p.prompt) {
                let snippet = make_snippet(&p.prompt, m.start(), m.end());
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

    matches.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
    matches
}

// --- Search transcripts (deep) ---

fn search_transcripts(pattern: &str, start_ms: i64, end_ms: i64) -> Vec<SearchMatch> {
    let regex = Regex::new(&format!("(?i){}", pattern)).unwrap_or_else(|_| {
        eprintln!("Invalid regex pattern: {}", pattern);
        std::process::exit(1);
    });

    let proj_dir = projects_dir();
    if !proj_dir.exists() {
        return Vec::new();
    }

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
                                    .as_secs() as i64;
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
    let all_matches: Vec<SearchMatch> = session_files
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
                    Some("user") | Some("assistant") => entry.entry_type.as_deref().unwrap(),
                    _ => continue,
                };

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
                    let role = if entry_type == "user" {
                        "you"
                    } else {
                        "claude"
                    };
                    let snippet = make_snippet(&text, m.start(), m.end());

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

    let mut matches = all_matches;
    matches.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
    matches
}

// --- Helpers ---

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

// --- Display ---

fn print_scan(prompts: &[Prompt], date_str: &str, full: bool) {
    let mut sessions: BTreeMap<String, SessionInfo> = BTreeMap::new();
    for p in prompts {
        let entry = sessions
            .entry(p.session_full.clone())
            .or_insert_with(|| {
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

fn print_search(matches: &[SearchMatch], pattern: &str, days: u32, deep: bool) {
    let mode = if deep {
        "full transcripts"
    } else {
        "prompts only"
    };
    println!("Search: \"{}\" (last {} days, {})", pattern, days, mode);

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
            let snippet: String = m.snippet.chars().take(100).collect();
            println!(
                "    {} [{}] {:9} {}",
                m.time_str, m.session, role_tag, snippet
            );
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

fn print_json_search(matches: &[SearchMatch]) {
    let output: Vec<_> = matches
        .iter()
        .map(|m| {
            serde_json::json!({
                "date": m.date,
                "time": m.time_str,
                "timestamp": m.timestamp_ms,
                "session": m.session,
                "role": m.role,
                "snippet": m.snippet,
                "tool": m.tool,
            })
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
            deep,
            tool,
            json,
        }) => {
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
                search_transcripts(&pattern, start_ms, end_ms)
            } else {
                search_prompts(&pattern, start_ms, end_ms, tool.as_deref())
            };

            let elapsed = t0.elapsed();

            if json {
                print_json_search(&matches);
            } else {
                print_search(&matches, &pattern, days, deep);
                println!("({:.1}s)", elapsed.as_secs_f64());
            }
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
