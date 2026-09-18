use std::collections::HashMap;
use std::fs;
use std::path::Path;

use rusqlite::Connection;

use crate::error::AgfError;
use crate::model::{Agent, Session};

use super::{collapse_whitespace, project_name_from_path, read_head_lines, truncate};

const MAX_SUMMARY_CHARS: usize = 160;
const HEAD_LOG_BYTES: u64 = 128 * 1024;
const HEAD_LOG_LINES: usize = 10;

pub fn scan() -> Result<Vec<Session>, AgfError> {
    let antigravity_dir = crate::config::antigravity_dir()?;
    scan_from(&antigravity_dir)
}

pub(crate) fn scan_from(base_dir: &Path) -> Result<Vec<Session>, AgfError> {
    if !base_dir.exists() {
        return Ok(Vec::new());
    }

    let brain_dir = base_dir.join("brain");
    let db_path = base_dir.join("conversation_summaries.db");

    let mut sessions: HashMap<String, Session> = HashMap::new();

    // 1. Scan SQLite database if it exists
    if db_path.exists()
        && let Ok(conn) = Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
    {
        scan_db(&conn, &brain_dir, &mut sessions)?;
    }

    // 2. Scan brain directory for any additional/orphaned sessions or to enrich prompts
    if brain_dir.is_dir()
        && let Ok(entries) = fs::read_dir(&brain_dir)
    {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let dir_name = entry.file_name().to_string_lossy().to_string();
            if !is_valid_uuid(&dir_name) {
                continue;
            }

            let transcript_path = entry
                .path()
                .join(".system_generated")
                .join("logs")
                .join("transcript.jsonl");

            if let Some(existing) = sessions.get_mut(&dir_name) {
                // Enrich existing session with prompt from transcript if not already present
                if transcript_path.exists()
                    && let Some((prompt, time_ms)) = extract_first_prompt(&transcript_path)
                {
                    if !existing.summaries.contains(&prompt) {
                        existing.summaries.push(prompt);
                    }
                    if existing.timestamp <= 0 && time_ms > 0 {
                        existing.timestamp = time_ms;
                    }
                }
            } else if transcript_path.exists() {
                // Orphaned or untracked brain session
                if let Some(session) =
                    parse_brain_session(&entry.path(), &dir_name, &transcript_path)
                {
                    sessions.insert(dir_name, session);
                }
            }
        }
    }

    Ok(sessions.into_values().collect())
}

fn scan_db(
    conn: &Connection,
    brain_dir: &Path,
    sessions: &mut HashMap<String, Session>,
) -> Result<(), AgfError> {
    let mut stmt = conn.prepare(
        "SELECT conversation_id, \
                title, \
                preview, \
                workspace_uris, \
                last_modified_time, \
                parent_conversation_id, \
                nesting_depth \
         FROM conversation_summaries",
    )?;

    let rows = stmt.query_map([], |row| {
        let conversation_id: String = row.get(0)?;
        let title: String = row.get(1).unwrap_or_default();
        let preview: String = row.get(2).unwrap_or_default();
        let workspace_uris: String = row.get(3).unwrap_or_default();
        let last_modified_time: String = row.get(4).unwrap_or_default();
        let parent_conversation_id: String = row.get(5).unwrap_or_default();
        let nesting_depth: i64 = row.get(6).unwrap_or(0);
        Ok((
            conversation_id,
            title,
            preview,
            workspace_uris,
            last_modified_time,
            parent_conversation_id,
            nesting_depth,
        ))
    })?;

    for row in rows.flatten() {
        let (
            conversation_id,
            title,
            preview,
            workspace_uris,
            last_modified_time,
            parent_conversation_id,
            nesting_depth,
        ) = row;

        let project_path = parse_workspace_uri(&workspace_uris);
        let project_name = if project_path.is_empty() {
            "unknown".to_string()
        } else {
            project_name_from_path(&project_path)
        };

        let mut timestamp = parse_timestamp(&last_modified_time);
        if timestamp <= 0 {
            // Fallback to brain folder mtime
            let folder = brain_dir.join(&conversation_id);
            timestamp = folder_mtime_ms(&folder).unwrap_or(0);
        }

        let mut summaries = Vec::new();
        let clean_title = collapse_whitespace(&title);
        if !clean_title.is_empty() {
            summaries.push(truncate(&clean_title, MAX_SUMMARY_CHARS));
        }
        let clean_preview = collapse_whitespace(&preview);
        if !clean_preview.is_empty() && clean_preview != clean_title {
            summaries.push(truncate(&clean_preview, MAX_SUMMARY_CHARS));
        }

        let interactive = parent_conversation_id.trim().is_empty() && nesting_depth == 0;

        sessions.insert(
            conversation_id.clone(),
            Session {
                agent: Agent::Antigravity,
                session_id: conversation_id,
                project_name,
                project_path,
                summaries,
                timestamp,
                git_branch: None,
                worktree: None,
                recap: None,
                interactive,
            },
        );
    }

    Ok(())
}

fn parse_brain_session(
    folder_path: &Path,
    conversation_id: &str,
    transcript_path: &Path,
) -> Option<Session> {
    let (prompt, mut timestamp) = extract_first_prompt(transcript_path)?;
    if timestamp <= 0 {
        timestamp = folder_mtime_ms(folder_path).unwrap_or(0);
    }

    let summaries = if prompt.is_empty() {
        Vec::new()
    } else {
        vec![prompt]
    };

    Some(Session {
        agent: Agent::Antigravity,
        session_id: conversation_id.to_string(),
        project_name: "unknown".to_string(),
        project_path: String::new(),
        summaries,
        timestamp,
        git_branch: None,
        worktree: None,
        recap: None,
        interactive: true,
    })
}

/// Extract first user prompt and timestamp from transcript.jsonl
fn extract_first_prompt(path: &Path) -> Option<(String, i64)> {
    let lines = read_head_lines(path, HEAD_LOG_LINES, HEAD_LOG_BYTES);
    for line in lines {
        let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let step_type = val.get("type").and_then(serde_json::Value::as_str);
        if step_type == Some("USER_INPUT") {
            let timestamp = val
                .get("created_at")
                .and_then(serde_json::Value::as_str)
                .and_then(parse_iso8601_ms)
                .unwrap_or(0);

            if let Some(content) = val.get("content").and_then(serde_json::Value::as_str) {
                let cleaned = clean_user_prompt(content);
                if !cleaned.is_empty() {
                    return Some((truncate(&cleaned, MAX_SUMMARY_CHARS), timestamp));
                }
            }
            return Some((String::new(), timestamp));
        }
    }
    None
}

/// Clean `<USER_REQUEST>...</USER_REQUEST>` and discard metadata wrappers
pub(crate) fn clean_user_prompt(raw: &str) -> String {
    let mut text = raw;

    // Strip <USER_REQUEST> tags if present
    if let Some(start) = text.find("<USER_REQUEST>") {
        let after_start = &text[start + "<USER_REQUEST>".len()..];
        text = if let Some(end) = after_start.find("</USER_REQUEST>") {
            &after_start[..end]
        } else {
            after_start
        };
    } else if let Some(start) = text.find("<ADDITIONAL_METADATA>") {
        text = &text[..start];
    }

    // Strip any trailing ADDITIONAL_METADATA or USER_SETTINGS_CHANGE
    if let Some(idx) = text.find("<ADDITIONAL_METADATA>") {
        text = &text[..idx];
    }
    if let Some(idx) = text.find("<USER_SETTINGS_CHANGE>") {
        text = &text[..idx];
    }

    collapse_whitespace(text.trim())
}

/// Parse first file:// URI from workspace_uris JSON string (e.g. `["file:///path/to/project"]`)
pub(crate) fn parse_workspace_uri(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return String::new();
    }

    // Try parsing as JSON array
    let uris: Vec<String> = if let Ok(arr) = serde_json::from_str(trimmed) {
        arr
    } else if trimmed.starts_with("file://") {
        vec![trimmed.to_string()]
    } else {
        vec![trimmed.to_string()]
    };

    let Some(first) = uris.first() else {
        return String::new();
    };

    strip_file_uri(first)
}

fn strip_file_uri(uri: &str) -> String {
    let path_str = uri.strip_prefix("file://").unwrap_or(uri);
    // Simple percent-decode if contains '%'
    if path_str.contains('%') {
        percent_decode(path_str)
    } else {
        path_str.to_string()
    }
}

fn percent_decode(s: &str) -> String {
    let mut bytes = Vec::with_capacity(s.len());
    let mut chars = s.as_bytes().iter().copied();
    while let Some(b) = chars.next() {
        if b == b'%' {
            let h1 = chars.next();
            let h2 = chars.next();
            if let (Some(h1), Some(h2)) = (h1, h2)
                && let Ok(hex_byte) = u8::from_str_radix(
                    &format!("{}{}", h1 as char, h2 as char),
                    16,
                )
            {
                bytes.push(hex_byte);
                continue;
            }
        }
        bytes.push(b);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn is_valid_uuid(s: &str) -> bool {
    s.len() == 36
        && s.chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-')
}

fn parse_timestamp(s: &str) -> i64 {
    parse_iso8601_ms(s).unwrap_or(0)
}

fn parse_iso8601_ms(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
        .or_else(|| {
            // Try parsing "YYYY-MM-DD HH:MM:SS.ssssss+00:00"
            chrono::DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f%:z")
                .ok()
                .map(|dt| dt.timestamp_millis())
        })
}

fn folder_mtime_ms(path: &Path) -> Option<i64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_workspace_uri_handles_json_and_plain() {
        assert_eq!(
            parse_workspace_uri("[\"file:///home/user/project\"]"),
            "/home/user/project"
        );
        assert_eq!(
            parse_workspace_uri("[\"file:///home/user/my%20project\"]"),
            "/home/user/my project"
        );
        assert_eq!(
            parse_workspace_uri("file:///home/user/plain"),
            "/home/user/plain"
        );
        assert_eq!(parse_workspace_uri(""), "");
        assert_eq!(parse_workspace_uri("[]"), "");
    }

    #[test]
    fn clean_user_prompt_strips_metadata_tags() {
        let raw = "<USER_REQUEST>\nFix the bug in login\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\ntime\n</ADDITIONAL_METADATA>";
        assert_eq!(clean_user_prompt(raw), "Fix the bug in login");

        let plain = "Simple prompt without tags";
        assert_eq!(clean_user_prompt(plain), "Simple prompt without tags");
    }

    #[test]
    fn parse_iso8601_ms_handles_rfc3339_and_sqlite_formats() {
        let rfc = "2026-09-18T05:31:00Z";
        assert!(parse_iso8601_ms(rfc).unwrap() > 0);

        let sqlite_dt = "2026-09-18 05:31:00.884634923+00:00";
        assert!(parse_iso8601_ms(sqlite_dt).unwrap() > 0);
    }
}
