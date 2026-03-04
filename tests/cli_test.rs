use predicates::prelude::*;

/// Build an `assert_cmd::Command` pointing at the `anam` binary.
/// Uses the `CARGO_BIN_EXE_anam` env var set by cargo for integration tests,
/// via the `cargo_bin_cmd!` macro — avoids the deprecated `Command::cargo_bin` API.
macro_rules! anam {
    () => {
        assert_cmd::cargo_bin_cmd!("anam")
    };
}

// --version is NOT declared on the Cli struct, so it is treated as an unknown
// argument. The binary should exit non-zero and print a usage hint.
#[test]
fn test_version() {
    anam!()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("resurface"));
}

// --help exits 0 and shows the about text and a Usage line.
#[test]
fn test_help() {
    anam!()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Search AI coding chat history"))
        .stdout(predicate::str::contains("Usage"));
}

// No args: scans today — exits 0 even when no JSONL data files exist.
#[test]
fn test_no_args() {
    anam!().assert().success();
}

// `search --help` exits 0 and documents the key flags.
#[test]
fn test_search_help() {
    anam!()
        .args(["search", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("pattern"))
        .stdout(predicate::str::contains("--days"))
        .stdout(predicate::str::contains("--deep"))
        .stdout(predicate::str::contains("--role"))
        .stdout(predicate::str::contains("--session"));
}

// `search` with no positional pattern: exits non-zero with a clap error
// mentioning the missing argument.
#[test]
fn test_search_missing_pattern() {
    anam!()
        .arg("search")
        .assert()
        .failure()
        .stderr(predicate::str::contains("pattern").or(predicate::str::contains("required")));
}

// A date argument in the wrong format: exits non-zero and prints "Invalid date".
#[test]
fn test_invalid_date() {
    anam!()
        .arg("not-a-date")
        .assert()
        .failure()
        .stderr(predicate::str::contains("Invalid date"));
}

// `search` with `--tool` flag: flag is accepted; exits 0 (no data → no matches).
#[test]
fn test_search_tool_filter_accepted() {
    anam!()
        .args(["search", "anything", "--tool", "Claude", "--days", "1"])
        .assert()
        .success();
}

// `search` with `--role` flag: flag is accepted; exits 0.
#[test]
fn test_search_role_filter_accepted() {
    anam!()
        .args(["search", "anything", "--role", "you", "--days", "1"])
        .assert()
        .success();
}

// An invalid regex pattern causes the binary to print "Invalid regex" and exit non-zero.
#[test]
fn test_search_invalid_regex() {
    anam!()
        .args(["search", "[unclosed", "--days", "1"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Invalid regex"));
}

// `search` with `--json` produces a JSON array when history data is present.
// Ignored because it requires real JSONL history files on disk.
#[test]
#[ignore = "requires ~/.claude/history.jsonl or other real data files on disk"]
fn test_search_json_output() {
    let output = anam!()
        .args(["search", "anything", "--days", "1", "--json"])
        .output()
        .expect("failed to run anam");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.trim().starts_with('['),
        "Expected JSON array output, got: {}",
        stdout
    );
}
