use std::{
    env, fs, io,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Command, ExitCode, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use inquire::{
    ui::{RenderConfig, Styled},
    Select,
};
use serde::Deserialize;
use serde_json::Value;

const CANDIDATES: &[&str] = &[
    "codex",
    "claude",
    "cursor-agent",
    "copilot",
    "agy",
    "antigravity-cli",
];
const BAR_WIDTH: usize = 18;
const QUERY_TIMEOUT: Duration = Duration::from_secs(15);
const ASCII_LOGO: &str = r#"
              ▐
  ▝▀▖▞▀▌▞▀▖▛▀▖▜▀
  ▞▀▌▚▄▌▛▀ ▌ ▌▐ ▖
  ▝▀▘▗▄▘▝▀▘▘ ▘ ▀
"#;

#[derive(Clone)]
enum QuotaStatus {
    /// Percentage of the configured quota still available.
    Available(u8),
    /// A provider has independent pools that must be shown separately.
    Breakdown(Vec<QuotaBar>),
    TimedOut,
    Unavailable,
}

#[derive(Clone)]
struct QuotaBar {
    category: String,
    remaining: u8,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    data: OpenRouterKey,
}

#[derive(Debug, Deserialize)]
struct OpenRouterKey {
    // The API may represent an unlimited key as null, so these must be optional.
    limit: Option<f64>,
    usage: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ClaudeCredentials {
    #[serde(rename = "claudeAiOauth")]
    oauth: Option<ClaudeOauth>,
}

#[derive(Debug, Deserialize)]
struct ClaudeOauth {
    #[serde(rename = "accessToken")]
    access_token: String,
}

#[derive(Debug, Deserialize)]
struct ClaudeUsage {
    five_hour: Option<ClaudeWindow>,
    seven_day: Option<ClaudeWindow>,
}

#[derive(Debug, Deserialize)]
struct ClaudeWindow {
    utilization: f64,
}

/// Returns the percentage left on the OpenRouter API key, if it can be derived
/// from a successful response. No fallback percentage is ever invented.
fn get_openrouter_quota() -> Option<u8> {
    let key = env::var("OPENROUTER_API_KEY").ok()?;
    if key.trim().is_empty() {
        return None;
    }

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(2))
        .timeout_read(std::time::Duration::from_secs(3))
        .build();

    let response: OpenRouterResponse = agent
        .get("https://openrouter.ai/api/v1/key")
        .set("Authorization", &format!("Bearer {key}"))
        .call()
        .ok()?
        .into_json()
        .ok()?;

    let limit = response.data.limit?;
    let usage = response.data.usage?;
    if !limit.is_finite() || !usage.is_finite() || limit <= 0.0 {
        return None;
    }

    let remaining = ((limit - usage) / limit * 100.0).clamp(0.0, 100.0);
    Some(remaining.round() as u8)
}

/// Uses the same read-only OAuth usage endpoint that Claude Code's /status
/// command uses. An expired or unavailable local login is simply unavailable.
fn get_claude_quota() -> Option<u8> {
    let config_dir = env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| home_dir().map(|home| home.join(".claude")))?;
    let credential_bytes = fs::read(config_dir.join(".credentials.json"))
        .ok()
        .or_else(claude_macos_keychain_credentials)?;
    let credentials: ClaudeCredentials = serde_json::from_slice(&credential_bytes).ok()?;
    let token = credentials.oauth?.access_token;

    let usage: ClaudeUsage = quota_http_agent()
        .get("https://api.anthropic.com/api/oauth/usage")
        .set("Authorization", &format!("Bearer {token}"))
        .set("anthropic-beta", "oauth-2025-04-20")
        .call()
        .ok()?
        .into_json()
        .ok()?;

    [usage.five_hour, usage.seven_day]
        .into_iter()
        .flatten()
        .filter_map(|window| percentage_left_from_utilization(window.utilization))
        .min()
}

#[cfg(target_os = "macos")]
fn claude_macos_keychain_credentials() -> Option<Vec<u8>> {
    let output = Command::new("security")
        .args([
            "find-generic-password",
            "-s",
            "Claude Code-credentials",
            "-w",
        ])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

#[cfg(not(target_os = "macos"))]
fn claude_macos_keychain_credentials() -> Option<Vec<u8>> {
    None
}

fn percentage_left_from_utilization(utilization: f64) -> Option<u8> {
    if !utilization.is_finite() || !(0.0..=100.0).contains(&utilization) {
        return None;
    }
    Some((100.0 - utilization).round() as u8)
}

/// Calls the same account-entitlement endpoint used by GitHub Copilot clients.
/// `gh auth token` is only a read-only fallback for an existing GitHub login.
fn get_copilot_quota() -> Option<u8> {
    let token = ["GITHUB_TOKEN", "GH_TOKEN", "COPILOT_TOKEN"]
        .into_iter()
        .find_map(|name| env::var(name).ok().filter(|value| !value.trim().is_empty()))
        .or_else(github_cli_token)?;

    let response: Value = quota_http_agent()
        .get("https://api.github.com/copilot_internal/user")
        .set("Authorization", &format!("Bearer {token}"))
        .set("Accept", "application/vnd.github+json")
        .set("User-Agent", "agent-quota")
        .call()
        .ok()?
        .into_json()
        .ok()?;

    collect_remaining_percentages(&response).into_iter().min()
}

fn github_cli_token() -> Option<String> {
    let output = Command::new("gh")
        .args(["auth", "token"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8(output.stdout).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_owned())
}

/// Cursor's dashboard Connect RPC returns the active billing cycle's actual
/// included-usage percentage. Tokens are read from existing local credential
/// sources and never written back.
fn get_cursor_quota() -> Option<Vec<QuotaBar>> {
    let token = cursor_access_token()?;
    let response: Value = quota_http_agent()
        .post("https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage")
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Type", "application/json")
        .set("Connect-Protocol-Version", "1")
        .send_string("{}")
        .ok()?
        .into_json()
        .ok()?;

    let plan_usage = response.pointer("/planUsage")?;
    let pools: Vec<QuotaBar> = [("Auto", "autoPercentUsed"), ("API", "apiPercentUsed")]
        .into_iter()
        .filter_map(|(category, field)| {
            plan_usage
                .get(field)
                .and_then(Value::as_f64)
                .and_then(percentage_left_from_utilization)
                .map(|remaining| QuotaBar {
                    category: category.to_owned(),
                    remaining,
                })
        })
        .collect();
    if !pools.is_empty() {
        return Some(pools);
    }

    // Older Cursor clients expose only the aggregate included quota.
    plan_usage
        .get("totalPercentUsed")
        .and_then(Value::as_f64)
        .and_then(percentage_left_from_utilization)
        .map(|remaining| {
            vec![QuotaBar {
                category: "Included".to_owned(),
                remaining,
            }]
        })
}

fn cursor_access_token() -> Option<String> {
    [
        "CURSOR_ACCESS_TOKEN",
        "CURSOR_SESSION_TOKEN",
        "CURSOR_USAGE_SESSION_TOKEN",
    ]
    .into_iter()
    .find_map(|name| env::var(name).ok().filter(|value| !value.trim().is_empty()))
    .or_else(cursor_cli_config_token)
    .or_else(cursor_ide_sqlite_token)
    .or_else(cursor_macos_keychain_token)
}

fn cursor_cli_config_token() -> Option<String> {
    let path = home_dir()?.join(".config/cursor/auth.json");
    let value: Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    find_json_token(&value)
}

/// Cursor IDE records the same bearer token in its VS Code-compatible global
/// state database. Query it with SQLite's read-only mode so the running IDE is
/// never modified and its WAL is left untouched.
fn cursor_ide_sqlite_token() -> Option<String> {
    let database = cursor_state_database()?;
    if !database.is_file() {
        return None;
    }

    let output = Command::new("sqlite3")
        .arg("-readonly")
        .arg(database)
        .arg("SELECT value FROM ItemTable WHERE key='cursorAuth/accessToken' LIMIT 1;")
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8(output.stdout).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_owned())
}

fn find_json_token(value: &Value) -> Option<String> {
    match value {
        Value::Object(fields) => fields.iter().find_map(|(key, value)| {
            if matches!(key.as_str(), "accessToken" | "access_token") {
                value
                    .as_str()
                    .filter(|token| !token.trim().is_empty())
                    .map(str::to_owned)
            } else {
                find_json_token(value)
            }
        }),
        Value::Array(values) => values.iter().find_map(find_json_token),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn cursor_macos_keychain_token() -> Option<String> {
    let output = Command::new("security")
        .args(["find-generic-password", "-s", "cursor-access-token", "-w"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8(output.stdout).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_owned())
}

#[cfg(not(target_os = "macos"))]
fn cursor_macos_keychain_token() -> Option<String> {
    None
}

/// Resolves the user's home without assuming a Unix-only `HOME` variable.
fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn cursor_state_database() -> Option<PathBuf> {
    Some(
        home_dir()?
            .join("Library")
            .join("Application Support")
            .join("Cursor")
            .join("User")
            .join("globalStorage")
            .join("state.vscdb"),
    )
}

#[cfg(target_os = "linux")]
fn cursor_state_database() -> Option<PathBuf> {
    Some(
        home_dir()?
            .join(".config")
            .join("Cursor")
            .join("User")
            .join("globalStorage")
            .join("state.vscdb"),
    )
}

#[cfg(windows)]
fn cursor_state_database() -> Option<PathBuf> {
    Some(
        env::var_os("APPDATA")
            .map(PathBuf::from)?
            .join("Cursor")
            .join("User")
            .join("globalStorage")
            .join("state.vscdb"),
    )
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
fn cursor_state_database() -> Option<PathBuf> {
    None
}

fn quota_http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(2))
        .timeout_read(std::time::Duration::from_secs(3))
        .build()
}

/// Reads Codex's local app-server protocol. This asks only for the current
/// account snapshot; it does not start an agent turn or consume quota.
fn get_codex_quota() -> Option<u8> {
    let mut child = Command::new("codex")
        .args(["app-server", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let result = (|| {
        let mut stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        let mut stdout = BufReader::new(stdout);

        writeln!(
            stdin,
            r#"{{"id":1,"method":"initialize","params":{{"clientInfo":{{"name":"agent","version":"0.1.0"}}}}}}"#
        )
        .ok()?;
        // The app-server rejects requests made before this response arrives.
        read_json_rpc_response(&mut stdout, 1)?;
        writeln!(stdin, r#"{{"method":"initialized","params":{{}}}}"#).ok()?;

        writeln!(
            stdin,
            r#"{{"id":2,"method":"account/rateLimits/read","params":{{"excludeResetCreditDetails":true}}}}"#
        )
        .ok()?;
        let response = read_json_rpc_response(&mut stdout, 2)?;

        let limits = response.pointer("/result/rateLimits")?;
        [
            limits.pointer("/primary/usedPercent"),
            limits.pointer("/secondary/usedPercent"),
        ]
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .filter(|used| *used <= 100)
        .map(|used| 100_u8.saturating_sub(used as u8))
        // Both windows must permit another turn, so use the limiting window.
        .min()
    })();

    // The app-server stays alive to serve more requests. We only need one
    // snapshot, so terminate this short-lived child on every exit path.
    let _ = child.kill();
    let _ = child.wait();
    result
}

fn read_json_rpc_response(reader: &mut impl BufRead, id: u64) -> Option<Value> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let message: Value = serde_json::from_str(&line).ok()?;
        if message.get("id").and_then(Value::as_u64) == Some(id) {
            return Some(message);
        }
    }
}

/// Antigravity's print-mode /usage is a read-only CLI command. Recent versions
/// return structured JSON without creating an agent turn.
fn get_antigravity_quota(command: &str) -> Option<Vec<QuotaBar>> {
    let output = Command::new(command)
        .args([
            "-p",
            "/usage",
            "--output-format",
            "json",
            "--print-timeout",
            "5s",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let value: Value = serde_json::from_slice(&output.stdout).ok()?;
    let groups = value.pointer("/command/data/groups")?.as_array()?;
    let pools: Vec<QuotaBar> = groups
        .iter()
        .filter_map(|group| {
            let category = group.get("name")?.as_str()?.to_owned();
            let remaining = group
                .get("buckets")?
                .as_array()?
                .iter()
                .filter_map(|bucket| bucket.get("remaining_fraction")?.as_f64())
                .filter(|fraction| fraction.is_finite() && (0.0..=1.0).contains(fraction))
                // A group can contain several windows; show its constraining one.
                .min_by(f64::total_cmp)?;
            Some(QuotaBar {
                category,
                remaining: (remaining * 100.0).round() as u8,
            })
        })
        .collect();
    (!pools.is_empty()).then_some(pools)
}

/// Accepts only explicit provider-supplied percentage fields. It intentionally
/// does not infer a percentage from token counts or plan names.
fn collect_remaining_percentages(value: &Value) -> Vec<u8> {
    let mut percentages = Vec::new();
    collect_remaining_percentages_inner(value, &mut percentages);
    percentages
}

fn collect_remaining_percentages_inner(value: &Value, percentages: &mut Vec<u8>) {
    match value {
        Value::Object(fields) => {
            for (key, value) in fields {
                match key.as_str() {
                    "remainingPercent"
                    | "remaining_percent"
                    | "remainingPercentage"
                    | "percent_remaining" => {
                        if let Some(number) = percentage_value(value) {
                            percentages.push(number);
                        }
                    }
                    "usedPercent" | "used_percent" | "usedPercentage" => {
                        if let Some(number) = percentage_value(value) {
                            percentages.push(100 - number);
                        }
                    }
                    // Google Antigravity's structured `/usage` output uses a
                    // fraction for each provider bucket (0.0 through 1.0).
                    // This is still a provider-supplied remaining quota, not
                    // an inferred value from consumption statistics.
                    "remaining_fraction" => {
                        if let Some(fraction) = value
                            .as_f64()
                            .filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
                        {
                            percentages.push((fraction * 100.0).round() as u8);
                        }
                    }
                    _ => collect_remaining_percentages_inner(value, percentages),
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_remaining_percentages_inner(value, percentages);
            }
        }
        _ => {}
    }
}

fn percentage_value(value: &Value) -> Option<u8> {
    let number = value.as_f64()?;
    (number.is_finite() && (0.0..=100.0).contains(&number)).then(|| number.round() as u8)
}

fn quota_for(
    command: &str,
    openrouter_command: Option<&str>,
    openrouter_quota: Option<u8>,
) -> QuotaStatus {
    match command {
        "codex" => get_codex_quota()
            .map(QuotaStatus::Available)
            .unwrap_or(QuotaStatus::Unavailable),
        "agy" | "antigravity-cli" => get_antigravity_quota(command)
            .map(QuotaStatus::Breakdown)
            .unwrap_or(QuotaStatus::Unavailable),
        "claude" => get_claude_quota()
            .map(QuotaStatus::Available)
            .unwrap_or(QuotaStatus::Unavailable),
        "copilot" => get_copilot_quota()
            .map(QuotaStatus::Available)
            .unwrap_or(QuotaStatus::Unavailable),
        "cursor-agent" => get_cursor_quota()
            .map(QuotaStatus::Breakdown)
            .unwrap_or(QuotaStatus::Unavailable),
        command if openrouter_command == Some(command) => openrouter_quota
            .map(QuotaStatus::Available)
            .unwrap_or(QuotaStatus::Unavailable),
        _ => QuotaStatus::Unavailable,
    }
}

fn render_label(name: &str, status: &QuotaStatus) -> String {
    match status {
        QuotaStatus::Available(percent) => render_progress_line(name, None, *percent),
        QuotaStatus::Breakdown(pools) => pools
            .iter()
            .enumerate()
            .map(|(index, pool)| {
                let line = render_progress_line(
                    if index == 0 { name } else { "" },
                    Some(&pool.category),
                    pool.remaining,
                );
                // Inquire adds a two-column selection marker only before the
                // first physical line of a choice. Indent continuations by
                // the same width so their bars share the exact column.
                if index == 0 {
                    line
                } else {
                    format!("  {line}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        QuotaStatus::Unavailable => {
            let bar = "░".repeat(BAR_WIDTH);
            format!("{name:<14} [{bar}] (Unavailable)")
        }
        QuotaStatus::TimedOut => {
            let bar = "░".repeat(BAR_WIDTH);
            format!("{name:<14} [{bar}] (Timed Out)")
        }
    }
}

fn render_progress_line(name: &str, category: Option<&str>, percent: u8) -> String {
    let filled = usize::from(percent) * BAR_WIDTH / 100;
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(BAR_WIDTH - filled));
    match category {
        Some(category) => format!("{name:<14} [{bar}] ({percent}% Left)  {category}"),
        None => format!("{name:<14} [{bar}] ({percent}% Left)"),
    }
}

fn render_loading_label(name: &str, status: Option<&QuotaStatus>) -> String {
    match status {
        None => format!("{name:<14} [{}] (Loading...)", "░".repeat(BAR_WIDTH)),
        Some(QuotaStatus::Breakdown(pools)) => pools
            .iter()
            .enumerate()
            .map(|(index, pool)| {
                render_progress_line(
                    if index == 0 { name } else { "" },
                    Some(&pool.category),
                    pool.remaining,
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(status) => render_label(name, status),
    }
}

/// Repaints only the transient quota area beneath the logo. This makes every
/// detected command visible immediately while worker threads fetch quotas.
fn render_loading_screen(
    commands: &[String],
    statuses: &[Option<QuotaStatus>],
    previous_lines: usize,
) -> usize {
    if previous_lines > 0 {
        print!("\x1b[{previous_lines}A\x1b[J");
    }

    println!("Checking quotas...");
    let mut lines = 1;
    for (command, status) in commands.iter().zip(statuses) {
        let label = render_loading_label(command, status.as_ref());
        lines += label.lines().count();
        println!("{label}");
    }
    let _ = io::stdout().flush();
    lines
}

fn clear_loading_screen(rendered_lines: usize) {
    if rendered_lines > 0 {
        print!("\x1b[{rendered_lines}A\x1b[J");
        let _ = io::stdout().flush();
    }
}

struct AgentChoice {
    command: String,
    label: String,
}

impl std::fmt::Display for AgentChoice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label)
    }
}

fn main() -> ExitCode {
    // OpenRouter credentials do not establish quota for another provider.
    // AGY and antigravity-cli are Google Antigravity clients, so do not map
    // either of them here. The command using an OpenRouter key is explicit.
    let openrouter_command = env::var("AGENT_OPENROUTER_COMMAND").ok();
    let commands: Vec<String> = CANDIDATES
        .iter()
        .filter_map(|&command| which::which(command).ok().map(|_| command))
        .map(str::to_owned)
        .collect();

    if commands.is_empty() {
        eprintln!("No supported Agent CLI was found in PATH.");
        return ExitCode::SUCCESS;
    }

    println!("{ASCII_LOGO}");

    // Every provider call is independent. Querying them concurrently makes
    // startup bounded by the slowest CLI/API (not the sum of all timeouts).
    let timeout_commands = commands.clone();
    let (sender, receiver) = mpsc::channel();
    let command_count = commands.len();
    let mut statuses: Vec<Option<QuotaStatus>> = (0..command_count).map(|_| None).collect();
    let mut rendered_lines = render_loading_screen(&timeout_commands, &statuses, 0);
    for (index, command) in commands.into_iter().enumerate() {
        let sender = sender.clone();
        let openrouter_command = openrouter_command.clone();
        std::thread::spawn(move || {
            let openrouter_quota = (openrouter_command.as_deref() == Some(command.as_str()))
                .then(get_openrouter_quota)
                .flatten();
            let status = quota_for(&command, openrouter_command.as_deref(), openrouter_quota);
            let _ = sender.send((index, status));
        });
    }
    drop(sender);

    let deadline = Instant::now() + QUERY_TIMEOUT;
    while statuses.iter().any(Option::is_none) {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        match receiver.recv_timeout(remaining) {
            Ok((index, status)) => {
                statuses[index] = Some(status);
                rendered_lines =
                    render_loading_screen(&timeout_commands, &statuses, rendered_lines);
            }
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    for status in &mut statuses {
        if status.is_none() {
            *status = Some(QuotaStatus::TimedOut);
        }
    }
    rendered_lines = render_loading_screen(&timeout_commands, &statuses, rendered_lines);
    clear_loading_screen(rendered_lines);

    let choices: Vec<AgentChoice> = timeout_commands
        .into_iter()
        .zip(statuses)
        .map(|(command, status)| {
            let status = status.expect("all quota statuses are filled before rendering menu");
            AgentChoice {
                label: render_label(&command, &status),
                command,
            }
        })
        .collect();

    let render_config = RenderConfig::default().with_canceled_prompt_indicator(Styled::new(""));
    let selected = match Select::new("Choose an Agent", choices)
        .with_render_config(render_config)
        .prompt()
    {
        Ok(choice) => choice,
        Err(
            inquire::InquireError::OperationCanceled | inquire::InquireError::OperationInterrupted,
        ) => {
            erase_canceled_prompt();
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("Unable to render Agent menu: {error}");
            return ExitCode::from(1);
        }
    };

    launch_selected(&selected.command)
}

/// `inquire` renders a final prompt line on cancellation. Remove that line so
/// Esc/Ctrl-C behaves as a silent exit, including when no color is available.
fn erase_canceled_prompt() {
    print!("\x1b[1A\x1b[2K\r");
    let _ = io::stdout().flush();
}

#[cfg(unix)]
fn launch_selected(command: &str) -> ExitCode {
    // exec never returns on success: the launcher process is replaced in-place.
    let error = Command::new(command).exec();
    eprintln!("Failed to start {command}: {error}");
    map_io_error(error)
}

#[cfg(windows)]
fn launch_selected(command: &str) -> ExitCode {
    // Windows has no `exec(2)`. Spawn and exit immediately so the launcher
    // leaves no foreground process once the selected CLI starts.
    match Command::new(command).spawn() {
        Ok(_) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Failed to start {command}: {error}");
            map_io_error(error)
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn launch_selected(command: &str) -> ExitCode {
    match Command::new(command).spawn() {
        Ok(_) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Failed to start {command}: {error}");
            map_io_error(error)
        }
    }
}

fn map_io_error(error: io::Error) -> ExitCode {
    match error.kind() {
        io::ErrorKind::NotFound => ExitCode::from(127),
        _ => ExitCode::from(1),
    }
}
