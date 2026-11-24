//! GPUs for a container on a Linux host, from a CDI spec.
//!
//! `docker run --gpus all`, Compose's `deploy.resources.reservations.devices`
//! and `--device nvidia.com/gpu=0` all name a device the host's Container
//! Device Interface spec describes (`nvidia-ctk cdi generate` writes one to
//! `/etc/cdi/nvidia.yaml`). Kiri reads that spec at every start, so a driver
//! update is picked up, and turns its `containerEdits` into what the guest
//! already does: device nodes (with their cgroup rules), bind mounts, env and
//! groups. The agent runs on the host's own kernel and filesystem there, so
//! the host's paths can be bound as they are.
//!
//! Hooks are not run, but the one the NVIDIA spec needs for CUDA is done in
//! place of it: `create-symlinks` makes `libcuda.so.1` from `libcuda.so.<v>`.
//! The driver's libraries are bound under [`LIB_DIR`] (where the `nvidia/cuda`
//! images already look) with those names, and `LD_LIBRARY_PATH` names it, so
//! no `ld.so.cache` is needed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Where the host's CDI specs are looked for, as the CDI spec says.
pub const SPEC_DIRS: [&str; 2] = ["/etc/cdi", "/var/run/cdi"];

/// Where the driver's libraries are bound in the container.
const LIB_DIR: &str = "/usr/local/nvidia/lib64";

/// What the specs for a container's devices add to it.
#[derive(Debug, Default, PartialEq)]
pub struct Edits {
    /// `device\t<host>\t<container>\t<perms>` and `group\t<gid>` lines for the guest's `opts`.
    pub opts: Vec<String>,
    /// `<host>\t<container>\t<flags>` lines for the guest's mounts.
    pub mounts: Vec<String>,
    /// `KEY=value`, for the container's environment: the ones it sets itself win.
    pub env: Vec<String>,
}

impl Edits {
    /// Adds what the spec sets to `env`, except the keys it already has;
    /// `LD_LIBRARY_PATH` gets [`LIB_DIR`] in front.
    pub fn apply_env(&self, env: &mut Vec<String>) {
        for e in &self.env {
            let key = e.split('=').next().unwrap_or_default();
            if key == "LD_LIBRARY_PATH" {
                continue;
            }
            if !env.iter().any(|x| x.split('=').next() == Some(key)) {
                env.push(e.clone());
            }
        }
        if let Some(path) = self
            .env
            .iter()
            .find_map(|e| e.strip_prefix("LD_LIBRARY_PATH="))
        {
            match env.iter_mut().find(|x| x.starts_with("LD_LIBRARY_PATH=")) {
                Some(x)
                    if x.split_once('=')
                        .is_some_and(|(_, v)| v.split(':').any(|d| d == path)) => {}
                Some(x) => {
                    *x = format!("LD_LIBRARY_PATH={path}:{}", &x["LD_LIBRARY_PATH=".len()..])
                }
                None => env.push(format!("LD_LIBRARY_PATH={path}")),
            }
        }
    }
}

/// `vendor.example/class=name`: how CDI names a device.
pub fn is_qualified(name: &str) -> bool {
    match name.split_once('=') {
        Some((kind, device)) => {
            !device.is_empty()
                && kind.split_once('/').is_some_and(|(vendor, class)| {
                    vendor.contains('.') && !class.is_empty() && !vendor.is_empty()
                })
        }
        None => false,
    }
}

/// Which devices of a kind a container asked for.
#[derive(Debug, PartialEq)]
enum Pick {
    All,
    First(usize),
    Name(String),
}

/// The devices HostConfig asks for, as (CDI kind, which): `--gpus` and
/// Compose's device reservations arrive as DeviceRequests, `--device` with a
/// CDI name as a `Devices` entry whose PathOnHost is that name.
fn requested(opts: &Value) -> Result<Vec<(String, Pick)>, String> {
    let mut out = Vec::new();
    for r in opts["DeviceRequests"].as_array().into_iter().flatten() {
        let driver = r["Driver"].as_str().unwrap_or("");
        let caps: Vec<&str> = r["Capabilities"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|c| c.as_array().into_iter().flatten())
            .filter_map(|c| c.as_str())
            .collect();
        let ids: Vec<&str> = r["DeviceIDs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| c.as_str())
            .collect();
        match driver {
            d if vendor_kind(d).is_some() && (caps.contains(&"gpu") || caps.contains(&"nvidia")) => {
                let kind = vendor_kind(d).unwrap_or_default();
                let count = r["Count"].as_i64().unwrap_or(0);
                if !ids.is_empty() {
                    for id in ids {
                        out.push(if is_qualified(id) {
                            split(id)
                        } else {
                            (kind.to_owned(), Pick::Name(id.to_owned()))
                        });
                    }
                } else if count < 0 {
                    out.push((kind.to_owned(), Pick::All));
                } else if count > 0 {
                    out.push((kind.to_owned(), Pick::First(count as usize)));
                }
            }
            "cdi" => {
                for id in ids {
                    if !is_qualified(id) {
                        return Err(format!(
                            "could not select device driver \"cdi\" with capabilities: [{}]: {id:?} is not a CDI device name",
                            caps.join(" ")
                        ));
                    }
                    out.push(split(id));
                }
            }
            _ => {
                return Err(format!(
                    "could not select device driver {driver:?} with capabilities: [{}]: Kiri supports GPUs through CDI specs (driver \"nvidia\" or \"\", \"amd\" or \"intel\", with capability \"gpu\", or driver \"cdi\")",
                    caps.iter()
                        .map(|c| format!("[{c}]"))
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
            }
        }
    }
    for d in opts["Devices"].as_array().into_iter().flatten() {
        if let Some(n) = d["PathOnHost"].as_str().filter(|n| is_qualified(n)) {
            out.push(split(n));
        }
    }
    Ok(out)
}

fn split(qualified: &str) -> (String, Pick) {
    let (kind, name) = qualified.split_once('=').unwrap_or((qualified, ""));
    (
        kind.to_owned(),
        if name == "all" {
            Pick::All
        } else {
            Pick::Name(name.to_owned())
        },
    )
}

/// The CDI kind `--gpus driver=<name>` means. NVIDIA is the default driver
/// (what `--gpus all` sends); `amd` and `intel` name the other two vendors.
fn vendor_kind(driver: &str) -> Option<&'static str> {
    match driver {
        "" | "nvidia" => Some("nvidia.com/gpu"),
        "amd" => Some("amd.com/gpu"),
        "intel" => Some("intel.com/gpu"),
        _ => None,
    }
}

/// What to do when the host has no spec for `kind`.
fn missing_hint(kind: &str) -> &'static str {
    match kind {
        "nvidia.com/gpu" => "for NVIDIA, install nvidia-container-toolkit and run: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml",
        "amd.com/gpu" => "for AMD, install the AMD container toolkit and run: sudo amd-ctk cdi generate --output=/etc/cdi/amd.json; or pass --device /dev/kfd --device /dev/dri",
        "intel.com/gpu" => "for Intel, pass --device /dev/dri, or add a CDI spec for intel.com/gpu to /etc/cdi",
        _ => "add a CDI spec for it to /etc/cdi",
    }
}

/// Whether `opts` (a container's kept HostConfig) asks for CDI devices.
pub fn wanted(opts: &Value) -> bool {
    !opts["DeviceRequests"]
        .as_array()
        .is_none_or(|a| a.is_empty())
        || opts["Devices"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|d| d["PathOnHost"].as_str().is_some_and(is_qualified))
}

/// What the container's devices add to it, from the specs in [`SPEC_DIRS`].
/// Nothing when it asks for none. Linux hosts only: elsewhere the paths a
/// spec names are not the ones the guest sees.
pub fn edits(opts: &Value) -> Result<Edits, String> {
    if !wanted(opts) {
        return Ok(Edits::default());
    }
    if !cfg!(target_os = "linux") {
        return Err("Kiri supports GPUs on Linux hosts only".into());
    }
    edits_in(opts, &SPEC_DIRS.map(PathBuf::from))
}

fn edits_in(opts: &Value, dirs: &[PathBuf]) -> Result<Edits, String> {
    let requests = requested(opts)?;
    let specs = load(dirs);
    let mut picked: Vec<(usize, &Value)> = Vec::new(); // (spec, device)
    for (kind, pick) in &requests {
        let found: Vec<(usize, &Value)> = specs
            .iter()
            .enumerate()
            .filter(|(_, s)| s["kind"].as_str() == Some(kind))
            .collect();
        if found.is_empty() {
            return Err(format!(
                "could not select device driver \"\" with capabilities: [[gpu]]: no CDI spec for {kind} in {} ({})",
                dirs.iter()
                    .map(|d| d.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" or "),
                missing_hint(kind)
            ));
        }
        let devices = |i: usize| specs[i]["devices"].as_array().into_iter().flatten();
        let name_of = |d: &Value| match &d["name"] {
            Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        let indexed = |d: &Value| name_of(d).bytes().all(|b| b.is_ascii_digit());
        match pick {
            Pick::Name(n) => {
                let Some(d) = found
                    .iter()
                    .find_map(|(i, _)| devices(*i).find(|d| name_of(d) == *n).map(|d| (*i, d)))
                else {
                    return Err(format!("unresolvable CDI devices {kind}={n}"));
                };
                picked.push(d);
            }
            Pick::All => {
                // The spec's own "all" if it has one, else every device with an index for a name.
                match found
                    .iter()
                    .find_map(|(i, _)| devices(*i).find(|d| name_of(d) == "all").map(|d| (*i, d)))
                {
                    Some(d) => picked.push(d),
                    None => found.iter().for_each(|(i, _)| {
                        picked.extend(devices(*i).filter(|d| indexed(d)).map(|d| (*i, d)))
                    }),
                }
            }
            Pick::First(n) => {
                let all: Vec<(usize, &Value)> = found
                    .iter()
                    .flat_map(|(i, _)| devices(*i).filter(|d| indexed(d)).map(|d| (*i, d)))
                    .collect();
                if all.len() < *n {
                    return Err(format!(
                        "could not select device driver \"\" with capabilities: [[gpu]]: requested {n} {kind} devices but only {} found",
                        all.len()
                    ));
                }
                picked.extend(all.into_iter().take(*n));
            }
        }
    }
    let mut spec_ids: Vec<usize> = picked.iter().map(|(i, _)| *i).collect();
    spec_ids.dedup();
    let mut all_edits: Vec<&Value> = Vec::new();
    let mut seen = BTreeSet::new();
    for i in spec_ids {
        if seen.insert(i) {
            all_edits.push(&specs[i]["containerEdits"]);
        }
    }
    all_edits.extend(picked.iter().map(|(_, d)| &d["containerEdits"]));
    // NVIDIA's libraries are bound together under one directory; another vendor's stay where its spec puts them.
    let relocate = requests.iter().any(|(k, _)| k.starts_with("nvidia.com/"));
    apply(&all_edits, relocate)
}

fn apply(edits: &[&Value], relocate: bool) -> Result<Edits, String> {
    let mut out = Edits::default();
    let mut nodes: BTreeMap<String, String> = BTreeMap::new(); // container path → line
    let mut mounts: Vec<(String, String, bool)> = Vec::new();
    let mut links: Vec<(String, String)> = Vec::new();
    let mut groups = BTreeSet::new();
    let mut env: Vec<String> = Vec::new();
    for e in edits {
        for n in e["deviceNodes"].as_array().into_iter().flatten() {
            let path = n["path"].as_str().unwrap_or("");
            let host = n["hostPath"]
                .as_str()
                .filter(|h| !h.is_empty())
                .unwrap_or(path);
            let perms = n["permissions"]
                .as_str()
                .filter(|p| !p.is_empty())
                .unwrap_or("rwm");
            for p in [host, path] {
                if !p.starts_with('/')
                    || p.contains(['\t', '\n'])
                    || p.split('/').any(|c| c == "..")
                {
                    return Err(format!("CDI spec: invalid device path {p:?}"));
                }
            }
            if perms.len() > 3 || !perms.chars().all(|c| "rwm".contains(c)) {
                return Err(format!("CDI spec: invalid device permissions {perms:?}"));
            }
            nodes
                .entry(path.to_owned())
                .or_insert_with(|| format!("device\t{host}\t{path}\t{perms}"));
        }
        for m in e["mounts"].as_array().into_iter().flatten() {
            let (host, ctr) = (
                m["hostPath"].as_str().unwrap_or(""),
                m["containerPath"].as_str().unwrap_or(""),
            );
            for p in [host, ctr] {
                if !p.starts_with('/')
                    || p.contains(['\t', '\n'])
                    || p.split('/').any(|c| c == "..")
                {
                    return Err(format!("CDI spec: invalid mount path {p:?}"));
                }
            }
            let ro = m["options"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|o| o.as_str() == Some("ro"));
            mounts.push((host.to_owned(), ctr.to_owned(), ro));
        }
        for h in e["hooks"].as_array().into_iter().flatten() {
            let args: Vec<&str> = h["args"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|a| a.as_str())
                .collect();
            if args.get(1) == Some(&"create-symlinks") {
                let mut it = args.iter();
                while let Some(a) = it.next() {
                    if *a == "--link"
                        && let Some((target, link)) = it.next().and_then(|l| l.split_once("::"))
                    {
                        links.push((target.to_owned(), link.to_owned()));
                    }
                }
            }
        }
        for g in e["additionalGids"].as_array().into_iter().flatten() {
            if let Some(g) = g.as_u64() {
                groups.insert(g);
            }
        }
        for v in e["env"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str())
        {
            let key = v.split('=').next().unwrap_or_default();
            // The toolkit's own hints about where it put the libraries.
            if !key.starts_with("NVIDIA_CTK_")
                && !env.iter().any(|x| x.split('=').next() == Some(key))
            {
                env.push(v.to_owned());
            }
        }
    }

    // The links hooks would make, as more binds of the file they point at.
    let mut bound: BTreeMap<String, (String, bool)> = BTreeMap::new();
    for (host, ctr, ro) in mounts {
        if Path::new(&host).exists() {
            bound.entry(ctr).or_insert((host, ro));
        }
    }
    for _ in 0..=links.len() {
        let mut changed = false;
        for (target, link) in &links {
            let at = if target.starts_with('/') {
                target.clone()
            } else {
                normalize(&format!(
                    "{}/{target}",
                    link.rsplit_once('/').map_or("", |d| d.0)
                ))
            };
            if let Some(b) = bound
                .get(&at)
                .cloned()
                .filter(|_| !bound.contains_key(link))
            {
                bound.insert(link.clone(), b);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut libs = BTreeSet::new();
    for (ctr, (host, ro)) in bound {
        let name = ctr.rsplit('/').next().unwrap_or("");
        let target = if relocate && is_lib(&ctr) {
            if !libs.insert(name.to_owned()) {
                continue;
            }
            format!("{LIB_DIR}/{name}")
        } else {
            ctr
        };
        out.mounts
            .push(format!("{host}\t{target}\t{}", if ro { "ro" } else { "" }));
    }
    if !libs.is_empty() {
        env.push(format!("LD_LIBRARY_PATH={LIB_DIR}"));
    }
    out.opts.extend(nodes.into_values());
    out.opts
        .extend(groups.into_iter().map(|g| format!("group\t{g}")));
    out.env = env;
    Ok(out)
}

/// A shared library of the driver (not an X server module): bound together under [`LIB_DIR`].
fn is_lib(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or("");
    name.starts_with("lib")
        && (name.ends_with(".so") || name.contains(".so."))
        && !path.contains("/xorg/")
}

/// `a/b/../c` as `a/c`.
fn normalize(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for c in path.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    format!("/{}", out.join("/"))
}

// How many times specs were read, for the test that a container asking for no GPU reads none.
#[cfg(test)]
thread_local!(static LOADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) });

/// Every spec in `dirs` that reads: `.json`, `.yaml` and `.yml` files, by name.
fn load(dirs: &[PathBuf]) -> Vec<Value> {
    #[cfg(test)]
    LOADS.with(|n| n.set(n.get() + 1));
    let mut files: Vec<PathBuf> = dirs
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e, "json" | "yaml" | "yml"))
        })
        .collect();
    files.sort();
    files
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .filter_map(|t| parse(&t).ok())
        .filter(|v| v["kind"].is_string())
        .collect()
}

/// A CDI spec: JSON, or the block-style YAML `nvidia-ctk` writes (maps,
/// lists, quoted and plain scalars; no anchors, tags or multi-line strings).
pub fn parse(text: &str) -> Result<Value, String> {
    if text.trim_start().starts_with('{') {
        return serde_json::from_str(text).map_err(|e| e.to_string());
    }
    let mut lines = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let t = raw.trim_end();
        let body = t.trim_start_matches(' ');
        if body.is_empty() || body.starts_with('#') || body == "---" {
            continue;
        }
        if body.starts_with('\t') {
            return Err(format!("line {}: tab indentation", n + 1));
        }
        lines.push((t.len() - body.len(), body.to_owned(), n + 1));
    }
    if lines.is_empty() {
        return Err("empty spec".into());
    }
    let (mut pos, indent) = (0, lines[0].0);
    let v = block(&mut lines, &mut pos, indent)?;
    if pos < lines.len() {
        return Err(format!("line {}: unexpected indentation", lines[pos].2));
    }
    Ok(v)
}

type Lines = Vec<(usize, String, usize)>;

fn is_item(t: &str) -> bool {
    t == "-" || t.starts_with("- ")
}

/// `key: rest` (or `key:`), when `t` is a mapping entry.
fn entry(t: &str) -> Option<(&str, &str)> {
    if t.starts_with(['"', '\'', '[', '{']) || is_item(t) {
        return None;
    }
    let b = t.as_bytes();
    (0..b.len())
        .find(|&i| b[i] == b':' && (i + 1 == b.len() || b[i + 1] == b' '))
        .map(|i| (&t[..i], t[i + 1..].trim()))
}

fn block(ls: &mut Lines, pos: &mut usize, indent: usize) -> Result<Value, String> {
    if is_item(&ls[*pos].1) {
        sequence(ls, pos, indent)
    } else {
        mapping(ls, pos, indent)
    }
}

fn sequence(ls: &mut Lines, pos: &mut usize, indent: usize) -> Result<Value, String> {
    let mut out = Vec::new();
    while *pos < ls.len() && ls[*pos].0 == indent && is_item(&ls[*pos].1) {
        let text = ls[*pos].1.clone();
        let rest = text[1..].trim_start();
        if rest.is_empty() {
            *pos += 1;
            if *pos < ls.len() && ls[*pos].0 > indent {
                let i = ls[*pos].0;
                out.push(block(ls, pos, i)?);
            } else {
                out.push(Value::Null);
            }
        } else if entry(rest).is_some() {
            // `- key: value`: the map goes on at the column of its first key.
            let at = indent + (text.len() - rest.len());
            ls[*pos].0 = at;
            ls[*pos].1 = rest.to_owned();
            out.push(mapping(ls, pos, at)?);
        } else {
            out.push(scalar(rest)?);
            *pos += 1;
        }
    }
    Ok(Value::Array(out))
}

fn mapping(ls: &mut Lines, pos: &mut usize, indent: usize) -> Result<Value, String> {
    let mut out = Map::new();
    while *pos < ls.len() && ls[*pos].0 == indent && !is_item(&ls[*pos].1) {
        let (text, n) = (ls[*pos].1.clone(), ls[*pos].2);
        let (key, rest) = entry(&text).ok_or_else(|| format!("line {n}: expected `key: value`"))?;
        let key = key.trim_matches('"').to_owned();
        *pos += 1;
        let value = if rest.is_empty() {
            // A list may sit at its key's own column.
            if *pos < ls.len()
                && (ls[*pos].0 > indent || (ls[*pos].0 == indent && is_item(&ls[*pos].1)))
            {
                let i = ls[*pos].0;
                block(ls, pos, i)?
            } else {
                Value::Null
            }
        } else {
            scalar(rest)?
        };
        out.insert(key, value);
    }
    if *pos < ls.len() && ls[*pos].0 > indent {
        return Err(format!("line {}: unexpected indentation", ls[*pos].2));
    }
    Ok(Value::Object(out))
}

fn scalar(s: &str) -> Result<Value, String> {
    let s = s.trim();
    Ok(match s {
        "[]" => Value::Array(Vec::new()),
        "{}" => Value::Object(Map::new()),
        "null" | "~" => Value::Null,
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ if s.starts_with('"') => {
            let inner = s
                .strip_prefix('"')
                .and_then(|t| t.strip_suffix('"'))
                .filter(|_| s.len() >= 2)
                .ok_or_else(|| format!("unterminated string {s}"))?;
            let mut out = String::new();
            let mut it = inner.chars();
            while let Some(c) = it.next() {
                if c == '\\' {
                    match it.next() {
                        Some('n') => out.push('\n'),
                        Some('t') => out.push('\t'),
                        Some(c) => out.push(c),
                        None => return Err(format!("bad escape in {s}")),
                    }
                } else {
                    out.push(c);
                }
            }
            Value::String(out)
        }
        _ if s.starts_with('\'') => Value::String(
            s.strip_prefix('\'')
                .and_then(|t| t.strip_suffix('\''))
                .filter(|_| s.len() >= 2)
                .ok_or_else(|| format!("unterminated string {s}"))?
                .replace("''", "'"),
        ),
        _ if s.starts_with(['[', '{']) => {
            return Err(format!("flow collections are not supported: {s}"));
        }
        _ => match s.parse::<i64>() {
            Ok(n) => Value::from(n),
            Err(_) => Value::String(s.to_owned()),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SPEC: &str = r#"---
cdiVersion: 0.7.0
kind: nvidia.com/gpu
devices:
    - name: "0"
      containerEdits:
        deviceNodes:
            - path: /dev/nvidia0
              major: 195
              permissions: rwm
        additionalGids:
            - 983
    - name: GPU-aaaa
      containerEdits:
        deviceNodes:
            - path: /dev/nvidia0
    - name: all
      containerEdits:
        deviceNodes:
            - path: /dev/nvidia0
containerEdits:
    env:
        - NVIDIA_CTK_LIBCUDA_DIR=/usr/lib
        - NVIDIA_VISIBLE_DEVICES=void
    deviceNodes:
        - path: /dev/nvidiactl
          fileMode: 438
    hooks:
        - hookName: createContainer
          path: /usr/bin/nvidia-cdi-hook
          args:
            - nvidia-cdi-hook
            - create-symlinks
            - --link
            - libcuda.so.1.2::/usr/lib/libcuda.so.1
            - --link
            - libcuda.so.1::/usr/lib/libcuda.so
          env:
            - NVIDIA_CTK_DEBUG=false
    mounts:
        - hostPath: HOSTLIB
          containerPath: /usr/lib/libcuda.so.1.2
          options:
            - ro
            - nosuid
            - nodev
            - bind
        - hostPath: /nonexistent/gone
          containerPath: /usr/bin/gone
        - hostPath: HOSTBIN
          containerPath: /usr/bin/nvidia-smi
          options:
            - ro
"#;

    #[test]
    fn yaml_subset() {
        let v = parse(SPEC).unwrap();
        assert_eq!(v["kind"], "nvidia.com/gpu");
        assert_eq!(v["devices"][0]["name"], "0");
        assert_eq!(
            v["devices"][0]["containerEdits"]["additionalGids"],
            json!([983])
        );
        assert_eq!(
            v["devices"][0]["containerEdits"]["deviceNodes"][0]["major"],
            195
        );
        assert_eq!(
            v["containerEdits"]["hooks"][0]["args"][3],
            "libcuda.so.1.2::/usr/lib/libcuda.so.1"
        );
        assert_eq!(
            v["containerEdits"]["mounts"][0]["options"],
            json!(["ro", "nosuid", "nodev", "bind"])
        );
        assert_eq!(
            parse("a: []\nb: \"\"\nc:\n- x\n- 'y'\n").unwrap(),
            json!({"a": [], "b": "", "c": ["x", "y"]})
        );
        assert!(parse("a: 1\n  b: 2\n").is_err());
        assert_eq!(parse("{\"kind\": \"a.b/c\"}").unwrap()["kind"], "a.b/c");
    }

    fn dir_with_spec() -> (PathBuf, String, String) {
        let d = std::env::temp_dir().join(format!(
            "kiri-cdi-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        let (lib, bin) = (d.join("libcuda.so.1.2"), d.join("nvidia-smi"));
        std::fs::write(&lib, "x").unwrap();
        std::fs::write(&bin, "x").unwrap();
        let (lib, bin) = (
            lib.to_str().unwrap().to_owned(),
            bin.to_str().unwrap().to_owned(),
        );
        std::fs::write(
            d.join("nvidia.yaml"),
            SPEC.replace("HOSTLIB", &lib).replace("HOSTBIN", &bin),
        )
        .unwrap();
        (d, lib, bin)
    }

    #[test]
    fn gpus_all_from_the_spec() {
        let (d, lib, bin) = dir_with_spec();
        let opts =
            json!({"DeviceRequests": [{"Driver": "", "Count": -1, "Capabilities": [["gpu"]]}]});
        let e = edits_in(&opts, std::slice::from_ref(&d)).unwrap();
        assert_eq!(
            e.opts,
            [
                "device\t/dev/nvidia0\t/dev/nvidia0\trwm",
                "device\t/dev/nvidiactl\t/dev/nvidiactl\trwm"
            ]
        );
        // The versioned file and both links are libraries bound under one directory; the missing mount is left out.
        assert_eq!(
            e.mounts,
            [
                format!("{bin}\t/usr/bin/nvidia-smi\tro"),
                format!("{lib}\t{LIB_DIR}/libcuda.so\tro"),
                format!("{lib}\t{LIB_DIR}/libcuda.so.1\tro"),
                format!("{lib}\t{LIB_DIR}/libcuda.so.1.2\tro"),
            ]
        );
        assert_eq!(
            e.env,
            [
                "NVIDIA_VISIBLE_DEVICES=void",
                format!("LD_LIBRARY_PATH={LIB_DIR}").as_str()
            ]
        );
        let mut env = vec![
            "NVIDIA_VISIBLE_DEVICES=all".to_owned(),
            "LD_LIBRARY_PATH=/opt/x".to_owned(),
        ];
        e.apply_env(&mut env);
        assert_eq!(
            env,
            [
                "NVIDIA_VISIBLE_DEVICES=all",
                &format!("LD_LIBRARY_PATH={LIB_DIR}:/opt/x")
            ]
        );
        e.apply_env(&mut env);
        assert_eq!(env[1], format!("LD_LIBRARY_PATH={LIB_DIR}:/opt/x"));

        // By count, by id and by CDI name; the spec's common edits count once.
        for opts in [
            json!({"DeviceRequests": [{"Driver": "nvidia", "Count": 1, "Capabilities": [["gpu"]]}]}),
            json!({"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": ["0"], "Capabilities": [["gpu"]]}]}),
            json!({"DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=GPU-aaaa"]}]}),
            json!({"Devices": [{"PathOnHost": "nvidia.com/gpu=0", "PathInContainer": "", "CgroupPermissions": ""}]}),
        ] {
            let e = edits_in(&opts, std::slice::from_ref(&d)).unwrap();
            assert_eq!(
                e.opts
                    .iter()
                    .filter(|l| l.starts_with("device\t/dev/nvidiactl"))
                    .count(),
                1,
                "{opts}"
            );
        }
        assert_eq!(edits_in(&json!({"DeviceRequests": [{"Driver": "nvidia", "Count": 1, "Capabilities": [["gpu"]]}]}), std::slice::from_ref(&d)).unwrap().opts[2], "group\t983");

        let e = edits_in(&json!({"DeviceRequests": [{"Driver": "nvidia", "Count": 2, "Capabilities": [["gpu"]]}]}), std::slice::from_ref(&d)).unwrap_err();
        assert!(
            e.contains("requested 2 nvidia.com/gpu devices but only 1 found"),
            "{e}"
        );
        let e = edits_in(&json!({"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": ["7"], "Capabilities": [["gpu"]]}]}), std::slice::from_ref(&d)).unwrap_err();
        assert_eq!(e, "unresolvable CDI devices nvidia.com/gpu=7");
        let e = edits_in(&json!({"DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]}), &[PathBuf::from("/nonexistent/cdi")]).unwrap_err();
        assert!(e.contains("no CDI spec for nvidia.com/gpu"), "{e}");
        let e = edits_in(
            &json!({"DeviceRequests": [{"Driver": "foo", "Count": -1, "Capabilities": [["gpu"]]}]}),
            std::slice::from_ref(&d),
        )
        .unwrap_err();
        assert!(
            e.starts_with("could not select device driver \"foo\""),
            "{e}"
        );
        // No AMD spec on this host: the message says how to make one, and that --device /dev/dri works.
        let e = edits_in(&json!({"DeviceRequests": [{"Driver": "amd", "Count": -1, "Capabilities": [["gpu"]]}]}), std::slice::from_ref(&d)).unwrap_err();
        assert!(e.contains("no CDI spec for amd.com/gpu") && e.contains("amd-ctk cdi generate") && e.contains("--device /dev/dri"), "{e}");
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn amd_and_intel_from_their_specs() {
        let d = std::env::temp_dir().join(format!("kiri-cdi-amd-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("libamd.so.1"), "x").unwrap();
        for (kind, driver, node) in [("amd.com/gpu", "amd", "/dev/kfd"), ("intel.com/gpu", "intel", "/dev/dri/renderD128")] {
            let spec = json!({"cdiVersion": "0.5.0", "kind": kind,
                "devices": [{"name": "0", "containerEdits": {"deviceNodes": [{"path": node, "hostPath": "/dev/null"}], "additionalGids": [44]}}],
                "containerEdits": {"mounts": [{"hostPath": d.join("libamd.so.1").to_str().unwrap(), "containerPath": "/opt/vendor/libamd.so.1", "options": ["ro"]}]}});
            let dir = d.join(driver);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("gpu.json"), spec.to_string()).unwrap();
            // `--gpus driver=<vendor> count=all`, and the CDI name, both give the vendor's nodes; its library stays where its spec puts it.
            for opts in [
                json!({"DeviceRequests": [{"Driver": driver, "Count": -1, "Capabilities": [["gpu"]]}]}),
                json!({"Devices": [{"PathOnHost": format!("{kind}=0")}]}),
            ] {
                let e = edits_in(&opts, std::slice::from_ref(&dir)).unwrap();
                assert_eq!(e.opts, [format!("device\t/dev/null\t{node}\trwm"), "group\t44".to_owned()], "{opts}");
                assert_eq!(e.mounts, [format!("{}\t/opt/vendor/libamd.so.1\tro", d.join("libamd.so.1").display())]);
                assert!(e.env.is_empty(), "no NVIDIA library path for {kind}");
            }
        }
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn no_gpu_asked_for_reads_nothing() {
        let before = LOADS.with(|n| n.get());
        for opts in [json!({}), json!({"DeviceRequests": []}), json!({"Devices": [{"PathOnHost": "/dev/fuse"}]}), Value::Null] {
            assert_eq!(edits(&opts).unwrap(), Edits::default());
        }
        assert!(crate::daemon::hostopts::check_device_requests(&json!({"Devices": [{"PathOnHost": "/dev/fuse"}]})).is_ok());
        let mut env = vec!["A=1".to_owned()];
        Edits::default().apply_env(&mut env);
        assert_eq!(env, ["A=1"]);
        assert_eq!(LOADS.with(|n| n.get()), before);
    }

    #[test]
    fn what_asks_for_devices() {
        assert!(!wanted(&json!({})));
        assert!(!wanted(&json!({"DeviceRequests": []})));
        assert!(wanted(&json!({"DeviceRequests": [{"Count": -1}]})));
        assert!(wanted(
            &json!({"Devices": [{"PathOnHost": "nvidia.com/gpu=all"}]})
        ));
        assert!(!wanted(&json!({"Devices": [{"PathOnHost": "/dev/fuse"}]})));
        assert!(
            is_qualified("nvidia.com/gpu=0")
                && !is_qualified("/dev/fuse")
                && !is_qualified("gpu=0")
                && !is_qualified("nvidia.com/gpu=")
        );
    }
}
