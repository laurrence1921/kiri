//! GPUs for a container on a Linux host, from a CDI spec.
//!
//! `docker run --gpus all`, Compose's `deploy.resources.reservations.devices`
//! and `--device nvidia.com/gpu=0` all name a device the host's Container
//! Device Interface spec describes (`nvidia-ctk cdi generate` writes one to
//! `/etc/cdi/nvidia.yaml`; AMD and Intel have their own kinds, see
//! [`VENDORS`]). Kiri reads that spec at every start of a container that asks
//! for a GPU, so a driver update is picked up, and turns its `containerEdits`
//! into what the guest already does: device nodes (with their cgroup rules),
//! bind mounts, env and groups. The agent runs on the host's own kernel and
//! filesystem there, so the host's paths can be bound as they are.
//!
//! Hooks are not run, but the one the NVIDIA spec needs for CUDA is done in
//! place of it: `create-symlinks` makes `libcuda.so.1` from `libcuda.so.<v>`.
//! The driver's libraries are bound under [`LIB_DIR`] (where the `nvidia/cuda`
//! images already look) with those names, and `LD_LIBRARY_PATH` names it, so
//! no `ld.so.cache` is needed.
//!
//! What differs between vendors is data in [`VENDORS`]; the code below does
//! not name one.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Where the host's CDI specs are looked for, as the CDI spec says.
pub const SPEC_DIRS: [&str; 2] = ["/etc/cdi", "/var/run/cdi"];

/// Where the libraries of a vendor with [`Vendor::shared_lib_dir`] are bound in the container.
const LIB_DIR: &str = "/usr/local/nvidia/lib64";

/// What a GPU vendor needs that another does not. A new vendor is one more row of [`VENDORS`].
struct Vendor {
    /// `--gpus driver=<name>`: the names that mean this vendor. NVIDIA is also the default driver (`""`).
    drivers: &'static [&'static str],
    /// The CDI kind of its devices.
    kind: &'static str,
    /// What to tell the user when the host has no spec for `kind`.
    hint: &'static str,
    /// Its libraries are bound together under [`LIB_DIR`], with the links its
    /// spec's hooks would make, and `LD_LIBRARY_PATH` names that directory.
    /// Else they stay where the spec puts them.
    shared_lib_dir: bool,
    /// Env keys with this prefix are its toolkit's own notes, not for the container.
    private_env: &'static str,
}

const VENDORS: [Vendor; 3] = [
    Vendor {
        drivers: &["", "nvidia"],
        kind: "nvidia.com/gpu",
        hint: "for NVIDIA, install nvidia-container-toolkit and run: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml",
        shared_lib_dir: true,
        private_env: "NVIDIA_CTK_",
    },
    Vendor {
        drivers: &["amd"],
        kind: "amd.com/gpu",
        hint: "for AMD, install the AMD container toolkit and run: sudo amd-ctk cdi generate --output=/etc/cdi/amd.json; or pass --device /dev/kfd --device /dev/dri",
        shared_lib_dir: false,
        private_env: "",
    },
    Vendor {
        drivers: &["intel"],
        kind: "intel.com/gpu",
        hint: "for Intel, pass --device /dev/dri, or add a CDI spec for intel.com/gpu to /etc/cdi",
        shared_lib_dir: false,
        private_env: "",
    },
];

/// What to tell the user about a kind no [`Vendor`] knows.
const OTHER_HINT: &str = "add a CDI spec for it to /etc/cdi";

fn vendor_of_driver(driver: &str) -> Option<&'static Vendor> {
    VENDORS.iter().find(|v| v.drivers.contains(&driver))
}

fn vendor_of_kind(kind: &str) -> Option<&'static Vendor> {
    VENDORS.iter().find(|v| v.kind == kind)
}

/// Docker's error for a device request Kiri cannot give.
fn select_error(why: impl std::fmt::Display) -> String {
    format!("could not select device driver \"\" with capabilities: [[gpu]]: {why}")
}

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
        const LD: &str = "LD_LIBRARY_PATH=";
        for e in self.env.iter().filter(|e| !e.starts_with(LD)) {
            let key = e.split('=').next().unwrap_or_default();
            if !env.iter().any(|x| x.split('=').next() == Some(key)) {
                env.push(e.clone());
            }
        }
        if let Some(path) = self.env.iter().find_map(|e| e.strip_prefix(LD)) {
            match env.iter_mut().find(|x| x.starts_with(LD)) {
                Some(x) if x[LD.len()..].split(':').any(|d| d == path) => {}
                Some(x) => *x = format!("{LD}{path}:{}", &x[LD.len()..]),
                None => env.push(format!("{LD}{path}")),
            }
        }
    }
}

/// `vendor.example/class=name`: how CDI names a device.
pub fn is_qualified(name: &str) -> bool {
    match name.split_once('=') {
        Some((kind, device)) => {
            !device.is_empty()
                && kind.split_once('/').is_some_and(|(vendor, class)| vendor.contains('.') && !class.is_empty() && !vendor.is_empty())
        }
        None => false,
    }
}

/// The elements of the array `v[key]`, if it is one.
fn items<'a>(v: &'a Value, key: &str) -> impl Iterator<Item = &'a Value> {
    v[key].as_array().into_iter().flatten()
}

/// The strings of the array `v[key]`.
fn strs<'a>(v: &'a Value, key: &str) -> impl Iterator<Item = &'a str> {
    items(v, key).filter_map(Value::as_str)
}

/// Which devices of a kind a container asked for.
#[derive(Debug, PartialEq)]
enum Pick {
    All,
    First(usize),
    Name(String),
}

/// `kind=name` as (kind, which).
fn split(qualified: &str) -> (String, Pick) {
    let (kind, name) = qualified.split_once('=').unwrap_or((qualified, ""));
    (kind.to_owned(), if name == "all" { Pick::All } else { Pick::Name(name.to_owned()) })
}

/// What one DeviceRequest asks for: with a vendor's driver, its devices by
/// number, id or all of them; with the driver `cdi`, devices by CDI name.
fn request(r: &Value) -> Result<Vec<(String, Pick)>, String> {
    let driver = r["Driver"].as_str().unwrap_or("");
    let caps: Vec<&str> = items(r, "Capabilities").flat_map(|c| c.as_array().into_iter().flatten()).filter_map(Value::as_str).collect();
    let ids: Vec<&str> = strs(r, "DeviceIDs").collect();
    if driver == "cdi" {
        return ids
            .into_iter()
            .map(|id| if is_qualified(id) { Ok(split(id)) } else { Err(format!("could not select device driver \"cdi\" with capabilities: [{}]: {id:?} is not a CDI device name", caps.join(" "))) })
            .collect();
    }
    let vendor = vendor_of_driver(driver).filter(|_| caps.contains(&"gpu") || caps.contains(&"nvidia")).ok_or_else(|| {
        format!(
            "could not select device driver {driver:?} with capabilities: [{}]: Kiri supports GPUs through CDI specs (driver \"nvidia\" or \"\", \"amd\" or \"intel\", with capability \"gpu\", or driver \"cdi\")",
            caps.iter().map(|c| format!("[{c}]")).collect::<Vec<_>>().join(" ")
        )
    })?;
    let count = r["Count"].as_i64().unwrap_or(0);
    Ok(if !ids.is_empty() {
        ids.into_iter().map(|id| if is_qualified(id) { split(id) } else { (vendor.kind.to_owned(), Pick::Name(id.to_owned())) }).collect()
    } else if count < 0 {
        vec![(vendor.kind.to_owned(), Pick::All)]
    } else if count > 0 {
        vec![(vendor.kind.to_owned(), Pick::First(count as usize))]
    } else {
        Vec::new()
    })
}

/// The devices HostConfig asks for, as (CDI kind, which): `--gpus` and
/// Compose's device reservations arrive as DeviceRequests, `--device` with a
/// CDI name as a `Devices` entry whose PathOnHost is that name.
fn requested(opts: &Value) -> Result<Vec<(String, Pick)>, String> {
    let mut out = Vec::new();
    for r in items(opts, "DeviceRequests") {
        out.extend(request(r)?);
    }
    out.extend(items(opts, "Devices").filter_map(|d| d["PathOnHost"].as_str()).filter(|n| is_qualified(n)).map(split));
    Ok(out)
}

/// Whether `opts` (a container's kept HostConfig) asks for CDI devices.
pub fn wanted(opts: &Value) -> bool {
    items(opts, "DeviceRequests").next().is_some() || items(opts, "Devices").any(|d| d["PathOnHost"].as_str().is_some_and(is_qualified))
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

fn device_name(d: &Value) -> String {
    match &d["name"] {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

/// A device named by its number: the ones `all` and a count choose among.
fn is_indexed(d: &Value) -> bool {
    device_name(d).bytes().all(|b| b.is_ascii_digit())
}

/// The devices of `kind` that `pick` names, as (index of its spec, device).
fn select<'a>(specs: &'a [Value], kind: &str, pick: &Pick, dirs: &[PathBuf]) -> Result<Vec<(usize, &'a Value)>, String> {
    let of_kind: Vec<usize> = specs.iter().enumerate().filter(|(_, s)| s["kind"].as_str() == Some(kind)).map(|(i, _)| i).collect();
    if of_kind.is_empty() {
        let hint = vendor_of_kind(kind).map_or(OTHER_HINT, |v| v.hint);
        let places = dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(" or ");
        return Err(select_error(format!("no CDI spec for {kind} in {places} ({hint})")));
    }
    let devices = || of_kind.iter().flat_map(|&i| items(&specs[i], "devices").map(move |d| (i, d)));
    match pick {
        Pick::Name(n) => devices().find(|(_, d)| device_name(d) == *n).map(|d| vec![d]).ok_or_else(|| format!("unresolvable CDI devices {kind}={n}")),
        // The spec's own "all" if it has one, else every device with an index for a name.
        Pick::All => Ok(devices().find(|(_, d)| device_name(d) == "all").map(|d| vec![d]).unwrap_or_else(|| devices().filter(|(_, d)| is_indexed(d)).collect())),
        Pick::First(n) => {
            let all: Vec<_> = devices().filter(|(_, d)| is_indexed(d)).collect();
            if all.len() < *n {
                return Err(select_error(format!("requested {n} {kind} devices but only {} found", all.len())));
            }
            Ok(all.into_iter().take(*n).collect())
        }
    }
}

fn edits_in(opts: &Value, dirs: &[PathBuf]) -> Result<Edits, String> {
    let requests = requested(opts)?;
    let specs = load(dirs);
    let mut picked: Vec<(usize, &Value)> = Vec::new();
    for (kind, pick) in &requests {
        picked.extend(select(&specs, kind, pick, dirs)?);
    }
    // A spec's own edits once, however many of its devices are picked, then each device's.
    let spec_ids: BTreeSet<usize> = picked.iter().map(|(i, _)| *i).collect();
    let all_edits: Vec<&Value> = spec_ids.iter().map(|&i| &specs[i]["containerEdits"]).chain(picked.iter().map(|(_, d)| &d["containerEdits"])).collect();
    let relocate = requests.iter().any(|(k, _)| vendor_of_kind(k).is_some_and(|v| v.shared_lib_dir));
    apply(&all_edits, relocate)
}

/// A path the guest takes in a tab-separated line: absolute, one line, no `..`.
fn check_path(p: &str, what: &str) -> Result<(), String> {
    if !p.starts_with('/') || p.contains(['\t', '\n']) || p.split('/').any(|c| c == "..") {
        return Err(format!("CDI spec: invalid {what} path {p:?}"));
    }
    Ok(())
}

/// The guest's `device` line for a `deviceNodes` entry, and the path it makes in the container.
fn device_line(n: &Value) -> Result<(String, String), String> {
    let path = n["path"].as_str().unwrap_or("");
    let host = n["hostPath"].as_str().filter(|h| !h.is_empty()).unwrap_or(path);
    let perms = n["permissions"].as_str().filter(|p| !p.is_empty()).unwrap_or("rwm");
    check_path(host, "device")?;
    check_path(path, "device")?;
    if perms.len() > 3 || !perms.chars().all(|c| "rwm".contains(c)) {
        return Err(format!("CDI spec: invalid device permissions {perms:?}"));
    }
    Ok((path.to_owned(), format!("device\t{host}\t{path}\t{perms}")))
}

/// A `mounts` entry as (host path, container path, read-only).
fn mount_of(m: &Value) -> Result<(String, String, bool), String> {
    let (host, ctr) = (m["hostPath"].as_str().unwrap_or(""), m["containerPath"].as_str().unwrap_or(""));
    check_path(host, "mount")?;
    check_path(ctr, "mount")?;
    Ok((host.to_owned(), ctr.to_owned(), strs(m, "options").any(|o| o == "ro")))
}

/// The links a `create-symlinks` hook makes, as (target, link): `--link target::link`.
fn symlinks(hook: &Value) -> Vec<(String, String)> {
    let args: Vec<&str> = strs(hook, "args").collect();
    if args.get(1) != Some(&"create-symlinks") {
        return Vec::new();
    }
    args.windows(2).filter(|w| w[0] == "--link").filter_map(|w| w[1].split_once("::")).map(|(t, l)| (t.to_owned(), l.to_owned())).collect()
}

/// An env key a vendor's toolkit keeps to itself.
fn is_private_env(key: &str) -> bool {
    VENDORS.iter().any(|v| !v.private_env.is_empty() && key.starts_with(v.private_env))
}

/// What the `containerEdits` say, checked and sorted by kind.
#[derive(Default)]
struct Collected {
    /// Container path → the guest's `device` line.
    nodes: BTreeMap<String, String>,
    /// (host path, container path, read-only)
    mounts: Vec<(String, String, bool)>,
    /// (target, link) of the `create-symlinks` hooks.
    links: Vec<(String, String)>,
    groups: BTreeSet<u64>,
    env: Vec<String>,
}

fn collect(edits: &[&Value]) -> Result<Collected, String> {
    let mut c = Collected::default();
    for e in edits {
        for n in items(e, "deviceNodes") {
            let (path, line) = device_line(n)?;
            c.nodes.entry(path).or_insert(line);
        }
        for m in items(e, "mounts") {
            c.mounts.push(mount_of(m)?);
        }
        for h in items(e, "hooks") {
            c.links.extend(symlinks(h));
        }
        c.groups.extend(items(e, "additionalGids").filter_map(Value::as_u64));
        for v in strs(e, "env") {
            let key = v.split('=').next().unwrap_or_default();
            if !is_private_env(key) && !c.env.iter().any(|x| x.split('=').next() == Some(key)) {
                c.env.push(v.to_owned());
            }
        }
    }
    Ok(c)
}

/// The links hooks would make, as more binds of the file they point at: each
/// `link` in `bound` with the host file of its `target`, until none is new.
fn bind_links(bound: &mut BTreeMap<String, (String, bool)>, links: &[(String, String)]) {
    for _ in 0..=links.len() {
        let mut changed = false;
        for (target, link) in links {
            let at = if target.starts_with('/') { target.clone() } else { normalize(&format!("{}/{target}", link.rsplit_once('/').map_or("", |d| d.0))) };
            if !bound.contains_key(link)
                && let Some(b) = bound.get(&at).cloned()
            {
                bound.insert(link.clone(), b);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// The guest's mount lines for `bound` (container path → host path, read-only),
/// and whether libraries went to [`LIB_DIR`]: with `relocate`, each library
/// under its own name, once.
fn mount_lines(bound: BTreeMap<String, (String, bool)>, relocate: bool) -> (Vec<String>, bool) {
    let mut libs = BTreeSet::new();
    let mut lines = Vec::new();
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
        lines.push(format!("{host}\t{target}\t{}", if ro { "ro" } else { "" }));
    }
    (lines, !libs.is_empty())
}

fn apply(edits: &[&Value], relocate: bool) -> Result<Edits, String> {
    let c = collect(edits)?;
    // A mount whose host file is not there is left out: the specs list optional files.
    let mut bound: BTreeMap<String, (String, bool)> = BTreeMap::new();
    for (host, ctr, ro) in c.mounts {
        if Path::new(&host).exists() {
            bound.entry(ctr).or_insert((host, ro));
        }
    }
    bind_links(&mut bound, &c.links);
    let (mounts, libs) = mount_lines(bound, relocate);
    let mut env = c.env;
    if libs {
        env.push(format!("LD_LIBRARY_PATH={LIB_DIR}"));
    }
    let opts = c.nodes.into_values().chain(c.groups.into_iter().map(|g| format!("group\t{g}"))).collect();
    Ok(Edits { opts, mounts, env })
}

/// A shared library of the driver (not an X server module): bound together under [`LIB_DIR`].
fn is_lib(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or("");
    name.starts_with("lib") && (name.ends_with(".so") || name.contains(".so.")) && !path.contains("/xorg/")
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
