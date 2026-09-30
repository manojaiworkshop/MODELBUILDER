//! Any file -> clean UTF-8 text on disk.
//!
//! Supported: plain text / code / CSV / JSON / logs in any common encoding (UTF-8, UTF-16, Latin-1…),
//! PDF (text layer), Word `.docx`, PowerPoint `.pptx`, Excel `.xlsx/.xlsm/.xlsb/.xls`, OpenDocument
//! `.odt/.odp/.ods`, `.epub`, HTML, RTF, `.zip` archives (recursively), and `.gz`.
//! Not supported (reported per file, never fatal): images/audio/video (no OCR / transcription),
//! encrypted files, and the legacy binary `.doc` / `.ppt` formats.
//!
//! Output is streamed through a [`Sink`] that normalises control characters and blank lines and
//! enforces a size cap, so a hostile or enormous file cannot exhaust the disk.

use serde::Serialize;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const HEAD: usize = 64 * 1024;
const MAX_ZIP_ENTRIES: usize = 50_000;
const MAX_DEPTH: u32 = 3;
/// Formats that must be read whole into memory get a size ceiling.
const MAX_WHOLE_FILE: u64 = 512 * 1024 * 1024;
/// Page marks kept per document.
const MAX_PAGES: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Text,
    Pdf,
    Docx,
    Pptx,
    Xlsx,
    Odf,
    Epub,
    Html,
    Rtf,
    Zip,
    Gzip,
    Legacy,
    Unsupported,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Pdf => "PDF",
            Kind::Docx => "Word",
            Kind::Pptx => "PowerPoint",
            Kind::Xlsx => "Excel",
            Kind::Odf => "OpenDocument",
            Kind::Epub => "EPUB",
            Kind::Html => "HTML",
            Kind::Rtf => "RTF",
            Kind::Zip => "ZIP",
            Kind::Gzip => "gzip",
            Kind::Legacy => "legacy Office",
            Kind::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug)]
pub enum Ex {
    /// The output cap was reached; what was written so far is kept.
    Cap,
    Msg(String),
}

impl From<io::Error> for Ex {
    fn from(e: io::Error) -> Self {
        Ex::Msg(format!("I/O error: {e}"))
    }
}
type R<T> = Result<T, Ex>;
fn msg<T>(s: impl Into<String>) -> R<T> {
    Err(Ex::Msg(s.into()))
}

// ------------------------------------------------------------------ sink

/// Destination for extracted text. Cleans as it writes.
pub struct Sink {
    w: BufWriter<File>,
    pub bytes: u64,
    cap: u64,
    newlines: u8,
    pub truncated: bool,
    /// Byte offset in the output where each page / slide starts (only for formats that have them).
    pub pages: Vec<u64>,
}

impl Sink {
    pub fn create(path: &Path, cap: u64) -> io::Result<Sink> {
        Ok(Sink { w: BufWriter::with_capacity(1 << 20, File::create(path)?), bytes: 0, cap, newlines: 0, truncated: false, pages: Vec::new() })
    }

    pub fn push(&mut self, s: &str) -> R<()> {
        let mut buf = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '\r' | '\u{feff}' => {}
                '\n' => {
                    if self.newlines < 2 {
                        self.newlines += 1;
                        buf.push('\n');
                    }
                }
                c if c.is_control() && c != '\t' => {}
                c => {
                    self.newlines = 0;
                    buf.push(c);
                }
            }
        }
        let room = self.cap.saturating_sub(self.bytes) as usize;
        if buf.len() > room {
            let mut cut = room;
            while !buf.is_char_boundary(cut) {
                cut -= 1;
            }
            self.w.write_all(buf[..cut].as_bytes())?;
            self.bytes += cut as u64;
            self.truncated = true;
            return Err(Ex::Cap);
        }
        self.w.write_all(buf.as_bytes())?;
        self.bytes += buf.len() as u64;
        Ok(())
    }

    /// Records that a new page / slide starts at the current output position.
    pub fn mark_page(&mut self) {
        if self.pages.len() < MAX_PAGES {
            self.pages.push(self.bytes);
        }
    }

    /// Paragraph break (used between documents / slides / sheets).
    pub fn break_paragraph(&mut self) -> R<()> {
        self.push("\n\n")
    }

    pub fn finish(mut self) -> io::Result<u64> {
        self.w.flush()?;
        Ok(self.bytes)
    }
}

// ------------------------------------------------------------- detection

fn ext_of(name: &str) -> String {
    Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

const MEDIA_EXT: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "webp", "tif", "tiff", "ico", "heic", "svgz", "mp3", "wav", "flac", "ogg", "m4a", "aac", "mp4", "mkv",
    "mov", "avi", "webm", "exe", "dll", "so", "bin", "iso", "dmg", "class", "pyc", "o", "a", "parquet", "sqlite", "db", "pkl", "npy", "pt", "onnx",
];

fn read_head(path: &Path) -> io::Result<Vec<u8>> {
    let mut v = Vec::with_capacity(HEAD);
    File::open(path)?.take(HEAD as u64).read_to_end(&mut v)?;
    Ok(v)
}

/// Decide what a file is from its content (extension only breaks ties). Returns the kind and,
/// for zip-based and spreadsheet formats, the extension libraries need to open it.
pub fn detect(name: &str, path: &Path) -> Result<(Kind, &'static str), String> {
    let head = read_head(path).map_err(|e| format!("cannot read file: {e}"))?;
    let ext = ext_of(name);

    if head.starts_with(b"%PDF-") {
        return Ok((Kind::Pdf, "pdf"));
    }
    if head.starts_with(&[0x1f, 0x8b]) {
        return Ok((Kind::Gzip, "gz"));
    }
    if head.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        // OLE2 container: .xls is readable, .doc/.ppt are not
        return Ok(match ext.as_str() {
            "xls" | "xlt" => (Kind::Xlsx, "xls"),
            _ => (Kind::Legacy, "bin"),
        });
    }
    if head.starts_with(b"PK\x03\x04") || head.starts_with(b"PK\x05\x06") {
        return detect_zip(path);
    }
    if head.starts_with(b"{\\rtf") {
        return Ok((Kind::Rtf, "rtf"));
    }
    if MEDIA_EXT.contains(&ext.as_str()) || is_media_magic(&head) {
        return Ok((Kind::Unsupported, "bin"));
    }
    let lower = String::from_utf8_lossy(&head[..head.len().min(2048)]).to_ascii_lowercase();
    if matches!(ext.as_str(), "html" | "htm" | "xhtml") || lower.contains("<!doctype html") || lower.contains("<html") {
        return Ok((Kind::Html, "html"));
    }
    // text sniff: UTF-16 BOM, or no NUL bytes
    if head.starts_with(&[0xFF, 0xFE]) || head.starts_with(&[0xFE, 0xFF]) || !head.contains(&0) {
        return Ok((Kind::Text, "txt"));
    }
    Ok((Kind::Unsupported, "bin"))
}

fn is_media_magic(h: &[u8]) -> bool {
    h.starts_with(&[0x89, b'P', b'N', b'G']) || h.starts_with(&[0xFF, 0xD8, 0xFF]) || h.starts_with(b"GIF8") || h.starts_with(b"ID3")
        || h.starts_with(b"OggS") || h.starts_with(b"fLaC") || h.starts_with(&[0x7F, b'E', b'L', b'F']) || h.starts_with(b"MZ")
        || (h.len() > 12 && &h[4..8] == b"ftyp") || (h.len() > 12 && &h[0..4] == b"RIFF")
}

fn detect_zip(path: &Path) -> Result<(Kind, &'static str), String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    let mut z = zip::ZipArchive::new(BufReader::new(f)).map_err(|e| format!("corrupt zip container: {e}"))?;
    let has = |z: &zip::ZipArchive<_>, n: &str| z.file_names().any(|x| x == n);
    if has(&z, "word/document.xml") {
        return Ok((Kind::Docx, "docx"));
    }
    if has(&z, "ppt/presentation.xml") {
        return Ok((Kind::Pptx, "pptx"));
    }
    if has(&z, "xl/workbook.xml") {
        return Ok((Kind::Xlsx, "xlsx"));
    }
    if has(&z, "xl/workbook.bin") {
        return Ok((Kind::Xlsx, "xlsb"));
    }
    if has(&z, "mimetype") {
        let mut mt = String::new();
        if let Ok(mut e) = z.by_name("mimetype") {
            let _ = e.by_ref().take(200).read_to_string(&mut mt);
        }
        if mt.starts_with("application/epub+zip") {
            return Ok((Kind::Epub, "epub"));
        }
        if mt.contains("opendocument.spreadsheet") {
            return Ok((Kind::Xlsx, "ods"));
        }
        if mt.starts_with("application/vnd.oasis.opendocument") {
            return Ok((Kind::Odf, "odt"));
        }
    }
    Ok((Kind::Zip, "zip"))
}

// ------------------------------------------------------------- entry point

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub kind: Kind,
    pub bytes: u64,
    pub truncated: bool,
    pub note: Option<String>,
    /// Start offset of each page / slide in the extracted text (empty when the format has none).
    pub pages: Vec<u64>,
}

struct Ctx<'a> {
    sink: &'a mut Sink,
    tmp: &'a Path,
    files: usize,
    skipped: Vec<String>,
    serial: u32,
}

/// Extract `src` (named `name` by the user) into a UTF-8 text file at `out`.
/// `tmp` is a scratch directory for nested extraction. `cap` limits the output size.
pub fn extract_to(src: &Path, name: &str, out: &Path, tmp: &Path, cap: u64) -> Result<Outcome, String> {
    fs::create_dir_all(tmp).map_err(|e| e.to_string())?;
    let (kind, _) = detect(name, src)?;
    let mut sink = Sink::create(out, cap).map_err(|e| e.to_string())?;
    let mut cx = Ctx { sink: &mut sink, tmp, files: 0, skipped: vec![], serial: 0 };
    let res = extract_path(&mut cx, src, name, 0);
    let (files, skipped) = (cx.files, std::mem::take(&mut cx.skipped));
    let truncated = sink.truncated;
    let pages = std::mem::take(&mut sink.pages);
    let bytes = sink.finish().map_err(|e| e.to_string())?;
    match res {
        Ok(()) | Err(Ex::Cap) => {}
        Err(Ex::Msg(m)) => return Err(m),
    }
    if bytes == 0 {
        return Err(match kind {
            Kind::Pdf => "No text found. This PDF looks like scanned images — OCR is not supported.".into(),
            Kind::Zip => "The archive contains no readable text files.".into(),
            _ => "No readable text found in this file.".into(),
        });
    }
    let mut notes = Vec::new();
    if matches!(kind, Kind::Zip) || files > 1 {
        notes.push(format!("{files} files read"));
    }
    if !skipped.is_empty() {
        let shown: Vec<_> = skipped.iter().take(3).cloned().collect();
        notes.push(format!("{} skipped ({}{})", skipped.len(), shown.join("; "), if skipped.len() > 3 { "; …" } else { "" }));
    }
    if truncated {
        notes.push("truncated at the size limit".into());
    }
    Ok(Outcome { kind, bytes, truncated, note: (!notes.is_empty()).then(|| notes.join(" · ")), pages })
}

fn extract_path(cx: &mut Ctx, path: &Path, name: &str, depth: u32) -> R<()> {
    if depth > MAX_DEPTH {
        return msg("archives are nested too deeply");
    }
    let (kind, ext) = detect(name, path).map_err(Ex::Msg)?;
    let size = fs::metadata(path)?.len();
    let whole = matches!(kind, Kind::Pdf | Kind::Xlsx | Kind::Html | Kind::Rtf);
    if whole && size > MAX_WHOLE_FILE {
        return msg(format!("{} files above {} MB can't be processed (they must be loaded whole into memory)", kind.label(), MAX_WHOLE_FILE >> 20));
    }
    match kind {
        Kind::Text => text_file(path, cx.sink)?,
        Kind::Html => html_file(path, cx.sink)?,
        Kind::Rtf => rtf_file(path, cx.sink)?,
        Kind::Pdf => pdf_file(path, cx.sink)?,
        Kind::Docx => docx(path, cx.sink)?,
        Kind::Pptx => pptx(path, cx.sink)?,
        Kind::Odf => odf(path, cx.sink)?,
        Kind::Epub => epub(path, cx.sink)?,
        Kind::Xlsx => sheets(path, ext, cx.tmp, cx)?,
        Kind::Gzip => gzip(cx, path, name, depth)?,
        Kind::Zip => archive(cx, path, depth)?,
        Kind::Legacy => {
            return msg("Legacy binary Office files (.doc/.ppt) aren't supported. Re-save as .docx / .pptx (or export to PDF/TXT) and upload that.")
        }
        Kind::Unsupported => return msg("Not a text-bearing file (images, audio, video and binaries can't be trained on; OCR/transcription isn't supported)."),
    }
    if !matches!(kind, Kind::Zip | Kind::Gzip) {
        cx.files += 1;
    }
    Ok(())
}

// ------------------------------------------------------------- plain text

fn sniff_encoding(head: &[u8]) -> (&'static encoding_rs::Encoding, usize) {
    if let Some((enc, len)) = encoding_rs::Encoding::for_bom(head) {
        return (enc, len);
    }
    // valid UTF-8 (allowing a code point cut off at the end of the sample)?
    let utf8 = match std::str::from_utf8(head) {
        Ok(_) => true,
        Err(e) => e.error_len().is_none() && e.valid_up_to() > 0 || e.valid_up_to() + 4 >= head.len() && e.error_len().is_none(),
    };
    if utf8 {
        return (encoding_rs::UTF_8, 0);
    }
    // UTF-16 without BOM: NUL in every other byte
    let n = head.len().min(4096) & !1;
    if n >= 8 {
        let (even, odd) = head[..n].chunks(2).fold((0, 0), |(e, o), p| (e + (p[0] == 0) as usize, o + (p[1] == 0) as usize));
        if odd * 3 > n / 2 * 2 {
            return (encoding_rs::UTF_16LE, 0);
        }
        if even * 3 > n / 2 * 2 {
            return (encoding_rs::UTF_16BE, 0);
        }
    }
    (encoding_rs::WINDOWS_1252, 0)
}

/// Streams any readable byte source through the right decoder into the sink.
fn decode_stream(mut r: impl Read, sink: &mut Sink) -> R<()> {
    let mut first = vec![0u8; HEAD];
    let mut n = 0;
    while n < first.len() {
        let k = r.read(&mut first[n..])?;
        if k == 0 {
            break;
        }
        n += k;
    }
    first.truncate(n);
    let (enc, skip) = sniff_encoding(&first);
    let mut dec = enc.new_decoder_without_bom_handling();
    let mut out = String::new();
    let mut feed = |chunk: &[u8], last: bool, out: &mut String, sink: &mut Sink| -> R<()> {
        out.clear();
        out.reserve(dec.max_utf8_buffer_length(chunk.len()).unwrap_or(chunk.len() * 3 + 16));
        let (_, _, _) = dec.decode_to_string(chunk, out, last);
        sink.push(out)
    };
    if n == 0 {
        return Ok(());
    }
    let mut buf = vec![0u8; 1 << 20];
    let head_len = first.len();
    let done = head_len < HEAD;
    feed(&first[skip..], done, &mut out, sink)?;
    if done {
        return Ok(());
    }
    loop {
        let k = r.read(&mut buf)?;
        feed(&buf[..k], k == 0, &mut out, sink)?;
        if k == 0 {
            return Ok(());
        }
    }
}

fn text_file(path: &Path, sink: &mut Sink) -> R<()> {
    decode_stream(File::open(path)?, sink)
}

// ------------------------------------------------------------------- HTML

const BLOCK: &[&str] = &[
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header", "footer", "blockquote",
    "pre", "hr", "title", "dt", "dd", "figcaption", "nav", "aside", "form",
];

fn entity(name: &str) -> Option<char> {
    Some(match name {
        "amp" => '&', "lt" => '<', "gt" => '>', "quot" => '"', "apos" => '\'', "nbsp" => ' ', "copy" => '©', "reg" => '®', "hellip" => '…',
        "mdash" => '—', "ndash" => '–', "lsquo" => '‘', "rsquo" => '’', "ldquo" => '“', "rdquo" => '”', "bull" => '•', "middot" => '·',
        "euro" => '€', "pound" => '£', "trade" => '™', "deg" => '°',
        n if n.starts_with("#x") || n.starts_with("#X") => return u32::from_str_radix(&n[2..], 16).ok().and_then(char::from_u32),
        n if n.starts_with('#') => return n[1..].parse::<u32>().ok().and_then(char::from_u32),
        _ => return None,
    })
}

/// Visible text of an HTML document. Scripts, styles and comments are dropped; block tags become
/// line breaks; whitespace is collapsed.
pub fn strip_html(html: &str, sink: &mut Sink) -> R<()> {
    let b = html.as_bytes();
    let mut out = String::with_capacity(html.len() / 2);
    let mut i = 0;
    let mut space = false;
    let flush = |out: &mut String, sink: &mut Sink| -> R<()> {
        if out.len() > 1 << 16 {
            sink.push(out)?;
            out.clear();
        }
        Ok(())
    };
    while i < b.len() {
        match b[i] {
            b'<' => {
                if html[i..].starts_with("<!--") {
                    i = html[i..].find("-->").map_or(b.len(), |e| i + e + 3);
                    continue;
                }
                let end = html[i..].find('>').map_or(b.len(), |e| i + e + 1);
                let tag = html[i + 1..end.saturating_sub(1).max(i + 1)].trim_start_matches('/');
                let tname: String = tag.chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
                if matches!(tname.as_str(), "script" | "style" | "noscript" | "svg" | "head") && !html[i + 1..].starts_with('/') {
                    let close = format!("</{tname}");
                    let lower_rest = html[end.min(b.len())..].to_ascii_lowercase();
                    i = lower_rest.find(&close).map_or(b.len(), |p| {
                        let from = end + p;
                        html[from..].find('>').map_or(b.len(), |q| from + q + 1)
                    });
                    continue;
                }
                if BLOCK.contains(&tname.as_str()) {
                    out.push('\n');
                    space = false;
                }
                i = end;
            }
            b'&' => {
                if let Some(semi) = html[i..].find(';').filter(|&s| s <= 10) {
                    if let Some(c) = entity(&html[i + 1..i + semi]) {
                        if c == ' ' {
                            space = true;
                        } else {
                            if space && !out.ends_with('\n') && !out.is_empty() {
                                out.push(' ');
                            }
                            space = false;
                            out.push(c);
                        }
                        i += semi + 1;
                        continue;
                    }
                }
                if space && !out.ends_with('\n') && !out.is_empty() {
                    out.push(' ');
                }
                space = false;
                out.push('&');
                i += 1;
            }
            _ => {
                let c = html[i..].chars().next().unwrap();
                if c.is_whitespace() {
                    space = true;
                } else {
                    if space && !out.ends_with('\n') && !out.is_empty() {
                        out.push(' ');
                    }
                    space = false;
                    out.push(c);
                }
                i += c.len_utf8();
            }
        }
        flush(&mut out, sink)?;
    }
    sink.push(&out)
}

fn read_decoded(path: &Path) -> R<String> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let (enc, skip) = sniff_encoding(&bytes[..bytes.len().min(HEAD)]);
    let (s, _) = enc.decode_without_bom_handling(&bytes[skip..]);
    Ok(s.into_owned())
}

fn html_file(path: &Path, sink: &mut Sink) -> R<()> {
    strip_html(&read_decoded(path)?, sink)
}

// -------------------------------------------------------------------- RTF

/// Plain text of an RTF document (control words dropped; `\par` -> newline; `\'hh` and `\uN` decoded).
pub fn rtf_to_text(rtf: &str) -> String {
    const SKIP: &[&str] = &[
        "fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer", "headerl", "headerr", "footerl", "footerr", "object", "themedata",
        "colorschememapping", "latentstyles", "datastore", "generator", "listtable", "listoverridetable", "rsidtbl", "filetbl", "revtbl",
        "fldinst", "xmlnstbl", "mmathPr", "wgrffmtfilter", "private",
    ];
    let cs: Vec<char> = rtf.chars().collect();
    let mut out = String::new();
    let mut stack: Vec<bool> = vec![false]; // skip flags
    let mut i = 0;
    let mut uc_skip = 0usize;
    while i < cs.len() {
        let c = cs[i];
        let skipping = *stack.last().unwrap();
        match c {
            '{' => {
                stack.push(skipping);
                i += 1;
                // destination groups: {\*\word ...} or {\fonttbl ...}
                if cs.get(i) == Some(&'\\') {
                    if cs.get(i + 1) == Some(&'*') {
                        *stack.last_mut().unwrap() = true;
                    } else {
                        let w: String = cs[i + 1..].iter().take_while(|c| c.is_ascii_alphabetic()).collect();
                        if SKIP.contains(&w.as_str()) {
                            *stack.last_mut().unwrap() = true;
                        }
                    }
                }
            }
            '}' => {
                if stack.len() > 1 {
                    stack.pop();
                }
                i += 1;
            }
            '\\' => {
                i += 1;
                let Some(&n) = cs.get(i) else { break };
                if n.is_ascii_alphabetic() {
                    let start = i;
                    while i < cs.len() && cs[i].is_ascii_alphabetic() {
                        i += 1;
                    }
                    let word: String = cs[start..i].iter().collect();
                    let ns = i;
                    if i < cs.len() && (cs[i] == '-' || cs[i].is_ascii_digit()) {
                        i += 1;
                        while i < cs.len() && cs[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                    let param: Option<i32> = cs[ns..i].iter().collect::<String>().parse().ok();
                    if i < cs.len() && cs[i] == ' ' {
                        i += 1;
                    }
                    if !skipping {
                        match word.as_str() {
                            "par" | "line" | "row" | "sect" | "page" => out.push('\n'),
                            "tab" | "cell" => out.push('\t'),
                            "emdash" => out.push('—'),
                            "endash" => out.push('–'),
                            "bullet" => out.push('•'),
                            "lquote" => out.push('‘'),
                            "rquote" => out.push('’'),
                            "ldblquote" => out.push('“'),
                            "rdblquote" => out.push('”'),
                            "u" => {
                                if let Some(p) = param {
                                    let cp = if p < 0 { (p + 65536) as u32 } else { p as u32 };
                                    if let Some(ch) = char::from_u32(cp) {
                                        out.push(ch);
                                    }
                                    uc_skip = 1;
                                }
                            }
                            _ => {}
                        }
                    }
                } else if n == '\'' {
                    let hex: String = cs[i + 1..].iter().take(2).collect();
                    i += 3;
                    if !skipping && uc_skip == 0 {
                        if let Ok(b) = u8::from_str_radix(&hex, 16) {
                            let byte = [b];
                            let (s, _) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(&byte);
                            out.push_str(&s);
                        }
                    }
                    uc_skip = uc_skip.saturating_sub(1);
                } else {
                    if !skipping && matches!(n, '\\' | '{' | '}') {
                        out.push(n);
                    }
                    if n == '~' && !skipping {
                        out.push(' ');
                    }
                    i += 1;
                }
            }
            '\r' | '\n' => i += 1,
            _ => {
                if uc_skip > 0 {
                    uc_skip -= 1; // fallback character after \uN
                } else if !skipping {
                    out.push(c);
                }
                i += 1;
            }
        }
    }
    out
}

fn rtf_file(path: &Path, sink: &mut Sink) -> R<()> {
    let mut s = String::new();
    File::open(path)?.read_to_string(&mut s).or_else(|_| -> io::Result<usize> {
        // RTF is 7-bit ASCII with escapes, but tolerate stray bytes
        let mut b = Vec::new();
        File::open(path)?.read_to_end(&mut b)?;
        s = String::from_utf8_lossy(&b).into_owned();
        Ok(s.len())
    })?;
    sink.push(&rtf_to_text(&s))
}

// -------------------------------------------------------------------- PDF

/// Offsets of every `N G obj` in the file (the last definition of an object wins, as in incremental updates).
fn scan_objects(b: &[u8]) -> Vec<(u32, u16, usize)> {
    let mut found: std::collections::BTreeMap<u32, (u16, usize)> = Default::default();
    let mut i = 0;
    while let Some(p) = b[i..].windows(3).position(|w| w == b"obj").map(|p| p + i) {
        i = p + 3;
        if p > 0 && b[p - 1].is_ascii_alphabetic() {
            continue; // "endobj"
        }
        // walk back over "<num> <gen> " (single spaces or any whitespace)
        let mut j = p;
        let skip_ws = |j: &mut usize| {
            let e = *j;
            while *j > 0 && b[*j - 1].is_ascii_whitespace() {
                *j -= 1;
            }
            *j < e
        };
        let digits = |j: &mut usize| {
            let e = *j;
            while *j > 0 && b[*j - 1].is_ascii_digit() {
                *j -= 1;
            }
            (*j, e)
        };
        if !skip_ws(&mut j) {
            continue;
        }
        let (gs, ge) = digits(&mut j);
        if gs == ge || !skip_ws(&mut j) {
            continue;
        }
        let (ns, ne) = digits(&mut j);
        if ns == ne || (ns > 0 && !b[ns - 1].is_ascii_whitespace() && b[ns - 1] != b'>' && b[ns - 1] != b']') {
            continue;
        }
        let (Some(num), Some(gen)) = (
            std::str::from_utf8(&b[ns..ne]).ok().and_then(|x| x.parse::<u32>().ok()),
            std::str::from_utf8(&b[gs..ge]).ok().and_then(|x| x.parse::<u16>().ok()),
        ) else {
            continue;
        };
        found.insert(num, (gen, ns));
    }
    found.into_iter().map(|(n, (g, o))| (n, g, o)).collect()
}

/// PDFs whose cross-reference table is wrong (very common after line-ending conversion or
/// hand-editing) are rejected by strict parsers. Like most PDF viewers, rebuild the index by
/// scanning for the objects, and append a fresh table + trailer. Returns `None` when there is
/// nothing to rebuild from (e.g. cross-reference *streams* / object streams, or encryption).
fn repair_pdf(b: &[u8]) -> Option<Vec<u8>> {
    if find(b, b"/Encrypt").is_some() {
        return None;
    }
    let objs = scan_objects(b);
    if objs.is_empty() {
        return None;
    }
    // Root: from the last trailer that names one, else the object whose dictionary says /Catalog
    let mut root: Option<(u32, u16)> = None;
    let mut from = 0;
    while let Some(t) = find(&b[from..], b"trailer").map(|x| x + from) {
        let end = (t + 2048).min(b.len());
        if let Some(r) = find(&b[t..end], b"/Root").map(|x| x + t) {
            let txt = String::from_utf8_lossy(&b[r + 5..end.min(r + 64)]).into_owned();
            let mut it = txt.split_whitespace();
            if let (Some(n), Some(g), Some("R")) = (it.next(), it.next(), it.next().map(|x| x.trim_end_matches(|c: char| !c.is_alphanumeric()))) {
                if let (Ok(n), Ok(g)) = (n.parse(), g.parse()) {
                    root = Some((n, g));
                }
            }
        }
        from = t + 7;
    }
    if root.is_none() {
        for &(n, g, off) in &objs {
            let end = (off + 512).min(b.len());
            if find(&b[off..end], b"/Type/Catalog").is_some() || find(&b[off..end], b"/Type /Catalog").is_some() {
                root = Some((n, g));
            }
        }
    }
    let (rn, rg) = root?;
    let max = objs.iter().map(|o| o.0).max()?;
    let mut out = b.to_vec();
    out.extend_from_slice(b"\n");
    let xref_at = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", max + 1).as_bytes());
    let by_num: std::collections::HashMap<u32, (u16, usize)> = objs.iter().map(|&(n, g, o)| (n, (g, o))).collect();
    for n in 0..=max {
        match by_num.get(&n) {
            Some(&(g, o)) => out.extend_from_slice(format!("{o:010} {g:05} n \n").as_bytes()),
            None => out.extend_from_slice(b"0000000000 65535 f \n"),
        }
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root {rn} {rg} R >>\nstartxref\n{xref_at}\n%%EOF\n", max + 1).as_bytes());
    Some(out)
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Last resort: poppler's `pdftotext`, if it happens to be installed.
fn pdftotext(path: &Path) -> Option<String> {
    let out = std::process::Command::new("pdftotext").arg("-enc").arg("UTF-8").arg(path).arg("-").stdin(std::process::Stdio::null()).output().ok()?;
    (out.status.success() && !out.stdout.is_empty()).then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// pdf-extract can panic on malformed files; keep that from taking the server down.
fn guarded<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> Option<T> {
    std::panic::catch_unwind(f).ok()
}

fn push_pages(sink: &mut Sink, pages: &[String]) -> R<()> {
    for p in pages {
        // blank pages keep their number (so page N is the PDF's page N) but add no text
        sink.mark_page();
        if !p.trim().is_empty() {
            sink.push(p)?;
            sink.break_paragraph()?;
        }
    }
    Ok(())
}

fn pdf_file(path: &Path, sink: &mut Sink) -> R<()> {
    // 1) the normal route
    let p = path.to_path_buf();
    let first = guarded(move || pdf_extract::extract_text_by_pages(&p));
    let err = match first {
        Some(Ok(pages)) => return push_pages(sink, &pages),
        Some(Err(e)) => e.to_string(),
        None => "the parser crashed".to_string(),
    };
    let bytes = fs::read(path)?;
    let encrypted = find(&bytes, b"/Encrypt").is_some();
    // 2) rebuild a broken cross-reference table and try again
    if let Some(fixed) = repair_pdf(&bytes) {
        if let Some(Ok(pages)) = guarded(move || pdf_extract::extract_text_from_mem_by_pages(&fixed)) {
            if pages.iter().any(|p| !p.trim().is_empty()) {
                return push_pages(sink, &pages);
            }
        }
    }
    // 3) poppler, when available (pages are separated by form feeds)
    if let Some(text) = pdftotext(path) {
        let pages: Vec<String> = text.split('\u{c}').map(str::to_string).collect();
        return push_pages(sink, &pages);
    }
    if encrypted {
        return msg("This PDF is password-protected.");
    }
    msg(format!("Could not read this PDF (its internal structure is damaged): {err}"))
}

// --------------------------------------------------------------- OOXML/ODF

type Zip = zip::ZipArchive<BufReader<File>>;

fn open_zip(path: &Path) -> R<Zip> {
    zip::ZipArchive::new(BufReader::new(File::open(path)?)).map_err(|e| Ex::Msg(format!("corrupt container: {e}")))
}

fn xml_err(e: quick_xml::Error) -> Ex {
    Ex::Msg(format!("malformed XML: {e}"))
}

/// Where extracted text goes. Inside a table cell it is buffered so a row can be written as
/// `cell<TAB>cell<TAB>cell\n` (paragraph breaks inside a cell become spaces).
struct Out<'a> {
    sink: &'a mut Sink,
    cell_depth: usize,
    cell: String,
    row: Vec<String>,
}

impl Out<'_> {
    fn text(&mut self, s: &str) -> R<()> {
        if self.cell_depth > 0 {
            self.cell.push_str(s);
            Ok(())
        } else {
            self.sink.push(s)
        }
    }
    fn paragraph_end(&mut self) -> R<()> {
        if self.cell_depth > 0 {
            if !self.cell.is_empty() && !self.cell.ends_with(' ') {
                self.cell.push(' ');
            }
            Ok(())
        } else {
            self.sink.push("\n")
        }
    }
    fn special(&mut self, s: &str) -> R<()> {
        // tabs / line breaks inside a cell would break the row layout
        if self.cell_depth > 0 {
            self.cell.push(' ');
            Ok(())
        } else {
            self.sink.push(s)
        }
    }
    fn cell_start(&mut self) {
        self.cell_depth += 1;
        if self.cell_depth == 1 {
            self.cell.clear();
        }
    }
    fn cell_end(&mut self) {
        if self.cell_depth == 1 {
            self.row.push(self.cell.split_whitespace().collect::<Vec<_>>().join(" "));
        }
        self.cell_depth = self.cell_depth.saturating_sub(1);
    }
    fn row_end(&mut self) -> R<()> {
        if self.cell_depth == 0 && !self.row.is_empty() {
            let line = self.row.join("\t");
            self.row.clear();
            self.sink.push(&line)?;
            self.sink.push("\n")?;
        }
        Ok(())
    }
}

/// Element names that give an XML part its text structure.
struct Tags<'a> {
    /// Elements whose character data is content (`w:t`, `a:t`); `None` = all text inside a paragraph (ODF).
    text: Option<&'a [u8]>,
    paragraphs: &'a [&'a [u8]],
    cell: Option<&'a [u8]>,
    row: Option<&'a [u8]>,
    /// Empty elements that stand for literal text (`w:tab` -> tab).
    specials: &'a [(&'a [u8], &'a str)],
}

/// Streams the text of one XML part.
fn xml_text(r: impl Read, sink: &mut Sink, t: &Tags) -> R<()> {
    use quick_xml::events::Event;
    let mut rd = quick_xml::Reader::from_reader(BufReader::new(r));
    let mut buf = Vec::new();
    let mut out = Out { sink, cell_depth: 0, cell: String::new(), row: Vec::new() };
    let (mut in_text, mut para_depth) = (false, 0usize);
    loop {
        match rd.read_event_into(&mut buf).map_err(xml_err)? {
            Event::Start(e) => {
                let n = e.name();
                let n = n.as_ref();
                if Some(n) == t.text {
                    in_text = true;
                }
                if t.paragraphs.contains(&n) {
                    para_depth += 1;
                }
                if Some(n) == t.cell {
                    out.cell_start();
                }
            }
            Event::End(e) => {
                let n = e.name();
                let n = n.as_ref();
                if Some(n) == t.text {
                    in_text = false;
                }
                if t.paragraphs.contains(&n) {
                    para_depth = para_depth.saturating_sub(1);
                    out.paragraph_end()?;
                }
                if Some(n) == t.cell {
                    out.cell_end();
                }
                if Some(n) == t.row {
                    out.row_end()?;
                }
            }
            Event::Empty(e) => {
                if let Some((_, s)) = t.specials.iter().find(|(name, _)| *name == e.name().as_ref()) {
                    out.special(s)?;
                }
            }
            Event::Text(x) => {
                let inside = if t.text.is_some() { in_text } else { para_depth > 0 };
                if inside {
                    let s = x.unescape().map_err(xml_err)?;
                    out.text(&s)?;
                }
            }
            Event::CData(x) => {
                if in_text {
                    out.text(&String::from_utf8_lossy(&x))?;
                }
            }
            Event::Eof => return Ok(()),
            _ => {}
        }
        buf.clear();
    }
}

fn zip_entry_reader<'a>(z: &'a mut Zip, name: &str) -> R<zip::read::ZipFile<'a>> {
    z.by_name(name).map_err(|e| Ex::Msg(format!("missing part {name}: {e}")))
}

fn docx(path: &Path, sink: &mut Sink) -> R<()> {
    let mut z = open_zip(path)?;
    let specials: &[(&[u8], &str)] = &[(b"w:tab", "\t"), (b"w:br", "\n"), (b"w:cr", "\n"), (b"w:noBreakHyphen", "-")];
    let tags = Tags { text: Some(b"w:t"), paragraphs: &[b"w:p"], cell: Some(b"w:tc"), row: Some(b"w:tr"), specials };
    let body = zip_entry_reader(&mut z, "word/document.xml")?;
    xml_text(body, sink, &tags)?;
    for extra in ["word/footnotes.xml", "word/endnotes.xml"] {
        if let Ok(e) = z.by_name(extra) {
            sink.break_paragraph()?;
            xml_text(e, sink, &tags)?;
        }
    }
    Ok(())
}

fn numbered(names: impl Iterator<Item = String>, prefix: &str) -> Vec<String> {
    let mut v: Vec<(u32, String)> = names
        .filter(|n| n.starts_with(prefix) && n.ends_with(".xml"))
        .filter_map(|n| n[prefix.len()..n.len() - 4].parse::<u32>().ok().map(|k| (k, n)))
        .collect();
    v.sort();
    v.into_iter().map(|x| x.1).collect()
}

fn pptx(path: &Path, sink: &mut Sink) -> R<()> {
    let mut z = open_zip(path)?;
    let slides = numbered(z.file_names().map(String::from), "ppt/slides/slide");
    let notes = numbered(z.file_names().map(String::from), "ppt/notesSlides/notesSlide");
    let specials: &[(&[u8], &str)] = &[(b"a:br", "\n")];
    // pptx tables use a:tc / a:tr
    let tags = Tags { text: Some(b"a:t"), paragraphs: &[b"a:p"], cell: Some(b"a:tc"), row: Some(b"a:tr"), specials };
    for (i, s) in slides.iter().enumerate() {
        sink.mark_page();
        let slide = zip_entry_reader(&mut z, s)?;
        xml_text(slide, sink, &tags)?;
        // speaker notes for the same slide number, if any
        let n = format!("ppt/notesSlides/notesSlide{}.xml", i + 1);
        if notes.contains(&n) {
            let note = zip_entry_reader(&mut z, &n)?;
            xml_text(note, sink, &tags)?;
        }
        sink.break_paragraph()?;
    }
    Ok(())
}

fn odf(path: &Path, sink: &mut Sink) -> R<()> {
    let mut z = open_zip(path)?;
    let specials: &[(&[u8], &str)] = &[(b"text:tab", "\t"), (b"text:line-break", "\n"), (b"text:s", " ")];
    let content = zip_entry_reader(&mut z, "content.xml")?;
    let tags = Tags { text: None, paragraphs: &[b"text:p", b"text:h"], cell: Some(b"table:table-cell"), row: Some(b"table:table-row"), specials };
    xml_text(content, sink, &tags)
}

fn epub(path: &Path, sink: &mut Sink) -> R<()> {
    let mut z = open_zip(path)?;
    let mut names: Vec<String> = z.file_names().filter(|n| n.ends_with(".xhtml") || n.ends_with(".html") || n.ends_with(".htm")).map(String::from).collect();
    names.sort();
    for n in names {
        let mut s = String::new();
        let mut e = zip_entry_reader(&mut z, &n)?;
        if e.by_ref().take(64 << 20).read_to_string(&mut s).is_err() {
            continue;
        }
        strip_html(&s, sink)?;
        sink.break_paragraph()?;
    }
    Ok(())
}

// ------------------------------------------------------------ spreadsheets

fn sheets(path: &Path, ext: &str, tmp: &Path, cx: &mut Ctx) -> R<()> {
    use calamine::{open_workbook_auto, Reader};
    // calamine picks the parser from the extension, so give the file the right one
    cx.serial += 1;
    let named = tmp.join(format!("sheet{}.{ext}", cx.serial));
    fs::hard_link(path, &named).or_else(|_| fs::copy(path, &named).map(|_| ()))?;
    let res = (|| -> R<()> {
        let mut wb = open_workbook_auto(&named).map_err(|e| Ex::Msg(format!("Could not open this spreadsheet: {e}")))?;
        for name in wb.sheet_names().to_vec() {
            let range = wb.worksheet_range(&name).map_err(|e| Ex::Msg(format!("Could not read sheet {name}: {e}")))?;
            if range.is_empty() {
                continue;
            }
            cx.sink.push(&format!("{name}\n"))?;
            for row in range.rows() {
                let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
                if cells.iter().all(|c| c.is_empty()) {
                    continue;
                }
                cx.sink.push(&cells.join("\t"))?;
                cx.sink.push("\n")?;
            }
            cx.sink.break_paragraph()?;
        }
        Ok(())
    })();
    let _ = fs::remove_file(&named);
    res
}

// ------------------------------------------------------ containers (nested)

fn scratch(cx: &mut Ctx, ext: &str) -> PathBuf {
    cx.serial += 1;
    cx.tmp.join(format!("n{}.{}", cx.serial, ext))
}

fn gzip(cx: &mut Ctx, path: &Path, name: &str, depth: u32) -> R<()> {
    let inner_name = name.strip_suffix(".gz").or_else(|| name.strip_suffix(".GZ")).unwrap_or(name).to_string();
    let tmp = scratch(cx, "inner");
    let res = (|| -> R<()> {
        let mut dec = flate2::read::MultiGzDecoder::new(BufReader::new(File::open(path)?));
        let mut out = BufWriter::new(File::create(&tmp)?);
        // A gzip bomb must not fill the disk: allow a generous multiple of the sink cap.
        let limit = cx.sink_room().saturating_mul(4).max(1 << 20);
        let copied = io::copy(&mut (&mut dec).take(limit), &mut out)?;
        out.flush()?;
        if copied >= limit {
            cx.sink.truncated = true;
        }
        extract_path(cx, &tmp, &inner_name, depth + 1)
    })();
    let _ = fs::remove_file(&tmp);
    res
}

impl Ctx<'_> {
    fn sink_room(&self) -> u64 {
        self.sink.cap_left()
    }
}

impl Sink {
    fn cap_left(&self) -> u64 {
        self.cap.saturating_sub(self.bytes)
    }
}

fn archive(cx: &mut Ctx, path: &Path, depth: u32) -> R<()> {
    let mut z = open_zip(path)?;
    let total = z.len();
    if total > MAX_ZIP_ENTRIES {
        return msg(format!("archive has {total} entries; the limit is {MAX_ZIP_ENTRIES}"));
    }
    for i in 0..total {
        let (name, is_dir, size) = match z.by_index(i) {
            Ok(e) => (e.name().to_string(), e.is_dir(), e.size()),
            Err(_) => continue,
        };
        let base = name.rsplit('/').next().unwrap_or(&name);
        if is_dir || name.starts_with("__MACOSX/") || base.starts_with('.') || base.is_empty() {
            continue;
        }
        if size > cx.sink_room().saturating_mul(4).max(1 << 20) {
            cx.skipped.push(format!("{base}: too large"));
            continue;
        }
        let tmp = scratch(cx, "entry");
        let copied = (|| -> io::Result<u64> {
            let mut e = z.by_index(i).map_err(|e| io::Error::other(e.to_string()))?;
            let mut out = BufWriter::new(File::create(&tmp)?);
            let n = io::copy(&mut e, &mut out)?;
            out.flush()?;
            Ok(n)
        })();
        if copied.is_err() {
            cx.skipped.push(format!("{base}: unreadable"));
            let _ = fs::remove_file(&tmp);
            continue;
        }
        let before = cx.sink.bytes;
        let res = extract_path(cx, &tmp, base, depth + 1);
        let _ = fs::remove_file(&tmp);
        match res {
            Ok(()) => {
                if cx.sink.bytes > before {
                    cx.sink.break_paragraph()?;
                }
            }
            Err(Ex::Cap) => return Err(Ex::Cap),
            Err(Ex::Msg(m)) => {
                // one bad entry must not sink the whole archive
                let short = m.split('.').next().unwrap_or(&m).chars().take(60).collect::<String>();
                cx.skipped.push(format!("{base}: {short}"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    fn run_path(src: &Path, name: &str) -> Result<(Outcome, String), String> {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");
        let o = extract_to(src, name, &out, &dir.path().join("scratch"), 1 << 30)?;
        Ok((o, fs::read_to_string(out).unwrap()))
    }

    fn run(name: &str) -> Result<(Outcome, String), String> {
        run_path(&fixture(name), name)
    }

    fn ok(name: &str) -> (Outcome, String) {
        run(name).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    #[test]
    fn word_docx_keeps_paragraphs_tables_and_unicode() {
        let (o, t) = ok("sample.docx");
        assert_eq!(o.kind, Kind::Docx);
        assert!(t.contains("DOCXMARK the harbour lights flicker over quiet water."), "{t}");
        assert!(t.contains("Naïve café résumé — unicode survives."));
        assert!(t.contains("Region\tRevenue") && t.contains("North\t4200"), "table cells should be tab separated: {t:?}");
    }

    #[test]
    fn pdf_text_layer_across_pages_and_scanned_pdf_error() {
        let (o, t) = ok("sample.pdf");
        assert_eq!(o.kind, Kind::Pdf);
        assert!(t.contains("PDFMARK the lighthouse keeper counted seven ships."), "{t}");
        assert!(t.contains("copper kettles whistle at dawn"), "{t}");
        let e = run("blank.pdf").unwrap_err();
        assert!(e.contains("scanned") && e.contains("OCR"), "{e}");
    }

    #[test]
    fn damaged_pdfs_are_repaired_like_a_pdf_viewer_would() {
        // the strict parser alone rejects these; the reader must not
        for f in ["shifted.pdf", "truncated.pdf", "crlf_header.pdf"] {
            assert!(pdf_extract::extract_text(fixture(f)).is_err(), "{f} should be broken for the strict parser (else this test proves nothing)");
            let (o, t) = ok(f);
            assert_eq!(o.kind, Kind::Pdf, "{f}");
            assert!(t.contains("PDFMARK the lighthouse keeper counted seven ships."), "{f}: {t:?}");
            assert!(t.contains("copper kettles whistle at dawn"), "{f}: {t:?}");
        }
    }

    #[test]
    fn repair_gives_up_cleanly_on_garbage_and_encrypted_files() {
        assert!(repair_pdf(b"%PDF-1.4\nnothing here\n").is_none());
        assert!(repair_pdf(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer << /Encrypt 5 0 R /Root 1 0 R >>").is_none());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("junk.pdf");
        fs::write(&p, b"%PDF-1.4\n1 0 obj\n<< /Foo /Bar >>\nendobj\n").unwrap();
        let e = run_path(&p, "junk.pdf").unwrap_err();
        assert!(e.contains("damaged") || e.contains("scanned") || e.contains("No text"), "{e}");
    }

    #[test]
    fn text_in_any_encoding() {
        assert!(ok("sample.txt").1.contains("TXTMARK plain utf-8 text with ünïcode and 日本語."));
        assert!(ok("latin1.txt").1.contains("LATINMARK café crème brûlée naïve"));
        assert!(ok("utf16.txt").1.contains("UTF16MARK wide characters ok"));
        assert!(ok("sample.csv").1.contains("grace,20"));
    }

    #[test]
    fn html_drops_scripts_styles_and_comments() {
        let (o, t) = ok("sample.html");
        assert_eq!(o.kind, Kind::Html);
        assert!(t.contains("HTMLMARK Heading") && t.contains("First bold& second para."), "{t:?}");
        assert!(!t.contains("NOTVISIBLE") && !t.contains("hidden") && !t.contains("color:red"), "{t:?}");
        assert!(t.contains("one") && t.contains("two"));
    }

    #[test]
    fn opendocument_rtf_and_epub_match_the_word_original() {
        for f in ["sample.odt", "sample.rtf", "sample.epub"] {
            let (_, t) = ok(f);
            assert!(t.contains("DOCXMARK the harbour lights flicker over quiet water."), "{f}: {t:?}");
        }
        assert!(ok("sample.odt").1.contains("Naïve café résumé"));
        assert!(ok("sample.rtf").1.contains("Naïve café résumé"), "{:?}", ok("sample.rtf").1);
        assert_eq!(ok("sample.odt").0.kind, Kind::Odf);
        assert_eq!(ok("sample.epub").0.kind, Kind::Epub);
    }

    #[test]
    fn spreadsheets_every_flavour() {
        for f in ["sheet.xlsx", "sheet.xls", "sheet.ods"] {
            let (o, t) = ok(f);
            assert_eq!(o.kind, Kind::Xlsx, "{f}");
            assert!(t.contains("city\tpopulation") && t.contains("Oslo\t709000") && t.contains("Bergen\t286000"), "{f}: {t:?}");
        }
    }

    #[test]
    fn presentations_slides_and_notes() {
        for f in ["slides.pptx", "slides.odp"] {
            let (_, t) = ok(f);
            assert!(t.contains("PPTMARK Slide one title") && t.contains("bullet alpha") && t.contains("granite quarries"), "{f}: {t:?}");
        }
        assert_eq!(ok("slides.pptx").0.kind, Kind::Pptx);
    }

    #[test]
    fn legacy_office_binaries_are_refused_with_advice() {
        for f in ["sample.doc", "slides.ppt"] {
            let e = run(f).unwrap_err();
            assert!(e.contains(".docx") && e.contains("Legacy"), "{f}: {e}");
        }
    }

    #[test]
    fn gzip_and_zip_archives_recurse_and_skip_junk() {
        let (o, t) = ok("sample.txt.gz");
        assert_eq!(o.kind, Kind::Gzip);
        assert!(t.contains("GZMARK compressed lines"));

        let (o, t) = ok("bundle.zip");
        assert_eq!(o.kind, Kind::Zip);
        assert!(t.contains("DOCXMARK") && t.contains("TXTMARK") && t.contains("DEEPMARK"), "{t:?}");
        let note = o.note.unwrap();
        assert!(note.contains("files read") && note.contains("logo.png"), "{note}");
    }

    #[test]
    fn detection_uses_content_not_the_file_name() {
        let dir = tempfile::tempdir().unwrap();
        for (real, fake, kind) in [("sample.docx", "report.bin", Kind::Docx), ("sample.pdf", "notes.txt", Kind::Pdf), ("sheet.xlsx", "data.dat", Kind::Xlsx)] {
            let p = dir.path().join(fake);
            fs::copy(fixture(real), &p).unwrap();
            assert_eq!(detect(fake, &p).unwrap().0, kind, "{real} renamed to {fake}");
            assert!(run_path(&p, fake).is_ok(), "{real} renamed to {fake} should still extract");
        }
        let png = dir.path().join("pic.png");
        fs::write(&png, b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR").unwrap();
        assert_eq!(detect("pic.png", &png).unwrap().0, Kind::Unsupported);
        assert!(run_path(&png, "pic.png").unwrap_err().contains("OCR"));
    }

    #[test]
    fn output_cap_truncates_instead_of_failing() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("big.txt");
        fs::write(&src, "lorem ipsum dolor sit amet\n".repeat(10_000)).unwrap();
        let out = dir.path().join("o.txt");
        let o = extract_to(&src, "big.txt", &out, &dir.path().join("t"), 1000).unwrap();
        assert!(o.truncated && o.bytes <= 1000 && o.note.unwrap().contains("truncated"));
    }

    #[test]
    fn corrupt_and_empty_files_give_clear_errors_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let bad_pdf = dir.path().join("bad.pdf");
        fs::write(&bad_pdf, b"%PDF-1.4\nthis is not really a pdf\n").unwrap();
        assert!(run_path(&bad_pdf, "bad.pdf").is_err());
        let bad_zip = dir.path().join("bad.docx");
        fs::write(&bad_zip, b"PK\x03\x04garbage garbage").unwrap();
        assert!(run_path(&bad_zip, "bad.docx").is_err());
        let empty = dir.path().join("e.txt");
        fs::write(&empty, b"").unwrap();
        assert!(run_path(&empty, "e.txt").unwrap_err().contains("No readable text"));
    }

    #[test]
    fn rtf_and_html_helpers() {
        assert_eq!(rtf_to_text(r"{\rtf1\ansi{\fonttbl{\f0 Arial;}}Hello\par World \'e9 \u8364?}").trim(), "Hello\nWorld é €");
        let dir = tempfile::tempdir().unwrap();
        let mut s = Sink::create(&dir.path().join("x"), 1 << 20).unwrap();
        strip_html("<p>a &lt; b &#8364; &amp;c</p><script>x</script><p>next</p>", &mut s).unwrap();
        s.finish().unwrap();
        assert_eq!(fs::read_to_string(dir.path().join("x")).unwrap().trim(), "a < b € &c\n\nnext");
    }
}


#[cfg(test)]
mod local_pdf_check {
    /// Manual check against a real file that must not be committed:
    /// `PDF_PATH=/path/to.pdf cargo test local_pdf_check -- --ignored --nocapture`
    /// Prints sizes only, never the document's text.
    #[test]
    #[ignore]
    fn compare_with_poppler() {
        let Ok(p) = std::env::var("PDF_PATH") else { return };
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("o.txt");
        let ours = super::extract_to(std::path::Path::new(&p), "x.pdf", &out, &dir.path().join("t"), 1 << 30);
        let poppler = std::process::Command::new("pdftotext").arg(&p).arg("-").output().map(|o| o.stdout.len()).unwrap_or(0);
        match ours {
            Ok(o) => println!("OURS: OK, {} bytes of text | poppler: {poppler} bytes | note {:?}", o.bytes, o.note),
            Err(e) => println!("OURS: ERROR {e} | poppler: {poppler} bytes"),
        }
    }
}
