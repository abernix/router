#[cfg(target_os = "macos")]
mod macos;

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use camino::Utf8PathBuf;
use xtask::*;

const INCLUDE: &[&str] = &["README.md", "LICENSE", "licenses.html"];
const JSON_SCHEMA_FILE: &str = "config-schema.json";

pub(crate) const TARGET_X86_64_MUSL_LINUX: &str = "x86_64-unknown-linux-musl";
pub(crate) const TARGET_X86_64_GNU_LINUX: &str = "x86_64-unknown-linux-gnu";
pub(crate) const TARGET_AARCH64_GNU_LINUX: &str = "aarch64-unknown-linux-gnu";
pub(crate) const TARGET_X86_64_WINDOWS: &str = "x86_64-pc-windows-msvc";
pub(crate) const TARGET_X86_64_MACOS: &str = "x86_64-apple-darwin";
pub(crate) const TARGET_ARM64_MACOS: &str = "aarch64-apple-darwin";

#[derive(Debug, clap::Parser)]
pub struct Package {
    /// Output tarball.
    #[clap(long)]
    output: Utf8PathBuf,

    #[cfg(target_os = "macos")]
    #[clap(flatten)]
    macos: macos::PackageMacos,

    #[clap(long)]
    target: Option<Target>,
}

impl Package {
    pub fn run(&self) -> Result<()> {
        let release_path = match &self.target {
            None => TARGET_DIR.join("release").join(RELEASE_BIN),
            Some(target) => TARGET_DIR
                .join(target.to_string())
                .join("release")
                .join(RELEASE_BIN),
        };

        ensure!(
            release_path.exists(),
            "Could not find binary at: {}",
            release_path
        );

        // We want to get the JSONSchema from the binary, which we can obtain
        // by running the binary with two positional arguments: `config` and `schema`
        // which will return the JSONSchema to stdout.  We want that to actually be
        // written to a file called `config-schema.json` in the current directory.
        let schema_output = std::process::Command::new(&release_path)
            .args(["config", "schema"])
            .output()
            .context("Failed to execute binary to get JSONSchema")?;

        if !schema_output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to get JSONSchema: {}",
                String::from_utf8_lossy(&schema_output.stderr)
            ));
        }

        let schema_path = Path::new(JSON_SCHEMA_FILE);
        std::fs::write(schema_path, &schema_output.stdout)
            .context(format!("Failed to write {}", JSON_SCHEMA_FILE))?;

        eprintln!("Generated JSONSchema at: {}", schema_path.display());

        #[cfg(target_os = "macos")]
        self.macos.run(&release_path)?;

        let output_path = if !self.output.exists() {
            if let Some(path) = self.output.parent() {
                let _ = std::fs::create_dir_all(path);
            }
            self.output.to_owned()
        } else if self.output.is_dir() {
            self.output.join(format!(
                "router-v{}-{}.tar.gz",
                *PKG_VERSION,
                self.target.clone().unwrap_or_default()
            ))
        } else {
            self.output.to_owned()
        };
        eprintln!("Creating tarball: {output_path}");
        let mut file = flate2::write::GzEncoder::new(
            std::io::BufWriter::new(
                std::fs::File::create(&output_path).context("could not create TGZ file")?,
            ),
            flate2::Compression::default(),
        );
        let mut ar = tar::Builder::new(&mut file);

        // Add the binary
        add_file_to_archive(&mut ar, &release_path, RELEASE_BIN)?;

        // Add the included files
        for path in INCLUDE {
            add_file_to_archive(&mut ar, &PKG_PROJECT_ROOT.join(path), path)?;
        }

        // Add the JSON Schema file
        add_file_to_archive(&mut ar, JSON_SCHEMA_FILE, JSON_SCHEMA_FILE)?;

        ar.finish().context("could not finish TGZ archive")?;

        Ok(())
    }
}

// Helper function to add a file to the archive
fn add_file_to_archive<P: AsRef<Path>, W: std::io::Write>(
    ar: &mut tar::Builder<W>,
    source_path: P,
    archive_name: &str,
) -> Result<()> {
    eprintln!("Adding {archive_name}...");
    ar.append_file(
        Path::new("dist").join(archive_name),
        &mut std::fs::File::open(source_path.as_ref())
            .context(format!("could not open {}", source_path.as_ref().display()))?,
    )
    .context(format!("could not add {} to TGZ archive", archive_name))?;
    Ok(())
}

#[derive(Debug, PartialEq, Clone, clap::ValueEnum)]
pub(crate) enum Target {
    #[value(name = "x86_64-unknown-linux-musl")]
    MuslLinux,
    #[value(name = "x86_64-unknown-linux-gnu")]
    GnuLinux,
    #[value(name = "aarch64-unknown-linux-gnu")]
    ArmLinux,
    #[value(name = "x86_64-pc-windows-msvc")]
    Windows,
    #[value(name = "x86_64-apple-darwin")]
    MacOS,
    #[value(name = "aarch64-apple-darwin")]
    ArmMacOS,
    #[value(skip)]
    Other,
}

impl Default for Target {
    fn default() -> Self {
        if cfg!(target_arch = "x86_64") {
            if cfg!(target_os = "windows") {
                Target::Windows
            } else if cfg!(target_os = "linux") {
                if cfg!(target_env = "gnu") {
                    Target::GnuLinux
                } else if cfg!(target_env = "musl") {
                    Target::MuslLinux
                } else {
                    Target::Other
                }
            } else if cfg!(target_os = "macos") {
                Target::MacOS
            } else {
                Target::Other
            }
        } else if cfg!(target_arch = "aarch64") {
            if cfg!(target_os = "linux") || cfg!(target_env = "gnu") {
                Target::ArmLinux
            } else if cfg!(target_os = "macos") {
                Target::ArmMacOS
            } else {
                Target::Other
            }
        } else {
            Target::Other
        }
    }
}

impl FromStr for Target {
    type Err = anyhow::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            TARGET_X86_64_MUSL_LINUX => Ok(Self::MuslLinux),
            TARGET_X86_64_GNU_LINUX => Ok(Self::GnuLinux),
            TARGET_AARCH64_GNU_LINUX => Ok(Self::ArmLinux),
            TARGET_X86_64_WINDOWS => Ok(Self::Windows),
            TARGET_X86_64_MACOS => Ok(Self::MacOS),
            TARGET_ARM64_MACOS => Ok(Self::ArmMacOS),
            _ => Ok(Self::Other),
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match &self {
            Target::MuslLinux => TARGET_X86_64_MUSL_LINUX,
            Target::GnuLinux => TARGET_X86_64_GNU_LINUX,
            Target::ArmLinux => TARGET_AARCH64_GNU_LINUX,
            Target::Windows => TARGET_X86_64_WINDOWS,
            Target::MacOS => TARGET_X86_64_MACOS,
            Target::ArmMacOS => TARGET_ARM64_MACOS,
            Target::Other => "unknown-target",
        };
        write!(f, "{msg}")
    }
}
