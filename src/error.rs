//! A single error type for the whole crate.
//!
//! Each operation reports a specific error type, such as
//! [`PosetError`] or [`TransferSystemError`], which you can match on for
//! details. All of them convert into [`Error`], so a script that only wants
//! to stop and report a problem can use [`Result`] and the `?` operator
//! throughout:
//!
//! ```
//! use hccr::prelude::*;
//!
//! fn main() -> hccr::Result<()> {
//!     let l = Lattice::from_covers(["0", "a", "1"], [("0", "a"), ("a", "1")])?;
//!     let t = l.transfer_system_generated_by([l.edge("0", "a")?])?;
//!     println!("{t}");
//!     Ok(())
//! }
//! ```

use crate::cotransfer_lattice::CotransferSystemError;
use crate::label::LabelError;
use crate::lattice::{HorizontalJoinError, LatticeError};
use crate::model_structure::ModelStructureError;
use crate::morphism::{LatticeMapError, PosetMapError};
use crate::poset::PosetError;
use crate::transfer_lattice::TransferSystemError;
use crate::transfer_morphism::{CompositionMapError, TransferMapError};
use std::fmt;

#[cfg(feature = "groups")]
use crate::g_cotransfer_lattice::GCotransferSystemError;
#[cfg(feature = "groups")]
use crate::g_lattice::{GLatticeError, GTransferSystemError};
#[cfg(feature = "groups")]
use crate::g_transfer_morphism::GTransferMapError;
#[cfg(feature = "groups")]
use crate::subgroup_morphism::SubgroupMapError;

/// `Result<T, hccr::Error>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Any error produced by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// See [`PosetError`].
    Poset(PosetError),
    /// See [`LatticeError`].
    Lattice(LatticeError),
    /// See [`LabelError`].
    Label(LabelError),
    /// See [`HorizontalJoinError`].
    HorizontalJoin(HorizontalJoinError),
    /// See [`PosetMapError`].
    PosetMap(PosetMapError),
    /// See [`LatticeMapError`].
    LatticeMap(LatticeMapError),
    /// See [`TransferSystemError`].
    TransferSystem(TransferSystemError),
    /// See [`CotransferSystemError`].
    CotransferSystem(CotransferSystemError),
    /// See [`TransferMapError`].
    TransferMap(TransferMapError),
    /// See [`CompositionMapError`].
    CompositionMap(CompositionMapError),
    /// See [`ModelStructureError`].
    ModelStructure(ModelStructureError),
    /// See [`GLatticeError`].
    #[cfg(feature = "groups")]
    GLattice(GLatticeError),
    /// See [`GTransferSystemError`].
    #[cfg(feature = "groups")]
    GTransferSystem(GTransferSystemError),
    /// See [`GCotransferSystemError`].
    #[cfg(feature = "groups")]
    GCotransferSystem(GCotransferSystemError),
    /// See [`GTransferMapError`].
    #[cfg(feature = "groups")]
    GTransferMap(GTransferMapError),
    /// See [`SubgroupMapError`].
    #[cfg(feature = "groups")]
    SubgroupMap(SubgroupMapError),
    /// An error reported by GAP.
    #[cfg(feature = "groups")]
    Gap(String),
}

impl Error {
    fn inner(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Poset(error) => error,
            Self::Lattice(error) => error,
            Self::Label(error) => error,
            Self::HorizontalJoin(error) => error,
            Self::PosetMap(error) => error,
            Self::LatticeMap(error) => error,
            Self::TransferSystem(error) => error,
            Self::CotransferSystem(error) => error,
            Self::TransferMap(error) => error,
            Self::CompositionMap(error) => error,
            Self::ModelStructure(error) => error,
            #[cfg(feature = "groups")]
            Self::GLattice(error) => error,
            #[cfg(feature = "groups")]
            Self::GTransferSystem(error) => error,
            #[cfg(feature = "groups")]
            Self::GCotransferSystem(error) => error,
            #[cfg(feature = "groups")]
            Self::GTransferMap(error) => error,
            #[cfg(feature = "groups")]
            Self::SubgroupMap(error) => error,
            #[cfg(feature = "groups")]
            Self::Gap(_) => return None,
        })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.inner() {
            Some(error) => write!(f, "{error}"),
            #[cfg(feature = "groups")]
            None => match self {
                Self::Gap(message) => write!(f, "GAP error: {message}"),
                _ => unreachable!("only GAP errors have no inner error"),
            },
            #[cfg(not(feature = "groups"))]
            None => unreachable!("every error has an inner error"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner()
    }
}

macro_rules! from_errors {
    ($($variant:ident($error:ty)),* $(,)?) => {$(
        impl From<$error> for Error {
            fn from(error: $error) -> Self {
                Self::$variant(error)
            }
        }
    )*};
}

from_errors!(
    Poset(PosetError),
    Lattice(LatticeError),
    Label(LabelError),
    HorizontalJoin(HorizontalJoinError),
    PosetMap(PosetMapError),
    LatticeMap(LatticeMapError),
    TransferSystem(TransferSystemError),
    CotransferSystem(CotransferSystemError),
    TransferMap(TransferMapError),
    CompositionMap(CompositionMapError),
    ModelStructure(ModelStructureError),
);

#[cfg(feature = "groups")]
from_errors!(
    GLattice(GLatticeError),
    GTransferSystem(GTransferSystemError),
    GCotransferSystem(GCotransferSystemError),
    GTransferMap(GTransferMapError),
    SubgroupMap(SubgroupMapError),
);

#[cfg(feature = "groups")]
impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self::Gap(format!("{error:#}"))
    }
}
