//! Everything needed for everyday calculations, in one import.
//!
//! ```
//! use hccr::prelude::*;
//!
//! let c3 = chain(3);
//! let l = horizontal_join([&c3, &c3, &c3])?;
//! assert_eq!(l.transfer_systems().size(), 298);
//! # Ok::<(), hccr::Error>(())
//! ```

pub use crate::cotransfer_lattice::{CotransferLattice, CotransferSystem};
pub use crate::label::Label;
pub use crate::lattice::{Lattice, boolean, chain, horizontal_join, product};
pub use crate::model_structure::ModelStructure;
pub use crate::morphism::{LatticeMap, MonotoneMap, PosetMap};
pub use crate::poset::{Edge, EdgeSet, ElementId, Poset};
pub use crate::tikz::ToTikz;
pub use crate::transfer_lattice::{TransferLattice, TransferPoset, TransferSystem};
pub use crate::{Error, Result};

#[cfg(feature = "groups")]
pub use crate::g_cotransfer_lattice::{GCotransferLattice, GCotransferSystem};
#[cfg(feature = "groups")]
pub use crate::g_lattice::{GLattice, GTransferLattice, GTransferSystem, SubgroupGLattice};
#[cfg(feature = "groups")]
pub use crate::group_theory::GapSubgroup;
#[cfg(feature = "groups")]
pub use crate::subgroup_morphism::SubgroupMaps;
