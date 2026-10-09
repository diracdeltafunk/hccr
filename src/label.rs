//! Element labels.
//!
//! Every element of a [`crate::poset::Poset`] or [`crate::lattice::Lattice`]
//! carries a [`Label`]: a printable name that is unique within that poset.
//! Labels are *dynamic*: there is a single label type, and each label records
//! its own shape (an integer, a string, a tuple of labels, a set of labels,
//! or a subgroup) as a value, much as Python or Sage values do.
//!
//! Ordinary Rust values convert into labels with [`From`]/[`Into`], so you
//! rarely build a [`Label`] by hand:
//!
//! ```
//! use hccr::label::Label;
//!
//! assert_eq!(Label::from(3), Label::Int(3));
//! assert_eq!(Label::from((1, "a")).to_string(), "(1, a)");
//! ```
//!
//! Going the other way is fallible, because a label might have a different
//! shape than you expect:
//!
//! ```
//! use hccr::label::Label;
//!
//! let label = Label::from((1, "a"));
//! let (n, s): (usize, String) = (&label).try_into()?;
//! assert_eq!((n, s.as_str()), (1, "a"));
//! assert!(i64::try_from(&label).is_err());
//! # Ok::<(), hccr::label::LabelError>(())
//! ```

use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "groups")]
use crate::group_theory::GapSubgroup;

/// A printable, hashable, totally ordered element label.
///
/// The composite variants store their contents behind an [`Arc`], so cloning
/// a label is cheap no matter how large it is.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Label {
    /// An integer.
    Int(i64),
    /// A string.
    Str(Arc<str>),
    /// A finite sequence of labels, such as `(1, a)`.
    Tuple(Arc<[Label]>),
    /// A finite set of labels, such as `{0, 2}`.
    Set(Arc<BTreeSet<Label>>),
    /// A subgroup in GAP's enumeration of a subgroup lattice.
    #[cfg(feature = "groups")]
    Subgroup(GapSubgroup),
}

impl Label {
    /// Constructs a tuple label from its entries.
    pub fn tuple<I>(entries: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Label>,
    {
        Self::Tuple(entries.into_iter().map(Into::into).collect())
    }

    /// Constructs a set label from its members. Duplicates are discarded.
    pub fn set<I>(members: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Label>,
    {
        Self::Set(Arc::new(members.into_iter().map(Into::into).collect()))
    }

    /// Returns the integer, if this is an integer label.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the string, if this is a string label.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the entries, if this is a tuple label.
    pub fn as_tuple(&self) -> Option<&[Label]> {
        match self {
            Self::Tuple(entries) => Some(entries),
            _ => None,
        }
    }

    /// Returns the members, if this is a set label.
    pub fn as_set(&self) -> Option<&BTreeSet<Label>> {
        match self {
            Self::Set(members) => Some(members),
            _ => None,
        }
    }

    /// Returns the subgroup, if this is a subgroup label.
    #[cfg(feature = "groups")]
    pub fn as_subgroup(&self) -> Option<GapSubgroup> {
        match self {
            Self::Subgroup(subgroup) => Some(*subgroup),
            _ => None,
        }
    }

    fn shape(&self) -> &'static str {
        match self {
            Self::Int(_) => "an integer",
            Self::Str(_) => "a string",
            Self::Tuple(_) => "a tuple",
            Self::Set(_) => "a set",
            #[cfg(feature = "groups")]
            Self::Subgroup(_) => "a subgroup",
        }
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn write_list<'a>(
            f: &mut fmt::Formatter<'_>,
            open: &str,
            close: &str,
            items: impl Iterator<Item = &'a Label>,
        ) -> fmt::Result {
            f.write_str(open)?;
            for (index, item) in items.enumerate() {
                if index > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{item}")?;
            }
            f.write_str(close)
        }

        match self {
            Self::Int(value) => write!(f, "{value}"),
            Self::Str(value) => f.write_str(value),
            Self::Tuple(entries) => write_list(f, "(", ")", entries.iter()),
            Self::Set(members) => write_list(f, "{", "}", members.iter()),
            #[cfg(feature = "groups")]
            Self::Subgroup(subgroup) => write!(f, "{subgroup}"),
        }
    }
}

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str(value) => write!(f, "{value:?}"),
            other => write!(f, "{other}"),
        }
    }
}

/// A label did not have the shape that a conversion expected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelError {
    /// A description of the expected shape, such as `"an integer"`.
    pub expected: &'static str,
    /// The label that was supplied.
    pub found: Label,
}

impl fmt::Display for LabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected {} label, found {} label `{}`",
            self.expected,
            self.found.shape(),
            self.found
        )
    }
}

impl std::error::Error for LabelError {}

fn mismatch(expected: &'static str, found: &Label) -> LabelError {
    LabelError {
        expected,
        found: found.clone(),
    }
}

impl From<&Label> for Label {
    fn from(label: &Label) -> Self {
        label.clone()
    }
}

macro_rules! int_labels {
    ($($int:ty),*) => {$(
        impl From<$int> for Label {
            fn from(value: $int) -> Self {
                Self::Int(i64::try_from(value).expect("integer label does not fit in an i64"))
            }
        }

        impl From<&$int> for Label {
            fn from(value: &$int) -> Self {
                Self::from(*value)
            }
        }

        impl TryFrom<&Label> for $int {
            type Error = LabelError;

            fn try_from(label: &Label) -> Result<Self, Self::Error> {
                label
                    .as_int()
                    .and_then(|value| <$int>::try_from(value).ok())
                    .ok_or_else(|| mismatch(concat!("an integer fitting in ", stringify!($int)), label))
            }
        }

        impl TryFrom<Label> for $int {
            type Error = LabelError;

            fn try_from(label: Label) -> Result<Self, Self::Error> {
                Self::try_from(&label)
            }
        }
    )*};
}

int_labels!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl From<&str> for Label {
    fn from(value: &str) -> Self {
        Self::Str(Arc::from(value))
    }
}

impl From<String> for Label {
    fn from(value: String) -> Self {
        Self::Str(Arc::from(value))
    }
}

impl From<&String> for Label {
    fn from(value: &String) -> Self {
        Self::from(value.as_str())
    }
}

impl From<char> for Label {
    fn from(value: char) -> Self {
        Self::from(value.to_string())
    }
}

impl From<Arc<str>> for Label {
    fn from(value: Arc<str>) -> Self {
        Self::Str(value)
    }
}

impl TryFrom<&Label> for String {
    type Error = LabelError;

    fn try_from(label: &Label) -> Result<Self, Self::Error> {
        label
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| mismatch("a string", label))
    }
}

impl TryFrom<Label> for String {
    type Error = LabelError;

    fn try_from(label: Label) -> Result<Self, Self::Error> {
        Self::try_from(&label)
    }
}

macro_rules! tuple_labels {
    ($len:literal; $($name:ident $index:tt),+) => {
        impl<$($name),+> From<($($name,)+)> for Label
        where
            $($name: Into<Label>),+
        {
            fn from(value: ($($name,)+)) -> Self {
                Self::Tuple(Arc::from([$(value.$index.into()),+]))
            }
        }

        impl<$($name),+> TryFrom<&Label> for ($($name,)+)
        where
            $($name: for<'a> TryFrom<&'a Label, Error = LabelError>),+
        {
            type Error = LabelError;

            fn try_from(label: &Label) -> Result<Self, Self::Error> {
                let entries = label
                    .as_tuple()
                    .filter(|entries| entries.len() == $len)
                    .ok_or_else(|| mismatch(concat!("a tuple of length ", $len), label))?;
                Ok(($($name::try_from(&entries[$index])?,)+))
            }
        }

        impl<$($name),+> TryFrom<Label> for ($($name,)+)
        where
            $($name: for<'a> TryFrom<&'a Label, Error = LabelError>),+
        {
            type Error = LabelError;

            fn try_from(label: Label) -> Result<Self, Self::Error> {
                Self::try_from(&label)
            }
        }
    };
}

tuple_labels!(1; A 0);
tuple_labels!(2; A 0, B 1);
tuple_labels!(3; A 0, B 1, C 2);
tuple_labels!(4; A 0, B 1, C 2, D 3);
tuple_labels!(5; A 0, B 1, C 2, D 3, E 4);
tuple_labels!(6; A 0, B 1, C 2, D 3, E 4, F 5);

impl<T: Into<Label>> From<Vec<T>> for Label {
    fn from(value: Vec<T>) -> Self {
        Self::tuple(value)
    }
}

impl<T: Into<Label>, const N: usize> From<[T; N]> for Label {
    fn from(value: [T; N]) -> Self {
        Self::tuple(value)
    }
}

impl<T: Clone + Into<Label>> From<&[T]> for Label {
    fn from(value: &[T]) -> Self {
        Self::tuple(value.iter().cloned())
    }
}

impl<T: Into<Label>> From<BTreeSet<T>> for Label {
    fn from(value: BTreeSet<T>) -> Self {
        Self::set(value)
    }
}

impl<T: Into<Label>, S> From<HashSet<T, S>> for Label {
    fn from(value: HashSet<T, S>) -> Self {
        Self::set(value)
    }
}

impl<T> TryFrom<&Label> for Vec<T>
where
    T: for<'a> TryFrom<&'a Label, Error = LabelError>,
{
    type Error = LabelError;

    fn try_from(label: &Label) -> Result<Self, Self::Error> {
        label
            .as_tuple()
            .ok_or_else(|| mismatch("a tuple", label))?
            .iter()
            .map(T::try_from)
            .collect()
    }
}

impl<T> TryFrom<&Label> for BTreeSet<T>
where
    T: Ord + for<'a> TryFrom<&'a Label, Error = LabelError>,
{
    type Error = LabelError;

    fn try_from(label: &Label) -> Result<Self, Self::Error> {
        label
            .as_set()
            .ok_or_else(|| mismatch("a set", label))?
            .iter()
            .map(T::try_from)
            .collect()
    }
}

#[cfg(feature = "groups")]
impl From<GapSubgroup> for Label {
    fn from(value: GapSubgroup) -> Self {
        Self::Subgroup(value)
    }
}

#[cfg(feature = "groups")]
impl TryFrom<&Label> for GapSubgroup {
    type Error = LabelError;

    fn try_from(label: &Label) -> Result<Self, Self::Error> {
        label
            .as_subgroup()
            .ok_or_else(|| mismatch("a subgroup", label))
    }
}
