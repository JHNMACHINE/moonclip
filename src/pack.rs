//! The pack file format: tensor data and the description of that data, in one
//! object.
//!
//! A pack used to be a bare concatenation of compressed blobs. Everything
//! needed to read it — where each tensor starts, its name, shape, dtype,
//! whether it is a delta and against which base — lived only in the global
//! manifest. That made a checkpoint two writes: the data, then the manifest
//! entry naming it. A process killed between them left data that was not
//! merely unreferenced but *unreadable*: there was nothing to check and
//! nothing to recover.
//!
//! Putting the description inside the pack removes the window rather than
//! recovering from it. One object, one write, atomic on both backends — so
//! "data without a description" stops being a state the system can reach.
//!
//! Layout:
//!
//! ```text
//! [ 32-byte header ][ compressed tensor blobs ][ descriptor JSON ]
//!   magic, version,
//!   descriptor offset + length
//! ```
//!
//! The order is forced: the descriptor records each tensor's offset, so it
//! cannot precede the blobs it describes. The header is fixed-size, so it can.
//! Reading the descriptor costs two small ranged reads and never touches the
//! blobs.
//!
//! Tensor offsets are relative to the start of the file and therefore begin at
//! [`HEADER_LEN`]. Readers take them from the manifest and never assume the
//! first tensor sits at zero, so the header cost them nothing.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::manifest::{CompressionAlgo, RankEntry};

/// Identifies a pack carrying its own description. A pack written before this
/// format existed starts with zstd's magic instead, so the check is also the
/// version test.
pub const MAGIC: &[u8; 8] = b"MOONPACK";

/// Bumped when the layout changes in a way older readers cannot follow.
pub const VERSION: u32 = 1;

/// magic(8) + version(4) + reserved(4) + descriptor offset(8) + length(8).
pub const HEADER_LEN: u64 = 32;

/// What one rank wrote, and the snapshot it belongs to.
///
/// Every rank's pack carries the snapshot-level fields as well as its own rank
/// entry. The duplication is deliberate: it means a snapshot can be rebuilt
/// from the packs alone, without a surviving rank 0 and without a separate
/// file that rank 0 alone knows to write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackDescriptor {
    pub snapshot_id: Uuid,
    pub step: u64,
    pub created_at: DateTime<Utc>,
    pub base_snapshot_id: Option<Uuid>,
    pub metadata: HashMap<String, String>,
    pub compression: CompressionAlgo,
    /// How many ranks the writing run had, so a reader can tell a complete
    /// snapshot from one whose other ranks never landed.
    pub world_size: u32,
    pub rank: RankEntry,
}

/// Build the fixed-size header that points at the descriptor.
pub fn encode_header(descriptor_offset: u64, descriptor_len: u64) -> Vec<u8> {
    let mut header = Vec::with_capacity(HEADER_LEN as usize);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes()); // reserved
    header.extend_from_slice(&descriptor_offset.to_le_bytes());
    header.extend_from_slice(&descriptor_len.to_le_bytes());
    debug_assert_eq!(header.len(), HEADER_LEN as usize);
    header
}

/// Where the descriptor lives, or `None` when this is not a pack of ours.
///
/// `None` is the ordinary answer for a pack written before this format, and
/// for a future version this build cannot read. Neither is an error: those
/// packs still load through the manifest, they simply cannot be recovered
/// without it.
pub fn decode_header(bytes: &[u8]) -> Option<(u64, u64)> {
    if bytes.len() < HEADER_LEN as usize || &bytes[..8] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    if version != VERSION {
        return None;
    }
    let offset = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
    let length = u64::from_le_bytes(bytes[24..32].try_into().ok()?);
    Some((offset, length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{TensorEntry, TensorStorage};

    fn rank_entry() -> RankEntry {
        RankEntry {
            rank: 0,
            tensors: vec![TensorEntry {
                name: "w".into(),
                shape: vec![8],
                dtype: "float32".into(),
                original_dtype: None,
                storage: TensorStorage::Full,
                alias_of: None,
                filename: None,
                offset: HEADER_LEN,
                compressed_size: 40,
                raw_size: 32,
                hash_raw: "abc".into(),
                hash_compressed: Some("def".into()),
                shuffled: false,
            }],
            pack_file: Some("snapshots/x/rank_0.pack".into()),
            total_compressed: 40,
            total_raw: 32,
            skipped_count: 0,
            delta_count: 0,
            full_count: 1,
        }
    }

    fn descriptor() -> PackDescriptor {
        PackDescriptor {
            snapshot_id: Uuid::new_v4(),
            step: 7,
            created_at: Utc::now(),
            base_snapshot_id: None,
            metadata: HashMap::from([("loss".into(), "0.1".into())]),
            compression: CompressionAlgo::Zstd { level: 3 },
            world_size: 1,
            rank: rank_entry(),
        }
    }

    #[test]
    fn a_header_round_trips() {
        let header = encode_header(4096, 512);
        assert_eq!(header.len(), HEADER_LEN as usize);
        assert_eq!(decode_header(&header), Some((4096, 512)));
    }

    /// The whole point of the format: read the description back out of the
    /// bytes, with nothing else to consult.
    #[test]
    fn a_descriptor_round_trips_through_a_pack() {
        let original = descriptor();
        let blobs = vec![9u8; 40];
        let json = serde_json::to_vec(&original).unwrap();

        let mut pack = encode_header(HEADER_LEN + blobs.len() as u64, json.len() as u64);
        pack.extend_from_slice(&blobs);
        pack.extend_from_slice(&json);

        let (offset, length) = decode_header(&pack).expect("our own header");
        let slice = &pack[offset as usize..(offset + length) as usize];
        let read: PackDescriptor = serde_json::from_slice(slice).unwrap();

        assert_eq!(read.snapshot_id, original.snapshot_id);
        assert_eq!(read.step, 7);
        assert_eq!(read.rank.tensors[0].name, "w");
        assert_eq!(read.rank.tensors[0].offset, HEADER_LEN);
        assert_eq!(read.metadata["loss"], "0.1");
    }

    /// A pack from before this format starts with zstd's magic. Recognising
    /// that as "not mine" rather than misreading it is what keeps old
    /// checkpoints loadable.
    #[test]
    fn a_legacy_pack_is_not_mistaken_for_one_of_ours() {
        let zstd_frame = [0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
        assert_eq!(decode_header(&zstd_frame), None);
    }

    #[test]
    fn a_truncated_or_future_pack_is_refused() {
        assert_eq!(decode_header(&[]), None);
        assert_eq!(decode_header(&MAGIC[..4]), None, "shorter than the header");

        let mut future = encode_header(1, 1);
        future[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        assert_eq!(decode_header(&future), None, "a version we cannot read");
    }
}
