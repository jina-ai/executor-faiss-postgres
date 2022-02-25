import os
import sys
from typing import BinaryIO, Generator, Optional, TextIO, Tuple, Union

import numpy as np
from jina.logging.logger import JinaLogger
from tqdm import tqdm

# TODO(winston): remove the following
RAND_STR_LEN = 20

BYTE_PADDING = 4
EMPTY_BYTES = b''
DUMP_DTYPE = np.float64
IDS_GENERATOR = Generator[str, None, None]
METAS_GENERATOR = Generator[Optional[bytes], None, None]
VECS_GENERATOR = Generator[Optional[np.ndarray], None, None]
GENERATOR_TYPE = Generator[
    Tuple[str, Union[np.ndarray, bytes], Optional[bytes]], None, None
]


def import_vectors(path: str, pea_id: str, logger: JinaLogger):
    """Import id and vectors
    :param path: the path to the dump
    :param pea_id: the id of the pea (as part of the shards)
    :return: the generators for the ids and for the vectors
    """
    logger.info(f'Importing ids and vectors from {path} for pea_id {pea_id}')
    path = os.path.join(path, pea_id)
    ids_gen = _ids_gen(path)
    vecs_gen = _vecs_gen(path)
    return ids_gen, vecs_gen


def _ids_gen(path: str) -> IDS_GENERATOR:
    with open(os.path.join(path, 'ids'), 'r') as ids_fh:
        for l in ids_fh:
            yield l.strip()


def _metas_gen(path: str) -> METAS_GENERATOR:
    with open(os.path.join(path, 'metas'), 'rb') as metas_fh:
        while True:
            next_size = metas_fh.read(BYTE_PADDING)
            next_size = int.from_bytes(next_size, byteorder=sys.byteorder)
            if next_size:
                meta = metas_fh.read(next_size)
                if meta != EMPTY_BYTES:
                    yield meta
                else:
                    yield None
            else:
                break


def _vecs_gen(path: str) -> VECS_GENERATOR:
    with open(os.path.join(path, 'vectors'), 'rb') as vectors_fh:
        while True:
            next_size = vectors_fh.read(BYTE_PADDING)
            next_size = int.from_bytes(next_size, byteorder=sys.byteorder)
            if next_size:
                vec_bytes = vectors_fh.read(next_size)
                if vec_bytes != EMPTY_BYTES:
                    vec = np.frombuffer(
                        vec_bytes,
                        dtype=DUMP_DTYPE,
                    )
                    yield vec
                else:
                    yield None
            else:
                break


def export_dump_streaming(
    path: str,
    shards: int,
    size: int,
    data: GENERATOR_TYPE,
    logger,
):
    """Export the data to a path, based on sharding,
    :param path: path to dump
    :param shards: the nr of shards this pea is part of
    :param size: total amount of entries
    :param data: the generator of the data (ids, vectors, metadata)
    """
    logger.info(f'Dumping {size} docs to {path} for {shards} shards')
    _handle_dump(data, path, shards, size)


def _handle_dump(
    data: GENERATOR_TYPE,
    path: str,
    shards: int,
    size: int,
):
    if not os.path.exists(path):
        os.makedirs(path)

    # directory must be empty to be safe
    if not os.listdir(path):
        size_per_shard = size // shards
        extra = size % shards
        shard_range = list(range(shards))
        for shard_id in shard_range:
            if shard_id == shard_range[-1]:
                size_this_shard = size_per_shard + extra
            else:
                size_this_shard = size_per_shard
            _write_shard_data(data, path, shard_id, size_this_shard)
    else:
        raise Exception(
            f'path for dump {path} contains data. Please empty. Not dumping...'
        )


def _write_shard_data(
    data: GENERATOR_TYPE,
    path: str,
    shard_id: int,
    size_this_shard: int,
):
    shard_path = os.path.join(path, str(shard_id))
    shard_docs_written = 0
    os.makedirs(shard_path)
    vectors_fp, metas_fp, ids_fp = _get_file_paths(shard_path)
    with open(vectors_fp, 'wb') as vectors_fh, open(metas_fp, 'wb') as metas_fh, open(
        ids_fp, 'w'
    ) as ids_fh:
        progress = tqdm(total=size_this_shard)
        while shard_docs_written < size_this_shard:
            _write_shard_files(data, ids_fh, metas_fh, vectors_fh)
            shard_docs_written += 1
            progress.update(1)
        progress.close()


def _write_shard_files(
    data: GENERATOR_TYPE,
    ids_fh: TextIO,
    metas_fh: BinaryIO,
    vectors_fh: BinaryIO,
):
    id_, vec, meta = next(data)
    # need to ensure compatibility to read time
    if vec is None:
        vec = EMPTY_BYTES
    if isinstance(vec, np.ndarray):
        if vec.dtype != DUMP_DTYPE:
            vec = vec.astype(DUMP_DTYPE)
        vec = vec.tobytes()
    vectors_fh.write(len(vec).to_bytes(BYTE_PADDING, sys.byteorder) + vec)
    if meta is None:
        meta = EMPTY_BYTES
    metas_fh.write(len(meta).to_bytes(BYTE_PADDING, sys.byteorder) + meta)
    ids_fh.write(id_ + '\n')


def _get_file_paths(shard_path: str):
    vectors_fp = os.path.join(shard_path, 'vectors')
    metas_fp = os.path.join(shard_path, 'metas')
    ids_fp = os.path.join(shard_path, 'ids')
    return vectors_fp, metas_fp, ids_fp
