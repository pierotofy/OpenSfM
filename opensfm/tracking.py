import logging
import typing as t

import networkx as nx
import numpy as np
from opensfm import context, pymap
from opensfm.dataset_base import DataSetBase
from opensfm.unionfind import UnionFind
from opensfm.pymap import TracksManager


logger: logging.Logger = logging.getLogger(__name__)


def load_features(
    dataset: DataSetBase, images: t.List[str]
) -> t.Tuple[
    t.Dict[str, np.ndarray],
    t.Dict[str, np.ndarray],
    t.Dict[str, np.ndarray],
    t.Dict[str, np.ndarray],
]:
    logging.info("reading features")

    def load_one(im):
        features_data = dataset.load_features(im)
        if not features_data:
            return im, None, None, None, None
        features = features_data.points[:, :3]
        colors = features_data.colors
        segmentations = None
        instances = None
        if features_data.semantic:
            segmentations = features_data.semantic.segmentation
            if features_data.semantic.has_instances():
                instances = features_data.semantic.instances

        return im, features, colors, segmentations, instances

    # Use context.parallel_map for parallel loading
    results = context.parallel_map(load_one, images, dataset.config.get("processes", 1))

    features = {}
    colors = {}
    segmentations = {}
    instances = {}
    for im, feat, color, seg, inst in results:
        if feat is not None:
            features[im] = feat
            colors[im] = color
            if seg is not None:
                segmentations[im] = seg
            if inst is not None:
                instances[im] = inst

    return features, colors, segmentations, instances


def load_matches(
    dataset: DataSetBase, images: t.List[str]
) -> t.Dict[t.Tuple[str, str], t.List[t.Tuple[int, int]]]:
    matches = {}
    for im1 in images:
        try:
            im1_matches = dataset.load_matches(im1)
        except IOError:
            continue
        for im2 in im1_matches:
            if im2 in images:
                matches[im1, im2] = im1_matches[im2]
    return matches


def create_tracks_manager(
    features: t.Dict[str, np.ndarray],
    colors: t.Dict[str, np.ndarray],
    segmentations: t.Dict[str, np.ndarray],
    instances: t.Dict[str, np.ndarray],
    matches: t.Dict[t.Tuple[str, str], t.List[t.Tuple[int, int]]],
    min_length: int,
) -> TracksManager:
    """Link matches into tracks."""
    logger.debug("Merging features onto tracks")
    uf = UnionFind()
    for im1, im2 in matches:
        for f1, f2 in matches[im1, im2]:
            uf.union((im1, f1), (im2, f2))

    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    tracks = [t for t in sets.values() if _good_track(t, min_length)]
    logger.debug("Good tracks: {}".format(len(tracks)))

    NO_VALUE = pymap.Observation.NO_SEMANTIC_VALUE
    tracks_manager = pymap.TracksManager()
    for track_id, track in enumerate(tracks):
        for image, featureid in track:
            if image not in features:
                continue
            x, y, s = features[image][featureid]
            r, g, b = colors[image][featureid]
            segmentation, instance = (
                segmentations[image][featureid] if image in segmentations else NO_VALUE,
                instances[image][featureid] if image in instances else NO_VALUE,
            )
            obs = pymap.Observation(
                x, y, s, int(r), int(g), int(b), featureid, segmentation, instance
            )
            tracks_manager.add_observation(image, str(track_id), obs)
    return tracks_manager


def common_tracks(
    tracks_manager: pymap.TracksManager, im1: str, im2: str
) -> t.Tuple[t.List[str], np.ndarray, np.ndarray]:
    """List of tracks observed in both images.

    Args:
        tracks_manager: tracks manager
        im1: name of the first image
        im2: name of the second image

    Returns:
        tuple: tracks, feature from first image, feature from second image
    """
    t1 = tracks_manager.get_shot_observations(im1)
    t2 = tracks_manager.get_shot_observations(im2)
    tracks, p1, p2 = [], [], []
    for track, obs in t1.items():
        if track in t2:
            p1.append(obs.point)
            p2.append(t2[track].point)
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2


TPairTracks = t.Tuple[t.List[str], np.ndarray, np.ndarray]


def all_common_tracks_with_features(
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
) -> t.Dict[t.Tuple[str, str], TPairTracks]:
    tracks = all_common_tracks(
        tracks_manager, include_features=True, min_common=min_common
    )
    return t.cast(t.Dict[t.Tuple[str, str], TPairTracks], tracks)


def all_common_tracks_without_features(
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
) -> t.Dict[t.Tuple[str, str], t.List[str]]:
    tracks = all_common_tracks(
        tracks_manager, include_features=False, min_common=min_common
    )
    return t.cast(t.Dict[t.Tuple[str, str], t.List[str]], tracks)


def all_common_tracks(
    tracks_manager: pymap.TracksManager,
    include_features: bool = True,
    min_common: int = 50,
) -> t.Dict[t.Tuple[str, str], t.Union[TPairTracks, t.List[str]]]:
    """List of tracks observed by each image pair.

    Args:
        tracks_manager: tracks manager
        include_features: whether to include the features from the images
        min_common: the minimum number of tracks the two images need to have
            in common

    Returns:
        tuple: im1, im2 -> tuple: tracks, features from first image, features
        from second image
    """
    common_tracks = {}
    for (im1, im2), size in tracks_manager.get_all_pairs_connectivity().items():
        if size < min_common:
            continue

        tuples = tracks_manager.get_all_common_observations(im1, im2)
        if include_features:
            common_tracks[im1, im2] = (
                [v for v, _, _ in tuples],
                np.array([p.point for _, p, _ in tuples]),
                np.array([p.point for _, _, p in tuples]),
            )
        else:
            common_tracks[im1, im2] = [v for v, _, _ in tuples]
    return common_tracks


def np_all_common_tracks_with_features(
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
):
    """Same as all_common_tracks, but as a numpy structure + dictionary for indexing"""
    all_pairs = tracks_manager.get_all_pairs_connectivity().items()
    common_tracks_data = []
    common_tracks_index = {}
    rec_start = 0
    for pair, size in all_pairs:
        if size < min_common:
            continue
        
        tuples = tracks_manager.get_all_common_observations(pair[0], pair[1])
        num_tracks = len(tuples)
        rec_end = rec_start + num_tracks * 8

        for j in range(num_tracks):
            pair_track, p1, p2 = tuples[j]

            common_tracks_data.append(float(pair_track))
            common_tracks_data.append(p1.point[0])
            common_tracks_data.append(p1.point[1])
            common_tracks_data.append(1.0) # Homogeneous
            common_tracks_data.append(p2.point[0])
            common_tracks_data.append(p2.point[1])
            common_tracks_data.append(1.0) # Homogeneous
            common_tracks_data.append(0.0) # Align to 8bit boundary

        common_tracks_index[pair] = (num_tracks, rec_start, rec_end)
        rec_start = rec_end

    return np.reshape(common_tracks_data, newshape=(len(common_tracks_data), )), common_tracks_index


def _good_track(track: t.List[t.Tuple[str, int]], min_length: int) -> bool:
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    if len(images) != len(set(images)):
        return False
    return True


def as_weighted_graph(tracks_manager: pymap.TracksManager) -> nx.Graph:
    """Return the tracks manager as a weighted graph
    having shots a snodes and weighted by the # of
    common tracks between two nodes.
    """
    images = tracks_manager.get_shot_ids()
    image_graph = nx.Graph()
    for im in images:
        image_graph.add_node(im)
    for k, v in tracks_manager.get_all_pairs_connectivity().items():
        image_graph.add_edge(k[0], k[1], weight=v)
    return image_graph


def as_graph(tracks_manager: pymap.TracksManager) -> nx.Graph:
    """Return the tracks manager as a bipartite graph (legacy)."""
    tracks = tracks_manager.get_track_ids()
    images = tracks_manager.get_shot_ids()

    graph = nx.Graph()
    for track_id in tracks:
        graph.add_node(track_id, bipartite=1)
    for shot_id in images:
        graph.add_node(shot_id, bipartite=0)
    for track_id in tracks:
        for im, obs in tracks_manager.get_track_observations(track_id).items():
            graph.add_edge(
                im,
                track_id,
                feature=obs.point,
                feature_scale=obs.scale,
                feature_id=obs.id,
                feature_color=obs.color,
                feature_segmentation=obs.segmentation,
                feature_instance=obs.instance,
            )
    return graph
