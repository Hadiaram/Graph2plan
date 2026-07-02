"""
HouseDiffusion inference wrapper for the Graph2plan Django interface.

Loads the pretrained model (RPLAN residential, 250k steps) and exposes
a generate() function that takes room types + adjacency edges and returns
polygon corner sequences in [0,255] pixel space.

The model is loaded once (singleton) on first call to generate().
"""

import os
import sys
import numpy as np
import torch as th

# ── Path setup ────────────────────────────────────────────────────────────────
HD_ROOT   = r'C:\Users\hmbashir\source\house_diffusion'
CKPT_PATH = os.path.join(HD_ROOT, 'ckpts', 'exp', 'model250000.pt')
MAX_POINTS = 100   # sequence length the pretrained model was built for

if HD_ROOT not in sys.path:
    sys.path.insert(0, HD_ROOT)

from house_diffusion.script_util import create_model_and_diffusion, update_arg_parser  # noqa: E402

# ── Type mappings ──────────────────────────────────────────────────────────────
# Graph2plan 0-indexed → RPLAN 1-indexed (what the pretrained model expects)
G2P_TO_RPLAN = {
    0:  1,   # LivingRoom  → Living Room
    1:  3,   # MasterRoom  → Bedroom
    2:  2,   # Kitchen     → Kitchen
    3:  4,   # Bathroom    → Bathroom
    4:  8,   # DiningRoom  → Dining Room
    5:  3,   # ChildRoom   → Bedroom
    6:  5,   # StudyRoom   → Study
    7:  3,   # SecondRoom  → Bedroom
    8:  3,   # GuestRoom   → Bedroom
    9:  6,   # Balcony     → Balcony
    10: 7,   # Entrance    → Entrance
    11: 10,  # Storage     → Storage
    12: 10,  # Wall-in     → Storage
}

# RPLAN type → Graph2plan display name (used by roomcolor() in the frontend)
RPLAN_TO_NAME = {
    1:  'LivingRoom',
    2:  'Kitchen',
    3:  'MasterRoom',
    4:  'Bathroom',
    5:  'StudyRoom',
    6:  'Balcony',
    7:  'Entrance',
    8:  'DiningRoom',
    10: 'Storage',
    11: 'InteriorDoor',
    12: 'FrontDoor',
    13: 'External',
}

# ── Singleton model ────────────────────────────────────────────────────────────
_model     = None
_diffusion = None
_device    = None


def _load_model():
    global _model, _diffusion, _device
    if _model is not None:
        return

    _device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f'[HouseDiffusion] Loading model on {_device}…')

    class Cfg:
        dataset = 'rplan'; analog_bit = False; use_checkpoint = False
        input_channels = 0; condition_channels = 0; out_channels = 0
        use_unet = False; num_channels = 128; learn_sigma = False
        diffusion_steps = 1000; noise_schedule = 'cosine'
        timestep_respacing = '100'; use_kl = False; predict_xstart = False
        rescale_timesteps = False; rescale_learned_sigmas = False
        target_set = 8; set_name = 'eval'

    cfg = Cfg()
    update_arg_parser(cfg)

    _model, _diffusion = create_model_and_diffusion(
        input_channels=cfg.input_channels,
        condition_channels=cfg.condition_channels,
        num_channels=cfg.num_channels,
        out_channels=cfg.out_channels,
        dataset=cfg.dataset,
        use_checkpoint=cfg.use_checkpoint,
        use_unet=cfg.use_unet,
        learn_sigma=cfg.learn_sigma,
        diffusion_steps=cfg.diffusion_steps,
        noise_schedule=cfg.noise_schedule,
        timestep_respacing=cfg.timestep_respacing,
        use_kl=cfg.use_kl,
        predict_xstart=cfg.predict_xstart,
        rescale_timesteps=cfg.rescale_timesteps,
        rescale_learned_sigmas=cfg.rescale_learned_sigmas,
        analog_bit=cfg.analog_bit,
        target_set=cfg.target_set,
        set_name=cfg.set_name,
    )

    state = th.load(CKPT_PATH, map_location='cpu')
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()
    print(f'[HouseDiffusion] Model ready ({sum(p.numel() for p in _model.parameters()):,} params)')


# ── Helpers ────────────────────────────────────────────────────────────────────
def _one_hot(idx, size):
    v = np.zeros(size, dtype=np.float32)
    v[min(int(idx), size - 1)] = 1.0
    return v


def _build_sequence(rplan_types, connected, corners_per_room):
    """Build the (MAX_POINTS, 94) house array and attention masks."""
    rows = []
    corner_bounds = []
    num_pts = 0

    for room_idx, (rtype, nc) in enumerate(zip(rplan_types, corners_per_room)):
        rt    = np.tile(_one_hot(rtype, 25),         (nc, 1))
        ri    = np.tile(_one_hot(room_idx + 1, 32),  (nc, 1))  # 1-indexed
        ci    = np.array([_one_hot(i, 32) for i in range(nc)], dtype=np.float32)
        pad   = np.ones((nc, 1),  dtype=np.float32)
        conns = np.array(
            [[num_pts + i, num_pts + (i + 1) % nc] for i in range(nc)],
            dtype=np.float32,
        )
        coords = np.zeros((nc, 2), dtype=np.float32)
        corner_bounds.append((num_pts, num_pts + nc))
        num_pts += nc
        rows.append(np.concatenate([coords, rt, ci, ri, pad, conns], axis=1))

    house   = np.concatenate(rows, axis=0)[:MAX_POINTS]
    padding = np.zeros((MAX_POINTS - len(house), 94), dtype=np.float32)
    house   = np.concatenate([house, padding], axis=0)   # (MAX_POINTS, 94)

    gen_mask  = np.ones((MAX_POINTS, MAX_POINTS), dtype=np.float32)
    gen_mask[:num_pts, :num_pts] = 0

    self_mask = np.ones((MAX_POINTS, MAX_POINTS), dtype=np.float32)
    for s, e in corner_bounds:
        if s < MAX_POINTS:
            self_mask[s:min(e, MAX_POINTS), s:min(e, MAX_POINTS)] = 0

    door_mask = np.ones((MAX_POINTS, MAX_POINTS), dtype=np.float32)
    for i, (s1, e1) in enumerate(corner_bounds):
        for j, (s2, e2) in enumerate(corner_bounds):
            if i != j and (min(i, j), max(i, j)) in connected:
                s1c, e1c = min(s1, MAX_POINTS), min(e1, MAX_POINTS)
                s2c, e2c = min(s2, MAX_POINTS), min(e2, MAX_POINTS)
                door_mask[s1c:e1c, s2c:e2c] = 0
                door_mask[s2c:e2c, s1c:e1c] = 0

    return house, gen_mask, self_mask, door_mask, corner_bounds, num_pts


def _to_tensor(arr, device):
    return th.tensor(arr, dtype=th.float32).unsqueeze(0).to(device)


# ── Public API ─────────────────────────────────────────────────────────────────
def generate(g2p_types, edges, n_corners=4):
    """
    Generate a floor plan layout using the pretrained HouseDiffusion model.

    Parameters
    ----------
    g2p_types : list[int]
        Room types in Graph2plan 0-based format (0=LivingRoom, 1=MasterRoom, …).
        Types not in G2P_TO_RPLAN mapping are skipped.
    edges : array-like of shape (E, 2) or (E, 3)
        Room adjacency.  Columns are [from, to] or [from, to, etype].
        Only door edges (etype==1, or all edges if etype is absent) are used.
    n_corners : int
        Corners per room for the initial sequence (default 4).
        The model generates the actual coordinates from scratch.

    Returns
    -------
    polygons   : list[list[[x, y]]]   — coords in [0, 255] pixel space
    room_types : list[int]            — RPLAN type codes (for frontend colour mapping)
    room_names : list[str]            — display names compatible with roomcolor()
    """
    _load_model()

    # ── Filter and convert types ──
    rplan_types  = []
    kept_indices = []
    for i, t in enumerate(g2p_types):
        rplan = G2P_TO_RPLAN.get(int(t))
        if rplan is None:
            continue
        rplan_types.append(rplan)
        kept_indices.append(i)

    n_rooms = len(rplan_types)
    if n_rooms == 0:
        return [], [], []

    # Remap edge indices to kept room indices
    idx_map = {orig: new for new, orig in enumerate(kept_indices)}
    connected = set()
    for e in edges:
        fi, ti = int(e[0]), int(e[1])
        # Accept both [from, to] and [from, to, etype] — only keep door/adjacency edges
        if len(e) >= 3 and int(e[2]) not in (1,):
            continue
        fi2 = idx_map.get(fi)
        ti2 = idx_map.get(ti)
        if fi2 is not None and ti2 is not None:
            connected.add((min(fi2, ti2), max(fi2, ti2)))

    corners_per_room = [n_corners] * n_rooms
    total_pts = sum(corners_per_room)
    if total_pts > MAX_POINTS:
        # Truncate to what the pretrained model supports
        print(f'[HouseDiffusion] Warning: {total_pts} total points exceeds '
              f'MAX_POINTS={MAX_POINTS}. Truncating room list.')
        max_rooms    = MAX_POINTS // n_corners
        rplan_types  = rplan_types[:max_rooms]
        corners_per_room = corners_per_room[:max_rooms]
        connected    = {(i, j) for i, j in connected if i < max_rooms and j < max_rooms}

    house, gen_mask, self_mask, door_mask, corner_bounds, num_pts = (
        _build_sequence(rplan_types, connected, corners_per_room)
    )

    # ── Build model kwargs ────────────────────────────────────────
    mk = {}
    for prefix in ('', 'syn_'):
        mk[f'{prefix}room_types']           = _to_tensor(house[:, 2:27],   _device)
        mk[f'{prefix}corner_indices']       = _to_tensor(house[:, 27:59],  _device)
        mk[f'{prefix}room_indices']         = _to_tensor(house[:, 59:91],  _device)
        mk[f'{prefix}connections']          = _to_tensor(house[:, 92:94],  _device)
        mk[f'{prefix}door_mask']            = _to_tensor(door_mask,        _device)
        mk[f'{prefix}self_mask']            = _to_tensor(self_mask,        _device)
        mk[f'{prefix}gen_mask']             = _to_tensor(gen_mask,         _device)
        mk[f'{prefix}src_key_padding_mask'] = (
            th.tensor(1.0 - house[:, 91], dtype=th.float32)
            .unsqueeze(0).to(_device)
        )

    # ── Run diffusion ─────────────────────────────────────────────
    # Iterate p_sample_loop_progressive directly so we only keep the final step.
    # p_sample_loop / ddim_sample_loop have hardcoded i>970/i>990 thresholds that
    # break when timestep_respacing reduces the total step count below 1000.
    print(f'[HouseDiffusion] Sampling {n_rooms} rooms ({num_pts} corners)…')
    last = None
    with th.no_grad():
        for out in _diffusion.p_sample_loop_progressive(
            _model,
            shape=(1, 2, MAX_POINTS),
            clip_denoised=True,
            model_kwargs=mk,
            analog_bit=False,
        ):
            last = out

    # last['sample']: (1, 2, MAX_POINTS) → (MAX_POINTS, 2)
    coords = last['sample'][0].permute(1, 0).cpu().numpy()   # in [-1, 1]

    # [-1, 1] → [0, 255]
    coords = np.clip((coords / 2.0 + 0.5) * 255.0, 0.0, 255.0)

    # ── Extract polygons ──────────────────────────────────────────
    polygons = []
    for s, e in corner_bounds:
        poly = coords[s:e].tolist()
        polygons.append([[round(x), round(y)] for x, y in poly])

    names = [RPLAN_TO_NAME.get(rt, 'Room') for rt in rplan_types]
    return polygons, rplan_types, names


# ── Guided generation (logic layer + retrieval-augmented init) ─────────────────
def generate_guided(g2p_types, edges, slot_polygons, T_start=500):
    """
    Generate a floor plan guided by slot bounding boxes from the logic layer.

    Instead of starting from pure Gaussian noise, the slot polygons are
    forward-diffused to timestep T_start and denoising starts from there.
    Between every denoising step, each room's corner coordinates are clipped
    to stay within its assigned slot bounds — enforcing the logical structure
    throughout generation.

    Parameters
    ----------
    g2p_types     : list[int]  — same as generate()
    edges         : array-like — same as generate()
    slot_polygons : list[list[[x,y]]] — one 4-corner polygon per room (logic layer output),
                    in [0, 255] canvas space.  Must be the same length as g2p_types
                    (after type filtering — the caller should only pass valid types).
    T_start       : int — forward-diffusion timestep to start from (0=no noise, 1000=pure noise).
                    500 retains the spatial skeleton while allowing model refinement.

    Returns
    -------
    polygons, rplan_types, room_names  — same format as generate().
    """
    _load_model()

    # ── Type filtering (same as generate) ────────────────────────────────────
    rplan_types  = []
    kept_indices = []
    for i, t in enumerate(g2p_types):
        rplan = G2P_TO_RPLAN.get(int(t))
        if rplan is None:
            continue
        rplan_types.append(rplan)
        kept_indices.append(i)

    n_rooms = len(rplan_types)
    if n_rooms == 0:
        return [], [], []

    idx_map = {orig: new for new, orig in enumerate(kept_indices)}
    connected = set()
    for e in edges:
        fi, ti = int(e[0]), int(e[1])
        if len(e) >= 3 and int(e[2]) not in (1,):
            continue
        fi2, ti2 = idx_map.get(fi), idx_map.get(ti)
        if fi2 is not None and ti2 is not None:
            connected.add((min(fi2, ti2), max(fi2, ti2)))

    # Filter slot polygons to match kept indices
    kept_slots = [slot_polygons[i] for i in kept_indices if i < len(slot_polygons)]
    if len(kept_slots) != n_rooms:
        # Slot count mismatch — fall back to unconstrained generation
        print('[HouseDiffusion] generate_guided: slot count mismatch, falling back.')
        return generate(g2p_types, edges)

    n_corners = 4   # slots are rectangles
    corners_per_room = [n_corners] * n_rooms
    total_pts = sum(corners_per_room)
    if total_pts > MAX_POINTS:
        max_rooms = MAX_POINTS // n_corners
        rplan_types  = rplan_types[:max_rooms]
        kept_slots   = kept_slots[:max_rooms]
        corners_per_room = corners_per_room[:max_rooms]
        connected    = {(i, j) for i, j in connected if i < max_rooms and j < max_rooms}

    house, gen_mask, self_mask, door_mask, corner_bounds, num_pts = (
        _build_sequence(rplan_types, connected, corners_per_room)
    )

    # ── Model kwargs (same as generate) ──────────────────────────────────────
    mk = {}
    for prefix in ('', 'syn_'):
        mk[f'{prefix}room_types']           = _to_tensor(house[:, 2:27],   _device)
        mk[f'{prefix}corner_indices']       = _to_tensor(house[:, 27:59],  _device)
        mk[f'{prefix}room_indices']         = _to_tensor(house[:, 59:91],  _device)
        mk[f'{prefix}connections']          = _to_tensor(house[:, 92:94],  _device)
        mk[f'{prefix}door_mask']            = _to_tensor(door_mask,        _device)
        mk[f'{prefix}self_mask']            = _to_tensor(self_mask,        _device)
        mk[f'{prefix}gen_mask']             = _to_tensor(gen_mask,         _device)
        mk[f'{prefix}src_key_padding_mask'] = (
            th.tensor(1.0 - house[:, 91], dtype=th.float32)
            .unsqueeze(0).to(_device)
        )

    # ── Build x_start from slot polygons (Task 6 initialisation) ─────────────
    # Convert slot corners from [0, 255] → [-1, 1] model space.
    slot_coords_model = np.zeros((MAX_POINTS, 2), dtype=np.float32)
    for room_idx, (s, e) in enumerate(corner_bounds):
        if room_idx >= len(kept_slots):
            break
        poly = np.array(kept_slots[room_idx], dtype=np.float32)   # (4, 2)
        # Normalise: [0, 255] → [-1, 1]
        poly_norm = (poly / 127.5) - 1.0
        n = min(e - s, len(poly_norm))
        slot_coords_model[s:s + n] = poly_norm[:n]

    # Shape: (1, 2, MAX_POINTS) — channel 0 = x, channel 1 = y
    x_start = th.tensor(
        slot_coords_model.T[np.newaxis],   # (1, 2, MAX_POINTS)
        dtype=th.float32,
    ).to(_device)

    # Forward-diffuse to T_start to add controlled noise
    t_tensor = th.tensor([T_start], device=_device, dtype=th.long)
    x_noisy  = _diffusion.q_sample(x_start, t_tensor)

    # ── Precompute slot bounds in [-1, 1] for per-step clipping ──────────────
    slot_bounds_model = []   # list of (x1, y1, x2, y2) per room in [-1, 1]
    for room_idx, slot in enumerate(kept_slots):
        poly = np.array(slot, dtype=np.float32)
        xs, ys = poly[:, 0], poly[:, 1]
        x1 = float((xs.min() / 127.5) - 1.0)
        x2 = float((xs.max() / 127.5) - 1.0)
        y1 = float((ys.min() / 127.5) - 1.0)
        y2 = float((ys.max() / 127.5) - 1.0)
        slot_bounds_model.append((x1, y1, x2, y2))

    # ── Denoising loop with between-step slot clipping ────────────────────────
    print(f'[HouseDiffusion] Guided sampling {n_rooms} rooms (T_start={T_start})…')
    last = None
    with th.no_grad():
        for out in _diffusion.p_sample_loop_progressive(
            _model,
            shape=(1, 2, MAX_POINTS),
            clip_denoised=True,
            model_kwargs=mk,
            analog_bit=False,
            noise=x_noisy,
        ):
            sample = out['sample']   # (1, 2, MAX_POINTS)
            # Clip each room's corners to its slot bounds
            for room_idx, (s, e) in enumerate(corner_bounds):
                if room_idx >= len(slot_bounds_model):
                    break
                sx1, sy1, sx2, sy2 = slot_bounds_model[room_idx]
                sample[0, 0, s:e].clamp_(sx1, sx2)   # x coordinates
                sample[0, 1, s:e].clamp_(sy1, sy2)   # y coordinates
            out['sample'] = sample
            last = out

    coords = last['sample'][0].permute(1, 0).cpu().numpy()   # in [-1, 1]
    coords = np.clip((coords / 2.0 + 0.5) * 255.0, 0.0, 255.0)

    polygons = []
    for s, e in corner_bounds:
        poly = coords[s:e].tolist()
        polygons.append([[round(x), round(y)] for x, y in poly])

    names = [RPLAN_TO_NAME.get(rt, 'Room') for rt in rplan_types]
    return polygons, rplan_types, names
