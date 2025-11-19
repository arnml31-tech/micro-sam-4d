import numpy as np
import threading
from qtpy import QtWidgets
from qtpy.QtGui import QKeySequence, QCursor
from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import QFileDialog
from napari.utils.notifications import show_info
from micro_sam.sam_annotator.annotator_3d import Annotator3d
from micro_sam.sam_annotator._state import AnnotatorState
from pathlib import Path
import json
from micro_sam import instance_segmentation
from .util import _load_amg_state, _load_is_state
from . import util as _vutil
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation
from skimage.transform import resize as _sk_resize
class TimestepToolsWidget(QtWidgets.QWidget):
    """Simple UI widget providing 4D timestep operations for manual workflows."""
    def __init__(self, annotator, parent=None):
        super().__init__(parent)
        self._annotator = annotator
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        btn_segment = QtWidgets.QPushButton("Segment All Timesteps")
        btn_commit = QtWidgets.QPushButton("Commit All Timesteps")

        btn_segment.clicked.connect(lambda: self._safe_call(self._annotator.segment_all_timesteps))
        btn_commit.clicked.connect(lambda: self._safe_call(self._annotator.commit_all_timesteps))

        layout.addWidget(btn_segment)
        layout.addWidget(btn_commit)

    def _safe_call(self, fn):
        try:
            fn()
        except Exception as e:
            try:
                show_info(f"Operation failed: {e}")
            except Exception:
                print(f"Operation failed: {e}")

def _select_array_from_zarr_group(f):
    """Pick a zarr.Array-like child from a zarr.Group.

    Preference order:
      - dataset named 'features'
      - common alternate names
      - first array-like child at depth 1
      - first array-like child inside a top-level group at depth 2

    Returns the array-like object or None if none found.
    """
    try:
        # Prefer explicit 'features'
        if "features" in f:
            return f["features"]
    except Exception:
        pass

    # common alternative names
    for alt in ("feats", "features_0", "features0", "arr", "data"):
        try:
            if alt in f:
                return f[alt]
        except Exception:
            pass

    # first pass: find first array-like child at depth 1
    try:
        for name, obj in f.items():
            try:
                if hasattr(obj, "ndim") or hasattr(obj, "shape"):
                    return obj
            except Exception:
                # if obj is a Group, try depth-2
                try:
                    for cname, cobj in obj.items():
                        if hasattr(cobj, "ndim") or hasattr(cobj, "shape"):
                            return cobj
                except Exception:
                    continue
    except Exception:
        pass

    return None



class TimestepEmbeddingManager:
    """Manage lazy loading of per-timestep embeddings stored as zarr files.

    This keeps only one materialized embedding in memory (the most recently
    used timestep). When the timestep changes we load the matching
    `embeddings_t{t}.zarr` lazily (in a background thread) and activate it on
    the parent annotator by calling `_ensure_embeddings_active_for_t(t)`.
    """

    def __init__(self, annotator, embeddings_dir: str | Path | None = None, lazy: bool = True):
        self.annotator = annotator
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir is not None else None
        self.lazy = bool(lazy)
        self._lock = threading.Lock()
        # cache for the currently materialized timestep
        self.cached_t = None
        self.cached_entry = None
        self.cached_path = None

    def set_embeddings_dir(self, path: str | Path):
        self.embeddings_dir = Path(path)

    def get_current_embedding(self):
        return self.cached_entry

    def on_timestep_changed(self, t: int):
        """Callback for timestep changes; triggers embedding loading and 4D-aware point updates."""
        self.current_timestep = t

        # --- update 4D point prompts ---
        try:
            if hasattr(self.annotator, "point_prompts_4d"):
                pts_t = np.array(self.annotator.point_prompts_4d.get(t, np.empty((0, 3))))
                if "point_prompts" in self.annotator._viewer.layers:
                    self.annotator._viewer.layers["point_prompts"].data = pts_t
        except Exception as e:
            print(f"[WARN] Failed updating 4D point prompts at timestep {t}: {e}")

        # --- background embedding loading ---
        try:
            thread = threading.Thread(
                target=self.load_embedding_for_timestep,
                args=(int(t),),
                daemon=True,
            )
            thread.start()
        except Exception:
            try:
                self.load_embedding_for_timestep(int(t))
            except Exception:
                pass

    def _make_zarr_path_for_t(self, t: int):
        # prefer explicit last dir from annotator
        p = None
        if getattr(self.annotator, "_last_embeddings_dir", None):
            p = Path(self.annotator._last_embeddings_dir)
        if p is None and self.embeddings_dir is not None:
            p = self.embeddings_dir
        if p is None:
            return None
        cand = p / f"embeddings_t{t}.zarr"
        if cand.exists():
            return cand
        # also accept alternative naming
        cand2 = p / f"t{t}.zarr"
        if cand2.exists():
            return cand2
        return None

    def load_embedding_for_timestep(self, t: int):
        """Load (lazily) the embedding for timestep `t` and activate it.

        This will materialize the zarr store (open it) and call
        `annotator._ensure_embeddings_active_for_t(t)` so the global state uses
        the newly-loaded embeddings. Only one embedding is kept materialized;
        the previous one is released (replaced with a path placeholder) if it
        was loaded from disk by this manager.
        """
        with self._lock:
            # If we already have the requested timestep cached, nothing to do
            if self.cached_t == int(t) and self.cached_entry is not None:
                return self.cached_entry

            # If annotator already knows about an embedding entry for t, prefer that
            try:
                existing = self.annotator.embeddings_4d.get(int(t))
                if existing is not None and isinstance(existing, dict) and "features" in existing:
                    # materialized already by other code; adopt as cache
                    self._release_cached_if_needed(exclude_t=int(t))
                    self.cached_t = int(t)
                    self.cached_entry = existing
                    self.cached_path = existing.get("path") if isinstance(existing, dict) else None
                    # ensure annotator binds it
                    try:
                        self.annotator._ensure_embeddings_active_for_t(int(t))
                    except Exception:
                        pass
                    return self.cached_entry
            except Exception:
                pass

            # Try to resolve zarr path
            zpath = self._make_zarr_path_for_t(int(t))
            if zpath is None:
                # Fallback: if annotator.embeddings_4d contains an entry (e.g., mapping produced by set_embeddings_folder), use it
                entry = self.annotator.embeddings_4d.get(int(t))
                if entry is not None:
                    # trigger annotator's activation path (it will materialize if needed)
                    try:
                        self.annotator._ensure_embeddings_active_for_t(int(t))
                    except Exception:
                        pass
                    return entry
                return None

            # Mark annotator mapping with a lazy path so other parts can see it
            try:
                self.annotator.embeddings_4d[int(t)] = {"path": str(zpath)}
            except Exception:
                pass

            # Materialize (open) the zarr in background thread context (we're already in a thread)
            try:
                import zarr as _zarr
                f = _zarr.open(str(zpath), mode="r")
                # Pick a suitable array-like dataset from the zarr group.
                feats = _select_array_from_zarr_group(f)
                if feats is None:
                    try:
                        show_info(f"No suitable array dataset found inside {zpath}; cannot load embeddings for timestep {t}.")
                    except Exception:
                        pass
                    return None
                attrs = getattr(feats, "attrs", {}) or {}
                input_size = attrs.get("input_size")
                original_size = attrs.get("original_size")
                if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                    try:
                        inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                        input_size = input_size or inferred
                        original_size = original_size or inferred
                    except Exception:
                        input_size = input_size or None
                        original_size = original_size or None

                entry = {"features": feats, "input_size": input_size, "original_size": original_size, "path": str(zpath)}

                # release old cached if it was loaded by manager
                self._release_cached_if_needed()

                # cache this one
                self.cached_t = int(t)
                self.cached_entry = entry
                self.cached_path = str(zpath)

                # store in annotator mapping and activate
                try:
                    self.annotator.embeddings_4d[int(t)] = entry
                except Exception:
                    pass
                try:
                    # call annotator activation so AnnotatorState binds this embedding
                    self.annotator._ensure_embeddings_active_for_t(int(t))
                except Exception:
                    pass
                return entry
            except Exception:
                try:
                    show_info(f"Failed to load embeddings for timestep {t} from {zpath}")
                except Exception:
                    pass
                return None

    def _release_cached_if_needed(self, exclude_t: int | None = None):
        """Release the currently cached embedding if it was loaded from disk by this manager.

        If the previous entry was materialized from a zarr on-disk store, replace
        it in `annotator.embeddings_4d` with a lightweight {'path': ...} mapping
        so memory can be reclaimed.
        """
        try:
            if self.cached_t is None:
                return
            if exclude_t is not None and int(self.cached_t) == int(exclude_t):
                return
            # Only replace if we have a cached_path (i.e., loaded from disk)
            if self.cached_path and self.cached_t is not None:
                try:
                    # replace materialized entry with a path-only placeholder
                    self.annotator.embeddings_4d[int(self.cached_t)] = {"path": str(self.cached_path)}
                except Exception:
                    try:
                        del self.annotator.embeddings_4d[int(self.cached_t)]
                    except Exception:
                        pass
            # clear local references so zarr can be freed by GC
            self.cached_t = None
            self.cached_entry = None
            self.cached_path = None
        except Exception:
            pass



class MicroSAM4DAnnotator(Annotator3d):
    """
    4D annotator for (T, Z, Y, X) time-series data.

    This class keeps a persistent 4D image layer (`raw_4d`) and a persistent
    4D labels layer (`committed_objects_4d`). Per-timestep interactive 3D
    layers (editable views) are created once and updated in-place when the
    Napari time slider (dims.current_step[0]) changes. Committing a
    segmentation writes the result back into the 4D label array and refreshes
    the visible 3D view.
    """

    def __init__(self, viewer):
        # mark this instance as a 4D annotator so base class will create _4d
        # container layers instead of per-timestep 3D layers during init
        try:
            # set flag before calling base init so _AnnotatorBase can detect 4D
            self._is_4d = True
        except Exception:
            pass
        super().__init__(viewer)
        self.current_timestep = 0
        self.image_4d = None
        self.use_preview = True
        # 4D arrays (T, Z, Y, X)
        self.segmentation_4d = None
        self.auto_segmentation_4d = None
        self.current_object_4d = None
        # 4D-aware point prompts: dict mapping timestep -> points array
        self.point_prompts_4d = {}
        self.n_timesteps = 0
        # small per-timestep cache (optional)
        self._segmentation_cache = None
        # per-timestep embeddings cache: mapping t -> embedding dict or lazy entry
        self.embeddings_4d = {}
        # flags for background materialization (t -> bool)
        self._embedding_loading = {}
        # currently-active timestep whose embeddings are bound to AnnotatorState
        self._active_embedding_t = None
        # track which timesteps we've shown an embedding info message for
        self._reported_embedding_info_t = set()

        # remember last directory where embeddings were saved/loaded
        self._last_embeddings_dir = None

        # Desired layer order (bottom to top)
        self._desired_layer_order = [
            "raw_4d",
            "current_object_4d",
            "auto_segmentation_4d",
            "committed_objects_4d",
            "point_prompts",
            "remap_points"
        ]

        # Timestep embedding manager for lazy per-timestep zarr loading
        try:
            self.timestep_embedding_manager = TimestepEmbeddingManager(self)
        except Exception:
            self.timestep_embedding_manager = None

                # Add small embedding controls to the annotator dock (Compute embeddings current/all T)
        try:
            emb_widget = QtWidgets.QWidget()
            emb_layout = QtWidgets.QVBoxLayout()
            emb_widget.setLayout(emb_layout)

            # Row with two compute buttons
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout()
            row.setLayout(row_layout)
            btn_current = QtWidgets.QPushButton("Compute embeddings (current T)")
            btn_all = QtWidgets.QPushButton("Compute embeddings (all T)")
            row_layout.addWidget(btn_current)
            row_layout.addWidget(btn_all)

            # Only add the compute buttons row
            emb_layout.addWidget(row)

            # Add ID remapper widget
            try:
                from ._widgets import IdRemapperWidget
                remapper = IdRemapperWidget(self)
                emb_layout.addWidget(remapper)
            except Exception:
                pass
            # Removed timestep list widget and activation controls

            def _compute_current():
                try:
                    t = int(getattr(self, "current_timestep", 0) or 0)
                    show_info(f"Computing embeddings for timestep {t} — this may take a while.")
                    self.compute_embeddings_for_timestep(t)
                    show_info("Embeddings computed for current timestep.")
                except Exception as e:
                    print(f"Failed to compute embeddings for timestep {t}: {e}")

            def _compute_all():
                try:
                    show_info("Computing embeddings for all timesteps — this may take a long time.")
                    self.compute_embeddings_for_all_timesteps()
                    show_info("Embeddings computed for all timesteps.")
                except Exception as e:
                    print(f"Failed to compute embeddings for all timesteps: {e}")

            btn_current.clicked.connect(lambda _: _compute_current())
            btn_all.clicked.connect(lambda _: _compute_all())

            # Add save and load embeddings buttons
            save_load_row = QtWidgets.QWidget()
            save_load_layout = QtWidgets.QHBoxLayout()
            save_load_row.setLayout(save_load_layout)
            btn_save = QtWidgets.QPushButton("Save embeddings")
            btn_load = QtWidgets.QPushButton("Load embeddings from directory")
            save_load_layout.addWidget(btn_save)
            save_load_layout.addWidget(btn_load)
            emb_layout.addWidget(save_load_row)

            def _save_embeddings():
                try:
                    # Select directory to save embeddings
                    directory = QFileDialog.getExistingDirectory(
                        None, 
                        "Select Directory to Save Embeddings",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    show_info(f"Computing and saving embeddings for all timesteps to {directory} — this may take a long time.")
                    
                    # Compute embeddings for all timesteps and save them
                    save_path = Path(directory)
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    for t in range(self.n_timesteps):
                        filename = save_path / f"t{t}.zarr"
                        show_info(f"Computing embeddings for timestep {t}...")
                        self.compute_embeddings_for_timestep(
                            t=t, 
                            save_path=str(filename)
                        )
                    
                    # Store the directory path for future reference
                    self._last_embeddings_dir = str(save_path)
                    show_info(f"All embeddings saved to {directory}")
                    
                except Exception as e:
                    print(f"Failed to save embeddings: {e}")
                    show_info(f"Failed to save embeddings: {e}")

            def _load_embeddings():
                try:
                    # Select directory containing embeddings
                    directory = QFileDialog.getExistingDirectory(
                        None,
                        "Select Directory Containing Embeddings",
                        str(Path.home())
                    )
                    if not directory:
                        return
                    
                    directory_path = Path(directory)
                    
                    # Check for t0, t1, t2, ... files
                    embedding_files = sorted(directory_path.glob("t*.zarr"))
                    if not embedding_files:
                        show_info(f"No embedding files (t0.zarr, t1.zarr, ...) found in {directory}")
                        return
                    
                    # Store lazy loading information
                    self._last_embeddings_dir = str(directory_path)
                    
                    # Initialize embeddings_4d with lazy entries (just store paths)
                    if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
                        self.embeddings_4d = {}
                    
                    # Only load embeddings for timesteps that exist in the viewer
                    loaded_count = 0
                    for t in range(min(self.n_timesteps, len(embedding_files))):
                        filename = directory_path / f"t{t}.zarr"
                        if filename.exists():
                            # Store path for lazy loading
                            self.embeddings_4d[t] = {"path": str(filename)}
                            loaded_count += 1
                    
                    # Load embeddings for current timestep immediately
                    current_t = getattr(self, "current_timestep", 0)
                    if current_t in self.embeddings_4d:
                        self._load_embedding_for_timestep(current_t)
                    
                    show_info(f"Embeddings loaded for {loaded_count} timesteps")
                    
                except Exception as e:
                    print(f"Failed to load embeddings: {e}")
                    show_info(f"Failed to load embeddings: {e}")

            btn_save.clicked.connect(lambda _: _save_embeddings())
            btn_load.clicked.connect(lambda _: _load_embeddings())

            # Removed save/load embeddings folder functions and handlers

            # Insert the embedding widget at the top of the annotator layout
            try:
                self._annotator_widget.layout().insertWidget(0, emb_widget)
            except Exception:
                # fallback: add at end
                try:
                    self._annotator_widget.layout().addWidget(emb_widget)
                except Exception:
                    pass
            # --- Remap points UI and Napari points layer ---
            try:
                # create / reuse a Napari Points layer named 'remap_points' (3D coords)
                try:
                    if "remap_points" in self._viewer.layers:
                        self._remap_points_layer = self._viewer.layers["remap_points"]
                    else:
                        self._remap_points_layer = self._viewer.add_points(np.empty((0, 3)), name="remap_points", ndim=3, face_color='red', size=5)
                except Exception:
                    self._remap_points_layer = None

                # right-sidebar widget that will hold per-point remap entries
                remap_widget = QtWidgets.QWidget()
                remap_widget.setLayout(QtWidgets.QVBoxLayout())
                remap_widget.layout().setContentsMargins(4, 4, 4, 4)
                remap_widget.layout().setSpacing(6)
                remap_label = QtWidgets.QLabel("Remap points")
                remap_widget.layout().addWidget(remap_label)

                
                # Scroll area with a vertical layout for entries
                try:
                    scroll = QtWidgets.QScrollArea()
                    scroll.setWidgetResizable(True)
                    inner = QtWidgets.QWidget()
                    inner.setLayout(QtWidgets.QVBoxLayout())
                    inner.layout().setSpacing(4)
                    inner.layout().setContentsMargins(0, 0, 0, 0)
                    scroll.setWidget(inner)
                    remap_widget.layout().addWidget(scroll)
                    self._remap_entries_container = inner.layout()
                except Exception:
                    # fallback: direct vertical layout
                    self._remap_entries_container = QtWidgets.QVBoxLayout()
                    remap_widget.layout().addLayout(self._remap_entries_container)

                # Apply button and shortcut hint
                apply_btn = QtWidgets.QPushButton("Apply remaps (Shift+R)")
                remap_widget.layout().addWidget(apply_btn)
                # Clear button to remove all remap points and entries
                clear_btn = QtWidgets.QPushButton("Clear remap points")
                remap_widget.layout().addWidget(clear_btn)

                # Insert the remap widget into the annotator panel (after embeddings)
                try:
                    self._annotator_widget.layout().addWidget(remap_widget)
                except Exception:
                    try:
                        self._annotator_widget.layout().insertWidget(1, remap_widget)
                    except Exception:
                        pass

                # storage for original IDs (aligned with points order) and widget refs
                self._remap_point_original_ids = []
                self._remap_target_widgets = []

                # connect points layer -> handler so new points create UI entries
                try:
                    if self._remap_points_layer is not None:
                        # keep a small wrapper that updates originals whenever points change
                        self._remap_points_layer.events.data.connect(lambda e=None: self._on_remap_points_changed())
                except Exception:
                    pass

                # connect apply button
                try:
                    apply_btn.clicked.connect(lambda _: self.apply_remaps())
                except Exception:
                    pass

                # connect clear button
                try:
                    clear_btn.clicked.connect(lambda _: self.clear_remap_points())
                except Exception:
                    pass

                # keyboard shortcut Shift+R to apply remaps
                try:
                    shortcut = QtWidgets.QShortcut(QKeySequence("Shift+R"), remap_widget)
                    shortcut.activated.connect(lambda: self.apply_remaps())
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            # don't fail initialization if Qt isn't available
            pass


    def _reorder_layers(self):
        """Reorder layers according to desired order and persist across timestep changes."""
        try:
            # Get all layer names currently in viewer
            current_layers = [layer.name for layer in self._viewer.layers]
            
            # Filter desired order to only include layers that exist
            existing_desired = [name for name in self._desired_layer_order if name in current_layers]
            
            # Move layers to desired positions (bottom to top)
            for idx, layer_name in enumerate(existing_desired):
                try:
                    if layer_name in self._viewer.layers:
                        current_idx = self._viewer.layers.index(layer_name)
                        if current_idx != idx:
                            self._viewer.layers.move(current_idx, idx)
                except Exception as e:
                    print(f"[WARN] Failed to move layer {layer_name}: {e}")
                    
        except Exception as e:
            print(f"[WARN] Layer reordering failed: {e}")

            # DROPDOWN REMAPPER WIDGET IF YOU WANT IT BACK
            # # Create and add the ID remapper widget if not already added
            # if not hasattr(self, '_remapper_widget'):
            #     try:
            #         from ._widgets import IdRemapperWidget
            #         from superqt import QCollapsible
            #         remapper = IdRemapperWidget(self)
            #         remapper_widget = QtWidgets.QWidget()
            #         remapper_widget.setLayout(QtWidgets.QVBoxLayout())
            #         remapper_collapsible = QCollapsible("ID Remapper", remapper_widget)
            #         remapper_collapsible.addWidget(remapper)
            #         remapper_widget.layout().addWidget(remapper_collapsible)
            #         if hasattr(self, '_annotator_widget') and hasattr(self._annotator_widget, 'layout'):
            #             self._annotator_widget.layout().insertWidget(1, remapper_widget)
            #         self._remapper_widget = remapper_widget
            #     except Exception as e:
            #         print(f"Failed to create ID remapper widget: {str(e)}")

    def update_image(self, image_4d):
        """Initialize annotator state with a 4D image.

        Adds/updates `raw_4d` and `committed_objects_4d` (4D labels). Also
        creates persistent 3D interactive layers for the current timestep
        that will be updated in-place when switching timesteps.
        """
        if image_4d.ndim != 4:
            raise ValueError(f"Expected 4D data (T,Z,Y,X), got {image_4d.shape}")

        self.image_4d = image_4d
        self.n_timesteps = image_4d.shape[0]
        self.current_timestep = 0

        # configure napari dims for time-series
        try:
            self._viewer.dims.ndim = 4
            self._viewer.dims.axis_labels = ["T", "Z", "Y", "X"]
        except Exception:
            pass
        # add or update persistent 4D raw image layer
        if "raw_4d" in self._viewer.layers:
            try:
                self._viewer.layers["raw_4d"].data = image_4d
                self._viewer.layers["raw_4d"].visible = True
            except Exception:
                pass
        else:
            try:
                self._viewer.add_image(image_4d, name="raw_4d")
            except Exception:
                pass

        # initialize persistent 4D containers and ensure Napari layers reference
        # the same underlying arrays (so edits in Napari mutate our arrays)
        self.segmentation_4d = np.zeros_like(image_4d, dtype=np.uint32)
        self.auto_segmentation_4d = np.zeros_like(image_4d, dtype=np.uint32)
        self.current_object_4d = np.zeros_like(image_4d, dtype=np.uint32)
        # Initialize 4D-aware point prompts dictionary
        self.point_prompts_4d = {}
        self._segmentation_cache = [self.segmentation_4d[t].copy() for t in range(self.n_timesteps)]

        # add or update persistent 4D labels layers so Napari edits directly
        # mutate the underlying 4D arrays (no per-timestep recreation)
        try:
            if "committed_objects_4d" in self._viewer.layers:
                self._viewer.layers["committed_objects_4d"].data = self.segmentation_4d
            else:
                self._viewer.add_labels(data=self.segmentation_4d, name="committed_objects_4d")
        except Exception:
            pass

        # Add the small 4D timestep tools widget (segment/commit across T)
        try:
            self._annotator_widget.layout().addWidget(TimestepToolsWidget(self))
        except Exception:
            pass

        try:
            if "current_object_4d" in self._viewer.layers:
                self._viewer.layers["current_object_4d"].data = self.current_object_4d
            else:
                self._viewer.add_labels(data=self.current_object_4d, name="current_object_4d")
        except Exception:
            pass

        try:
            if "auto_segmentation_4d" in self._viewer.layers:
                self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
            else:
                self._viewer.add_labels(data=self.auto_segmentation_4d, name="auto_segmentation_4d")
        except Exception:
            pass
        
        # make point_prompts 4D-aware: each timestep has its own points
        try:
            t = getattr(self, "current_timestep", 0)
            pts_t = np.array(self.point_prompts_4d.get(t, np.empty((0, 3))))

            # if layer exists, update its data
            if "point_prompts" in self._viewer.layers:
                self._viewer.layers["point_prompts"].data = pts_t
            else:
                layer = self._viewer.add_points(pts_t, name="point_prompts", size=10,
                                                face_color="green", edge_color="black",
                                                blending="translucent")
                # SAM-style color cycle
                layer.face_color_cycle = ["limegreen", "red"]
                
                # Setup cursor handling for point prompts layer
                try:
                    canvas = self._viewer.window.qt_viewer.canvas.native
                    
                    def on_mouse_enter(event):
                        """Set crosshair cursor when entering viewer in add mode"""
                        try:
                            if layer.mode == 'add':
                                canvas.setCursor(Qt.CrossCursor)
                        except Exception:
                            pass
                    
                    def on_mouse_leave(event):
                        """Reset cursor to normal when leaving the viewer"""
                        try:
                            canvas.setCursor(Qt.ArrowCursor)
                        except Exception:
                            pass
                    
                    # Connect to canvas enter/leave events
                    try:
                        original_enter = canvas.enterEvent
                        canvas.enterEvent = lambda event: (on_mouse_enter(event), original_enter(event) if callable(original_enter) else None)
                        canvas.leaveEvent = lambda event: on_mouse_leave(event)
                    except Exception:
                        pass
                except Exception:
                    pass

            # listener: when user adds/deletes points, save back to the CURRENT timestep
            def _update_point_prompts(event=None):
                # Get current timestep dynamically, not from closure
                t_now = getattr(self, "current_timestep", 0)
                self.point_prompts_4d[t_now] = np.array(self._viewer.layers["point_prompts"].data)

            self._viewer.layers["point_prompts"].events.data.connect(_update_point_prompts)

        except Exception as e:
            print(f"Error initializing 4D-aware point prompts: {e}")

        # ensure our local arrays are the same object as Napari layer data
        try:
            if "committed_objects_4d" in self._viewer.layers:
                self.segmentation_4d = self._viewer.layers["committed_objects_4d"].data
            if "current_object_4d" in self._viewer.layers:
                self.current_object_4d = self._viewer.layers["current_object_4d"].data
            if "auto_segmentation_4d" in self._viewer.layers:
                self.auto_segmentation_4d = self._viewer.layers["auto_segmentation_4d"].data
        except Exception:
            pass

        # set initial visible timestep via Napari dims (no layer recreation)
        try:
            # set dims current step to (t, z, y, x) = (0, 0, 0, 0)
            self._viewer.dims.current_step = (0,) + (0,) * (self._viewer.dims.ndim - 1)
            # connect dims handler once
            if not getattr(self, "_dims_handler_connected", False):
                self._viewer.dims.events.current_step.connect(self._on_dims_current_step)
                self._dims_handler_connected = True
        except Exception:
            pass
        
        # Apply desired layer order after all layers are created
        try:
            self._reorder_layers()
        except Exception:
            pass

        # final UI hook (no-op by default)
        self._update_timestep_controls()
    

    def _load_timestep(self, t: int):
        """Switch visible timestep by updating Napari dims without recreating layers.

        Persist any UI-only state (points) before switching, update Napari's
        dims.current_step to show the requested T slice, and refresh the
        points layer to show prompts for the new timestep.
        """
        new_t = int(t)
        if self.image_4d is None:
            return
        if not (0 <= new_t < self.n_timesteps):
            return

        prev_t = getattr(self, "current_timestep", None)
        # persist current points
        try:
            if prev_t is not None and "point_prompts" in self._viewer.layers:
                try:
                    pts = np.array(self._viewer.layers["point_prompts"].data)
                    self.point_prompts_4d[prev_t] = pts.copy() if pts.size else np.empty((0, 3))
                except Exception:
                    pass
        except Exception:
            pass

        # set Napari's current_step to the new timestep while preserving
        # the non-time axes (Z/Y/X) if possible. This prevents resetting
        # the Z slider back to 0 when switching timesteps.
        try:
            # get existing step tuple and preserve its non-time entries
            current = list(self._viewer.dims.current_step)
            ndim = max(4, getattr(self._viewer.dims, "ndim", 4))
            if len(current) < ndim:
                current = current + [0] * (ndim - len(current))
            current[0] = new_t
            self._viewer.dims.current_step = tuple(current)
        except Exception:
            try:
                self._viewer.dims.current_step = (new_t,) + (0,) * (max(4, getattr(self._viewer.dims, "ndim", 4)) - 1)
            except Exception:
                pass

        # refresh points layer to the new timestep (in-place)
        try:
            pts_new = self.point_prompts_4d.get(new_t, np.empty((0, 3)))
            lay = self._viewer.layers["point_prompts"] if "point_prompts" in self._viewer.layers else None
            if lay is not None:
                lay.data = np.array(pts_new) if pts_new is not None else np.empty((0, 3))
        except Exception:
            pass

        self.current_timestep = new_t

        if hasattr(self, "_ensure_embeddings_active_for_t"):
            try:
                self._ensure_embeddings_active_for_t(t)
                # activation is silent to avoid spamming the console
                # (previously printed activation messages here)
            except Exception:
                # silently ignore activation errors; callers may inspect state
                pass

    def _preview_timestep(self, t: int, downscale=(4, 4, 4)):
        """Optional preview while scrubbing — currently a no-op to avoid
        replacing the persistent raw layers which would re-order layers.
        """
        return

    def commit_segmentation(self, seg_volume: np.ndarray):
        """Save a 3D segmentation slice into the 4D labels layer in-place.

        Write directly into the `committed_objects_4d` Napari labels layer so
        that no layer recreation is necessary. Refresh the layer after write.
        """
        t = self.current_timestep
        if t is None:
            return

        try:
            layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
            if layer is None:
                # ensure our 4D container exists
                if self.segmentation_4d is None and self.image_4d is not None:
                    self.segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
                    try:
                        self._viewer.add_labels(data=self.segmentation_4d, name="committed_objects_4d")
                        layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
                    except Exception:
                        layer = None

            if layer is not None and seg_volume is not None:
                # Ensure the segmentation slice matches the target shape. If not,
                # attempt a nearest-neighbour resize to avoid broadcasting/aliasing.
                try:
                    target_shape = layer.data.shape[1:]
                    if getattr(seg_volume, "shape", None) != target_shape:
                        seg_volume = _sk_resize(
                            seg_volume.astype("float32"), target_shape, order=0, preserve_range=True, anti_aliasing=False
                        ).astype(seg_volume.dtype)
                except Exception:
                    # If resizing fails, continue and let assignment raise if incompatible.
                    pass

                # Prevent event-driven handlers from seeing an intermediate state
                # and avoid possible aliasing by assigning a copy under the event blocker.
                try:
                    ev = getattr(layer, "events", None)
                    if ev is not None and hasattr(ev, "data"):
                        with layer.events.data.blocker():
                            layer.data[t] = seg_volume.copy()
                    else:
                        layer.data[t] = seg_volume.copy()
                except Exception:
                    # Fallback to direct assignment without blocker.
                    layer.data[t] = seg_volume.copy()

                # keep local ref in sync
                self.segmentation_4d = layer.data
                try:
                    layer.refresh()
                except Exception:
                    pass

                # update cache
                try:
                    if self._segmentation_cache is None:
                        self._segmentation_cache = [None] * self.n_timesteps
                    self._segmentation_cache[t] = self.segmentation_4d[t].copy()
                except Exception:
                    pass
        except Exception:
            pass

        print(f"✅ Committed segmentation for timestep {t}")

    def save_current_object_to_4d(self):
        """Ensure `current_object_4d` Napari layer and local array are in sync.

        Editing is expected to happen directly on the 4D `current_object_4d`
        labels layer. This method refreshes the local reference so the
        Python-side array remains the same object as Napari layer.data.
        """
        try:
            if "current_object_4d" in self._viewer.layers:
                self.current_object_4d = self._viewer.layers["current_object_4d"].data
        except Exception:
            pass

    def save_point_prompts(self):
        """Save the point_prompts 3D layer into per-timestep storage."""
        t = int(getattr(self, "current_timestep", 0) or 0)
        if "point_prompts" in self._viewer.layers:
            try:
                pts = np.array(self._viewer.layers["point_prompts"].data)
                self.point_prompts_4d[t] = pts.copy() if pts.size else np.empty((0, 3))
            except Exception:
                pass

    # ----------------- Embedding helpers for 4D -----------------
    def compute_embeddings_for_timestep(self, t: int, model_type: str = None, device: str | None = None, save_path: str | None = None, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute image embeddings for a single timestep and store them in AnnotatorState.

        This wraps AnnotatorState.initialize_predictor for convenience when working with 4D (T,Z,Y,X)
        data. It extracts the 3D volume at timestep `t` and computes embeddings with ndim=3.
        """
        from ._state import AnnotatorState

        if self.image_4d is None:
            raise RuntimeError("No 4D image loaded")
        if not (0 <= t < self.n_timesteps):
            raise IndexError("t out of range")

        image3d = self.image_4d[int(t)]
        state = AnnotatorState()
        # default model_type if not provided
        model_type = model_type or getattr(state, "predictor", None) and getattr(state.predictor, "model_name", None) or None
        # initialize predictor and compute embeddings for this 3D volume
        state.initialize_predictor(
            image3d,
            model_type=model_type or "vit_b_lm",
            ndim=3,
            save_path=save_path,
            device=device,
            tile_shape=tile_shape,
            halo=halo,
            prefer_decoder=prefer_decoder,
        )
        # Capture the computed embeddings and store them per-timestep only
        try:
            embeds = state.image_embeddings
        except Exception:
            embeds = None
        try:
            if embeds is not None:
                # store only in per-timestep cache
                self.embeddings_4d[int(t)] = embeds
        except Exception:
            pass

        # Only bind the embeddings into the global AnnotatorState if this
        # timestep is currently active; otherwise detach to avoid leaking the
        # embedding globally across timesteps.
        try:
            if int(getattr(self, "current_timestep", 0) or 0) == int(t):
                try:
                    state.image_embeddings = embeds
                    # update active marker
                    self._active_embedding_t = int(t)
                except Exception:
                    pass
            else:
                try:
                    # detach any embeddings from the global state
                    state.image_embeddings = None
                    # do not clear predictor, only the embedding handle
                    if getattr(state, "embedding_path", None) is not None and state.embedding_path == save_path:
                        state.embedding_path = None
                except Exception:
                    pass
        except Exception:
            pass
        # Update the state's image_name so widgets reflect the selection
        try:
            state.image_name = (self._viewer.layers["raw_4d"].name if "raw_4d" in self._viewer.layers else state.image_name)
        except Exception:
            pass
        # Ensure AnnotatorState has image_shape / image_scale set so downstream
        # segmentation widgets and helpers (which read state.image_shape) work
        # when embeddings are computed via this helper (instead of the embedding widget).
        try:
            # image3d has shape (Z, Y, X)
            state.image_shape = tuple(image3d.shape)
        except Exception:
            pass
        try:
            layer = self._viewer.layers["raw_4d"] if "raw_4d" in self._viewer.layers else None
            if layer is not None:
                # Napari image layer scale for raw_4d is (T, Z, Y, X). Use the spatial part.
                scale = getattr(layer, "scale", None)
                if scale is not None and len(scale) >= 4:
                    state.image_scale = tuple(scale[1:])
                elif scale is not None and len(scale) == 3:
                    state.image_scale = tuple(scale)
        except Exception:
            pass

        # return the per-timestep stored embedding (may be None)
        return self.embeddings_4d.get(int(t))

    def compute_embeddings_for_all_timesteps(self, model_type: str = None, device: str | None = None, base_save_path: str | None = None, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute embeddings for every timestep. Saves to separate files if base_save_path is provided.

        WARNING: this can be slow and memory/disk intensive. Use with care.
        Returns a list of image_embeddings objects (one per timestep).
        """
        results = []
        # ensure we have a per-timestep embeddings dict
        if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
            self.embeddings_4d = {}

        for t in range(self.n_timesteps):
            sp = None if base_save_path is None else f"{base_save_path}_t{t}.zarr"
            embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=sp, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
            results.append(embeds)
            try:
                # store embedding for this timestep; keep whatever structure compute returned
                self.embeddings_4d[int(t)] = embeds
            except Exception:
                # best-effort: ignore failures to cache
                pass
        return results

    def compute_and_save_embeddings(self, output_dir: str | Path, model_type: str = None, device: str | None = None, per_timestep: bool = True, overwrite: bool = False, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute embeddings for all timesteps and save into `output_dir`.

        By default this creates one zarr store per timestep named
        `embeddings_t{t}.zarr` inside `output_dir`. If `per_timestep` is
        False the method will attempt to compute embeddings in-memory and
        store a single `embeddings.npz` file (may be large).

        Returns a list of embedding objects (as returned by precompute), and
        writes a `manifest.json` into `output_dir` describing the files.
        """
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)

        manifest = {
            "n_timesteps": int(self.n_timesteps),
            "per_timestep": bool(per_timestep),
            "files": {},
        }

        results = []
        if per_timestep:
            for t in range(self.n_timesteps):
                fname = outp / f"embeddings_t{t}.zarr"
                if fname.exists() and not overwrite:
                    # load existing (skip compute)
                    results.append({"path": str(fname)})
                    manifest["files"][str(t)] = str(fname.name)
                    continue

                # compute and save to zarr
                save_path = str(fname)
                embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=save_path, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
                results.append(embeds)
                manifest["files"][str(t)] = str(fname.name)

        else:
            # compute all embeddings in-memory and write a single npz
            arrs = {}
            for t in range(self.n_timesteps):
                embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=None, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
                # `embeds` is a dict containing 'features' (numpy or zarr)
                feats = embeds.get("features")
                # If features are zarr-like, read into memory (could be large)
                try:
                    if hasattr(feats, "[:]"):
                        feats_np = feats[:]  # zarr or numpy-like
                    else:
                        feats_np = np.asarray(feats)
                except Exception:
                    feats_np = np.asarray(feats)
                arrs[f"t{t}"] = feats_np
                results.append({"features": feats_np})

            npz_path = outp / "embeddings.npz"
            if npz_path.exists() and not overwrite:
                raise FileExistsError(f"{npz_path} already exists. Use overwrite=True to replace.")
            np.savez_compressed(str(npz_path), **arrs)
            manifest["files"] = {str(t): str(npz_path.name) for t in range(self.n_timesteps)}

        # write manifest
        manifest_path = outp / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        return results

    def load_saved_embeddings(self, path: str | Path, lazy: bool = True):
        """Load embeddings previously saved with `compute_and_save_embeddings`.

        Args:
            path: Directory containing embeddings (manifest.json) or a single .npz file.
            lazy: If True and the embeddings are stored as zarr stores, return zarr objects
                  without loading full arrays into memory. If False, load numpy arrays.

        Returns:
            A dict mapping timestep (int) -> embedding dict (with at least key 'features').
        """
        p = Path(path)
        results = {}
        if p.is_dir():
            # Detect a parent directory containing per-timestep subfolders
            # named like t0, t1, ... and map them to results[t] = {"path": str(t_folder)}
            try:
                subdirs = [d for d in sorted(p.iterdir()) if d.is_dir() and d.name.startswith("t") and d.name[1:].isdigit()]
                if subdirs:
                    for d in subdirs:
                        try:
                            t = int(d.name[1:])
                            results[t] = {"path": str(d)}
                        except Exception:
                            # ignore non-numeric t* dirs
                            pass
                    return results
            except Exception:
                # fall through to legacy manifest/zarr discovery
                pass
            manifest = p / "manifest.json"
            if not manifest.exists():
                # try to discover files by pattern embeddings_t*.zarr
                zarrs = sorted(p.glob("embeddings_t*.zarr"))
                if not zarrs:
                    raise FileNotFoundError(f"No manifest.json or embeddings_* found in {p}")
                for z in zarrs:
                    tstr = z.stem.split("_t")[-1]
                    import zarr as _zarr
                    f = _zarr.open(str(z), mode="r")
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"Skipping {z}: no array-like 'features' dataset found.")
                        except Exception:
                            pass
                        continue
                    # Try to surface input/original size metadata expected by downstream code
                    attrs = getattr(feats, "attrs", {}) or {}
                    # If the zarr store doesn't contain tiling metadata, synthesize
                    # a conservative `input_size`/`original_size` so downstream
                    # prompt-based segmentation treats this as a non-tiled embedding
                    # (avoids accessing attrs['shape'] which may be missing).
                    input_size = attrs.get("input_size")
                    original_size = attrs.get("original_size")
                    if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                        # fallback: infer spatial size from the last two dimensions
                        try:
                            inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                            input_size = input_size or inferred
                            original_size = original_size or inferred
                        except Exception:
                            input_size = input_size or None
                            original_size = original_size or None
                    results[int(tstr)] = {"features": feats, "input_size": input_size, "original_size": original_size}
                return results

            with open(manifest, "r") as fh:
                manifestd = json.load(fh)
            for tstr, fname in manifestd.get("files", {}).items():
                t = int(tstr)
                filep = p / fname
                if filep.suffix == ".npz":
                    data = np.load(str(filep))
                    # assume key 't{t}' exists
                    arr = data.get(f"t{t}")
                    # npz does not contain metadata about input/original size -> leave as None
                    results[t] = {"features": arr, "input_size": None, "original_size": None}
                else:
                    import zarr as _zarr
                    f = _zarr.open(str(filep), mode="r")
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"Skipping {filep}: no array-like 'features' dataset found.")
                        except Exception:
                            pass
                        continue
                    attrs = getattr(feats, "attrs", {}) or {}
                    input_size = attrs.get("input_size")
                    original_size = attrs.get("original_size")
                    if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                        try:
                            inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                            input_size = input_size or inferred
                            original_size = original_size or inferred
                        except Exception:
                            input_size = input_size or None
                            original_size = original_size or None
                    if lazy:
                        results[t] = {"features": feats, "input_size": input_size, "original_size": original_size}
                    else:
                        results[t] = {"features": feats[:], "input_size": input_size, "original_size": original_size}
            return results

        elif p.is_file() and p.suffix == ".npz":
            data = np.load(str(p))
            for key in data.files:
                if key.startswith("t"):
                    t = int(key[1:])
                    results[t] = {"features": data[key], "input_size": None, "original_size": None}
            return results
        else:
            raise FileNotFoundError(f"No embeddings found at {path}")

    def set_embeddings_folder(self, path: str | Path, lazy: bool = True):
        """Set a directory containing per-timestep embeddings and load the mapping.

        Stores the path in `self._last_embeddings_dir` and populates `self.embeddings_4d`.
        Returns the loaded mapping (t -> embedding entry).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Embeddings path does not exist: {p}")
        self._last_embeddings_dir = str(p)
        loaded = self.load_saved_embeddings(p, lazy=lazy)
        # store mapping in per-timestep cache
        try:
            self.embeddings_4d = {int(k): v for k, v in loaded.items()}
        except Exception:
            self.embeddings_4d = loaded
        return self.embeddings_4d

    def reload_embeddings_from_last(self, lazy: bool = True):
        """Reload embeddings mapping from the last set embeddings directory.

        Raises RuntimeError if no directory was previously set via `set_embeddings_folder` or
        by the compute/save helpers. Returns the loaded mapping.
        """
        if not getattr(self, "_last_embeddings_dir", None):
            raise RuntimeError("No last embeddings directory is set. Call set_embeddings_folder(path) first.")
        return self.set_embeddings_folder(self._last_embeddings_dir, lazy=lazy)

    def _load_embedding_for_timestep(self, t: int):
        """Load embeddings for a specific timestep from disk (lazy loading helper).
        
        This method is called when switching to a timestep that has a lazy embedding entry.
        It loads the embedding from disk and activates it in AnnotatorState.
        """
        try:
            if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
                return
            
            entry = self.embeddings_4d.get(t)
            if entry is None:
                return
            
            # If entry only has a path, load it from disk
            if isinstance(entry, dict) and "path" in entry and "features" not in entry:
                import zarr as _zarr
                # _select_array_from_zarr_group is defined at the top of this file
                
                path = Path(entry["path"])
                if not path.exists():
                    print(f"Warning: Embedding file not found: {path}")
                    return
                
                # Load the zarr file
                f = _zarr.open(str(path), mode="r")
                feats = _select_array_from_zarr_group(f)
                
                if feats is None:
                    print(f"Warning: No features found in {path}")
                    return
                
                # Get metadata from array attrs and root group attrs; fall back to image shape
                arr_attrs = dict(getattr(feats, "attrs", {}) or {})
                root_attrs = dict(getattr(f, "attrs", {}) or {})
                input_size = arr_attrs.get("input_size") or root_attrs.get("input_size")
                original_size = arr_attrs.get("original_size") or root_attrs.get("original_size")

                # Final fallback: use the current image shape (Y, X) rather than feature shape
                if original_size is None:
                    try:
                        img_shape_yx = tuple(self.image_4d[int(t)].shape[-2:])
                        original_size = img_shape_yx
                    except Exception:
                        # As a last resort do not guess from feats (would be 64x64 and wrong)
                        original_size = None
                if input_size is None:
                    input_size = original_size
                
                # Update the entry with loaded features
                self.embeddings_4d[t] = {
                    "features": feats,
                    "input_size": input_size,
                    "original_size": original_size,
                    "path": str(path)
                }
            
            # Now activate the embeddings for this timestep
            self._ensure_embeddings_active_for_t(t)
            
        except Exception as e:
            print(f"Failed to load embedding for timestep {t}: {e}")

    def _materialize_embedding_entry(self, entry):
        """Best-effort materialize a saved embedding entry.

        Accepts:
          - dict with 'features' -> returned as-is
          - dict with 'path' -> path to a .zarr store (or parent dir)
          - str path -> treated like dict with 'path'

        Returns embedding dict or None on failure.
        """
        if entry is None:
            return None
        # already materialized
        if isinstance(entry, dict) and "features" in entry:
            return entry

        p = None
        if isinstance(entry, dict) and "path" in entry and entry["path"]:
            p = Path(entry["path"])
        elif isinstance(entry, str):
            p = Path(entry)

        if p is None:
            return None

        # If p points to a per-timestep folder like 't3', treat it as a lazy
        # entry referencing that folder and return a simple {'path': str(p)}
        try:
            if p.is_dir() and p.name.startswith("t") and p.name[1:].isdigit():
                return {"path": str(p)}
        except Exception:
            pass

        # If given a parent directory, try to load via manifest (may return a mapping)
        try:
            if p.is_dir():
                loaded = self.load_saved_embeddings(p, lazy=True)
                # prefer the exact timestep if present; otherwise return the mapping
                return loaded
        except Exception:
            pass

            # If it's a zarr store, open it and return the 'features' object with synthesized metadata
        try:
            if p.suffix == ".zarr" or (p.exists() and p.is_dir() and any(p.glob("*.zarr"))):
                import zarr as _zarr

                # If p is a file-like zarr store, open it directly
                try:
                    f = _zarr.open(str(p), mode="r")
                except Exception:
                    # maybe the path points to a file-like directory; try opening parent
                    try:
                        f = _zarr.open(str(p), mode="r")
                    except Exception:
                        f = None

                if f is not None:
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"No suitable array-like dataset found inside {p}; cannot materialize embeddings.")
                        except Exception:
                            pass
                        return None
                    arr_attrs = dict(getattr(feats, "attrs", {}) or {})
                    root_attrs = dict(getattr(f, "attrs", {}) or {})
                    input_size = arr_attrs.get("input_size") or root_attrs.get("input_size")
                    original_size = arr_attrs.get("original_size") or root_attrs.get("original_size")
                    if original_size is None:
                        try:
                            # Use annotator image shape if available
                            # Note: we cannot know 't' here reliably; leave None and let activation fill in
                            original_size = None
                        except Exception:
                            original_size = None
                    if input_size is None:
                        input_size = original_size
                    return {"features": feats, "input_size": input_size, "original_size": original_size}
        except Exception:
            pass

        # Last resort: if it's an npz file, let load_saved_embeddings handle it
        try:
            if p.exists() and p.suffix == ".npz":
                loaded = self.load_saved_embeddings(p)
                # return the first timestep's entry if single
                if isinstance(loaded, dict):
                    # if only one entry, return that
                    if len(loaded) == 1:
                        return list(loaded.values())[0]
                    return loaded
        except Exception:
            pass

        return None

    def _ensure_embeddings_active_for_t(self, t: int):
        """If we have cached embeddings for timestep t, activate them on AnnotatorState.

        This will materialize lazy entries if necessary.
        """
        try:
            if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
                return
            # Detach any previously active embeddings if switching timesteps
            try:
                state_detach = AnnotatorState()
                if getattr(self, "_active_embedding_t", None) is not None and self._active_embedding_t != int(t):
                    try:
                        state_detach.image_embeddings = None
                        state_detach.embedding_path = None
                    except Exception:
                        pass
                    try:
                        self._active_embedding_t = None
                    except Exception:
                        pass
            except Exception:
                pass

            entry = self.embeddings_4d.get(int(t))
            if entry is None:
                # No embeddings for this timestep: ensure global state is cleared
                try:
                    state_clear = AnnotatorState()
                    state_clear.image_embeddings = None
                    state_clear.embedding_path = None
                except Exception:
                    pass
                return

            # If entry is a mapping of multiple timesteps (returned from load_saved_embeddings on a dir),
            # prefer the exact t key.
            if isinstance(entry, dict) and any(isinstance(k, str) and k.isdigit() for k in entry.keys()):
                # already a mapping produced by load_saved_embeddings; pick the t entry if present
                maybe = entry.get(str(t)) or entry.get(int(t))
                if maybe is not None:
                    entry = maybe

            # If entry is a lazy path or string, materialize. Do this in background
            # to avoid blocking the UI for large zarr loads.
            if not (isinstance(entry, dict) and "features" in entry):
                # If already loading, return early
                if self._embedding_loading.get(int(t), False):
                    return

                # Mark as loading and spawn background thread
                self._embedding_loading[int(t)] = True

                def _bg():
                    try:
                        mat = self._materialize_embedding_entry(entry)
                        if mat is not None:
                            try:
                                self.embeddings_4d[int(t)] = mat
                            except Exception:
                                pass
                            # once materialized, call this method again to perform activation
                            try:
                                # clear loading flag before re-entering
                                self._embedding_loading[int(t)] = False
                            except Exception:
                                pass
                            try:
                                # call activation synchronously now that mat exists
                                self._ensure_embeddings_active_for_t(t)
                            except Exception:
                                pass
                        else:
                            try:
                                show_info(f"Failed to materialize embeddings for timestep {t}.")
                            except Exception:
                                pass
                    finally:
                        try:
                            self._embedding_loading[int(t)] = False
                        except Exception:
                            pass

                thread = threading.Thread(target=_bg, daemon=True)
                thread.start()
                return

            # Finally, set AnnotatorState's image_embeddings to this dict so downstream code uses it
            try:
                state = AnnotatorState()
                # if predictor is missing we'll (re)initialize it for this timestep
                image3d = None
                try:
                    image3d = self.image_4d[int(t)]
                except Exception:
                    image3d = None

                # If the entry is already materialized and contains features, use it as save_path
                save_path = None
                if isinstance(entry, dict) and "features" in entry:
                    # Ensure minimal metadata is present; do not reject on mismatch
                    if image3d is not None:
                        expected_shape = tuple(image3d.shape[-2:])
                        if entry.get("original_size") is None:
                            entry["original_size"] = expected_shape
                        if entry.get("input_size") is None:
                            entry["input_size"] = entry.get("original_size")
                    save_path = entry
                elif isinstance(entry, dict) and entry.get("path"):
                    save_path = str(entry.get("path"))
                elif isinstance(entry, str):
                    save_path = entry

                # Determine a model_type to use if predictor is not present
                try:
                    model_type = getattr(state.predictor, "model_type", None) or getattr(state.predictor, "model_name", None)
                except Exception:
                    model_type = None
                if model_type is None:
                    try:
                        model_type = _vutil._DEFAULT_MODEL
                    except Exception:
                        model_type = "vit_b_lm"

                # Initialize predictor / embeddings for this timestep. If predictor already exists,
                # initialize_predictor will reuse it.
                try:
                    if image3d is not None:
                        state.initialize_predictor(
                            image3d,
                            model_type=model_type,
                            ndim=3,
                            save_path=save_path,
                            predictor=getattr(state, "predictor", None),
                            prefer_decoder=getattr(state, "decoder", None) is not None,
                        )
                except Exception:
                    # Continue even if predictor init fails; downstream code may handle it.
                    pass

                # ensure state.image_embeddings references the entry
                try:
                    state.image_embeddings = entry
                    try:
                        self._active_embedding_t = int(t)
                    except Exception:
                        pass
                except Exception:
                    pass

                # also set embedding_path if available (do this early so loaders can access files)
                try:
                    if isinstance(entry, dict) and entry.get("path"):
                        state.embedding_path = str(entry.get("path"))
                    elif isinstance(entry, str):
                        state.embedding_path = entry
                except Exception:
                    pass

                # Ensure image_shape/scale/name are set for downstream widgets
                try:
                    if image3d is not None:
                        state.image_shape = tuple(image3d.shape)
                except Exception:
                    pass
                try:
                    layer = self._viewer.layers.get("raw_4d", None)
                    if layer is not None:
                        scale = getattr(layer, "scale", None)
                        if scale is not None and len(scale) >= 4:
                            state.image_scale = tuple(scale[1:])
                        elif scale is not None and len(scale) == 3:
                            state.image_scale = tuple(scale)
                except Exception:
                    pass

                # Ensure AMG state exists. If embeddings were saved on disk (embedding_path), try to load
                # the precomputed amg/is state. Otherwise compute AMG state in-memory.
                try:
                    if state.amg_state is None:
                        # If embedding_path exists on disk, load cached AMG/IS state
                        if getattr(state, "embedding_path", None):
                            try:
                                if state.decoder is not None:
                                    state.amg_state = _load_is_state(state.embedding_path)
                                else:
                                    state.amg_state = _load_amg_state(state.embedding_path)
                            except Exception:
                                state.amg_state = None

                        # If still missing and we have in-memory embeddings, compute AMG state now
                        if state.amg_state is None and isinstance(state.image_embeddings, dict) and "features" in state.image_embeddings and image3d is not None:
                            try:
                                is_tiled = state.image_embeddings.get("input_size") is None
                                amg = instance_segmentation.get_amg(state.predictor, is_tiled=is_tiled, decoder=state.decoder)
                                # initialize amg on the full 3D volume
                                amg.initialize(image3d, image_embeddings=state.image_embeddings, verbose=False)
                                state.amg = amg
                                state.amg_state = amg.get_state()
                            except Exception:
                                # best-effort; leave amg_state None if computation fails
                                pass
                except Exception:
                    pass

                # Show a concise one-time info about active embeddings for this timestep
                try:
                    if int(t) not in getattr(self, "_reported_embedding_info_t", set()):
                        orig = None
                        tiled = False
                        src_name = "memory"
                        try:
                            if isinstance(entry, dict):
                                orig = entry.get("original_size")
                                feats = entry.get("features")
                                if feats is not None and hasattr(feats, "attrs"):
                                    tiled = feats.attrs.get("tile_shape") is not None
                                pth = entry.get("path")
                                if pth:
                                    src_name = Path(pth).name
                            elif isinstance(entry, str):
                                src_name = Path(entry).name
                        except Exception:
                            pass
                        try:
                            show_info(f"Embeddings t{t} active • size={orig if orig is not None else 'unknown'} • tiled={bool(tiled)} • source={src_name}")
                        except Exception:
                            pass
                        try:
                            self._reported_embedding_info_t.add(int(t))
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def _on_dims_current_step(self, event):
        """Handle napari dims current_step changes (time slider).
        Persists current 3D edits and loads the new timestep into the 3D views.
        Each timestep now keeps independent point prompts (4D-aware).
        """
        import numpy as np

        # --- detect new timestep ---
        try:
            val = getattr(event, "value", None)
            if val is None:
                val = getattr(event, "current_step", None) or getattr(self._viewer.dims, "current_step", None)
            new_t = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
        except Exception:
            return

        prev_t = getattr(self, "current_timestep", None)
        if new_t == prev_t:
            return

        # --- persist old points ---
        try:
            if prev_t is not None and "point_prompts" in self._viewer.layers:
                pts = np.array(self._viewer.layers["point_prompts"].data)
                if not hasattr(self, "point_prompts_4d"):
                    self.point_prompts_4d = {}
                self.point_prompts_4d[prev_t] = pts.copy() if pts.size else np.empty((0, 3))
        except Exception as e:
            print(f"[WARN] Save point prompts failed: {e}")

        # --- update current timestep ---
        self.current_timestep = new_t

        # --- embeddings switching ---
        try:
            mgr = getattr(self, "timestep_embedding_manager", None)
            if mgr is not None:
                try:
                    mgr.on_timestep_changed(new_t)
                except Exception:
                    self._ensure_embeddings_active_for_t(new_t)
            else:
                self._ensure_embeddings_active_for_t(new_t)
        except Exception:
            pass

        # --- load the new timestep volume ---
        try:
            self._load_timestep(new_t)
        except Exception as e:
            print(f"[WARN] Load timestep failed: {e}")

        # --- load per-timestep point prompts ---
        try:
            if not hasattr(self, "point_prompts_4d"):
                self.point_prompts_4d = {}

            new_pts = self.point_prompts_4d.get(new_t, np.empty((0, 3)))

            # if layer doesn't exist yet, create once
            if "point_prompts" not in self._viewer.layers:
                layer = self._viewer.add_points(
                    new_pts,
                    name="point_prompts",
                    size=10,
                    face_color="green",  # SAM positive color
                    edge_color="black",
                    blending="translucent",
                )
                # SAM-style color cycle
                layer.face_color_cycle = ["limegreen", "red"]
                
                # Setup cursor handling for point prompts layer
                try:
                    canvas = self._viewer.window.qt_viewer.canvas.native
                    
                    def on_mouse_enter(event):
                        """Set crosshair cursor when entering viewer in add mode"""
                        try:
                            if layer.mode == 'add':
                                canvas.setCursor(Qt.CrossCursor)
                        except Exception:
                            pass
                    
                    def on_mouse_leave(event):
                        """Reset cursor to normal when leaving the viewer"""
                        try:
                            canvas.setCursor(Qt.ArrowCursor)
                        except Exception:
                            pass
                    
                    # Connect to canvas enter/leave events
                    try:
                        original_enter = canvas.enterEvent
                        canvas.enterEvent = lambda event: (on_mouse_enter(event), original_enter(event) if callable(original_enter) else None)
                        canvas.leaveEvent = lambda event: on_mouse_leave(event)
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                layer = self._viewer.layers["point_prompts"]
                layer.data = new_pts

            # --- reconnect event listener cleanly ---
            try:
                if hasattr(self, "_point_prompt_connection"):
                    layer.events.data.disconnect(self._point_prompt_connection)
            except Exception:
                pass

            def _update_points(event=None):
                t_now = getattr(self, "current_timestep", 0)
                self.point_prompts_4d[t_now] = np.array(layer.data)

            self._point_prompt_connection = _update_points
            layer.events.data.connect(self._point_prompt_connection)

        except Exception as e:
            print(f"[WARN] Reload point prompts failed: {e}")
        
        # Restore layer order after timestep change
        try:
            self._reorder_layers()
        except Exception:
            pass

    def next_timestep(self):
        """Move to the next timestep (if available)."""
        self.save_current_object_to_4d()
        self.save_point_prompts()
        if self.current_timestep + 1 < self.n_timesteps:
            self._load_timestep(self.current_timestep + 1)
        else:
            print("🚫 Already at last timestep.")

    def _update_timestep_controls(self):
        """Placeholder for UI controls update (no-op)."""
        pass

    # ----------------- Automatic segmentation helpers for 4D -----------------
    def auto_segment_timestep(self, t: int, mode: str = "auto", device: str | None = None, tile_shape=None, halo=None, gap_closing: int | None = None, min_z_extent: int | None = None, with_background: bool = True, min_object_size: int = 100, prefer_decoder: bool = True):
        """Run automatic 3D segmentation for a single timestep and store result in auto_segmentation_4d[t].

        This uses the project's `automatic_3d_segmentation` implementation which handles tiled vs untiled
        embeddings and decoder vs AMG-based segmentation.
        """
        if self.image_4d is None:
            raise RuntimeError("No 4D image loaded")
        if not (0 <= t < self.n_timesteps):
            raise IndexError("t out of range")

        # Ensure embeddings and predictor are initialized for this timestep.
        state = AnnotatorState()
        # If no predictor or embeddings are present for this timestep, compute them.
        if state.predictor is None or state.image_embeddings is None:
            # compute embeddings for this timestep (store per-timestep only)
            self.compute_embeddings_for_timestep(t=t, model_type=None, device=device, save_path=None, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
            # After computing, bind the per-timestep embeddings into the global state
            try:
                emb = self.embeddings_4d.get(int(t))
                if emb is None:
                    try:
                        show_info(f"❌ No embeddings available for timestep {t}; cannot run segmentation.")
                    except Exception:
                        print(f"❌ No embeddings available for timestep {t}; cannot run segmentation.")
                    return None
                state.image_embeddings = emb
                # set embedding path if available
                try:
                    if isinstance(emb, dict) and emb.get("path"):
                        state.embedding_path = str(emb.get("path"))
                except Exception:
                    pass
                # mark active
                try:
                    self._active_embedding_t = int(t)
                except Exception:
                    pass
            except Exception:
                pass

        predictor = state.predictor

        # Determine if tiled embeddings are used.
        is_tiled = False
        try:
            feats = state.image_embeddings.get("features") if isinstance(state.image_embeddings, dict) else None
            if feats is not None and hasattr(feats, "attrs") and feats.attrs.get("tile_shape") is not None:
                is_tiled = True
        except Exception:
            is_tiled = False

        # Create segmentor (AMG or decoder-based) using existing util.
        segmentor = instance_segmentation.get_amg(predictor, is_tiled=is_tiled, decoder=state.decoder)

        # Extract the 3D volume for this timestep
        vol3d = np.asarray(self.image_4d[int(t)])

        # Run automatic 3d segmentation
        seg = automatic_3d_segmentation(
            volume=vol3d,
            predictor=predictor,
            segmentor=segmentor,
            embedding_path=state.embedding_path,
            with_background=with_background,
            gap_closing=gap_closing,
            min_z_extent=min_z_extent,
            tile_shape=tile_shape,
            halo=halo,
            verbose=True,
            return_embeddings=False,
            min_object_size=min_object_size,
        )

        # Ensure segmentation output matches the volume shape (Z,Y,X). If not,
        # resize the segmentation volume (nearest-neighbour) to the target shape.
        try:
            target_shape = vol3d.shape
            if getattr(seg, "shape", None) != target_shape:
                seg = _sk_resize(seg.astype("float32"), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(seg.dtype)
        except Exception:
            # If resizing fails, continue and let assignment handle or raise.
            pass

        # Ensure our container exists
        if self.auto_segmentation_4d is None:
            self.auto_segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                if "auto_segmentation_4d" in self._viewer.layers:
                    self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
                else:
                    self._viewer.add_labels(data=self.auto_segmentation_4d, name="auto_segmentation_4d")
            except Exception:
                pass

        # Write back into the 4D auto segmentation container and refresh layer in-place
        try:
            self.auto_segmentation_4d[int(t)] = seg
            layer = self._viewer.layers["auto_segmentation_4d"] if "auto_segmentation_4d" in self._viewer.layers else None
            if layer is not None:
                layer.data[int(t)] = seg
                try:
                    layer.refresh()
                except Exception:
                    pass
        except Exception:
            # As a fallback, try replacing the whole layer data
            try:
                if "auto_segmentation_4d" in self._viewer.layers:
                    self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
            except Exception:
                pass

        print(f"✅ Auto-segmentation completed for timestep {t}")
        return seg

    def auto_segment_all_timesteps(self, mode: str = "auto", device: str | None = None, tile_shape=None, halo=None, gap_closing: int | None = None, min_z_extent: int | None = None, with_background: bool = True, min_object_size: int = 100, prefer_decoder: bool = True, overwrite: bool = False):
        """Run automatic 3D segmentation for every timestep and store results in auto_segmentation_4d.

        Returns a list of segmentation arrays per timestep.
        """
        results = []
        
        for t in range(self.n_timesteps):
            # skip existing unless overwrite
            if not overwrite and self.auto_segmentation_4d is not None:
                if np.any(self.auto_segmentation_4d[t]):
                    results.append(self.auto_segmentation_4d[t])
                    continue
            seg = self.auto_segment_timestep(t=t, mode=mode, device=device, tile_shape=tile_shape, halo=halo, gap_closing=gap_closing, min_z_extent=min_z_extent, with_background=with_background, min_object_size=min_object_size, prefer_decoder=prefer_decoder)
            results.append(seg)
        return results

    def remap_segment_id(self, timestep: int, old_id: int, new_id: int, propagate_forward: bool = False):
        """Remap a segment ID in a specific timestep.

        Args:
            timestep: The timestep containing the segment to remap
            old_id: The current ID of the segment
            new_id: The new ID to assign
            propagate_forward: If True, propagate the remapping to future timesteps
        """
        if self.segmentation_4d is None:
            raise ValueError("No segmentation available")
        if not (0 <= timestep < self.n_timesteps):
            raise ValueError(f"Invalid timestep {timestep}")

        # Do the remapping for the specified timestep
        mask = self.segmentation_4d[timestep] == old_id
        if not np.any(mask):
            print(f"⚠️ No object with ID {old_id} found in timestep {timestep}")
            return

        self.segmentation_4d[timestep][mask] = new_id

        # Update the view
        try:
            layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
            if layer is not None:
                layer.data = self.segmentation_4d
                try:
                    layer.refresh()
                except Exception:
                    pass
        except Exception:
            pass

        # Propagate to future timesteps if requested
        if propagate_forward:
            for t in range(timestep + 1, self.n_timesteps):
                mask = self.segmentation_4d[t] == old_id
                if np.any(mask):
                    self.segmentation_4d[t][mask] = new_id

        print(f"✅ Remapped segment ID {old_id} to {new_id} in timestep {timestep}"
              f"{' and propagated forward' if propagate_forward else ''}")

        # Update cache if it exists
        if self._segmentation_cache is not None:
            try:
                if 0 <= timestep < len(self._segmentation_cache):
                    self._segmentation_cache[timestep] = self.segmentation_4d[timestep].copy()
                    if propagate_forward:
                        for t in range(timestep + 1, min(self.n_timesteps, len(self._segmentation_cache))):
                            self._segmentation_cache[t] = self.segmentation_4d[t].copy()
            except Exception:
                pass

    # ----------------- Remap points helpers -----------------
    def _on_remap_points_changed(self):
        """Called whenever `remap_points` layer data changes.

        Adds/removes UI entries and records the original segment ID under each point
        for the current timestep.
        """
        try:
            lay = getattr(self, "_remap_points_layer", None)
            if lay is None:
                return
            pts = np.array(getattr(lay, "data", []))
            if pts is None:
                pts = np.empty((0, 3))

            n = len(pts)
            prev = len(getattr(self, "_remap_point_original_ids", []))

            # remove trailing widgets if points were deleted
            if n < prev:
                try:
                    while len(self._remap_target_widgets) > n:
                        w = self._remap_target_widgets.pop()
                        widget = w.get("widget")
                        try:
                            self._remap_entries_container.removeWidget(widget)
                        except Exception:
                            pass
                        try:
                            widget.setParent(None)
                        except Exception:
                            pass
                        self._remap_point_original_ids.pop()
                except Exception:
                    pass

            # For each point, compute original segment id and create UI entry if new
            for i in range(n):
                try:
                    coord = pts[i]
                    z = int(round(float(coord[0])))
                    y = int(round(float(coord[1])))
                    x = int(round(float(coord[2])))
                    orig = 0
                    try:
                        if self.segmentation_4d is not None and 0 <= self.current_timestep < self.n_timesteps:
                            vol = self.segmentation_4d[int(self.current_timestep)]
                            # bounds check
                            if 0 <= z < vol.shape[0] and 0 <= y < vol.shape[1] and 0 <= x < vol.shape[2]:
                                orig = int(vol[z, y, x])
                    except Exception:
                        orig = 0

                    if i < prev:
                        # update stored original id and refresh label text
                        try:
                            self._remap_point_original_ids[i] = orig
                            self._remap_target_widgets[i]["label"].setText(f"Point #{i+1} (orig {orig}) → target ID:")
                        except Exception:
                            pass
                    else:
                        # create new UI entry for this point
                        try:
                            self._add_remap_entry(i, orig)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _add_remap_entry(self, index: int, original_id: int):
        """Create a labeled entry (label + SpinBox) for a remap point and add it to the UI container."""
        try:
            container = QtWidgets.QWidget()
            container.setLayout(QtWidgets.QHBoxLayout())
            container.layout().setContentsMargins(0, 0, 0, 0)
            label = QtWidgets.QLabel(f"Point #{index+1} (orig {original_id}) → target ID:")
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 2_000_000_000)
            spin.setValue(0)
            container.layout().addWidget(label)
            container.layout().addWidget(spin)

            try:
                self._remap_entries_container.addWidget(container)
            except Exception:
                try:
                    # If container is a QVBoxLayout instance
                    self._remap_entries_container.addWidget(container)
                except Exception:
                    pass

            # store references aligned with points order
            try:
                self._remap_point_original_ids.append(int(original_id))
            except Exception:
                self._remap_point_original_ids.append(0)
            self._remap_target_widgets.append({"widget": container, "label": label, "spin": spin})
        except Exception:
            pass

    def apply_remaps(self):
        """Apply all remapping rules defined by remap points and their target widgets.

        Replaces every voxel of each point's original segment ID with the specified
        target ID in the current timestep's segmentation layer.
        """
        try:
            t = int(getattr(self, "current_timestep", 0) or 0)
            if self.segmentation_4d is None:
                show_info("No segmentation loaded; cannot apply remaps.")
                return

            # iterate over entries
            for i, entry in enumerate(self._remap_target_widgets):
                try:
                    orig = int(self._remap_point_original_ids[i])
                    target = int(entry["spin"].value())
                    if orig == 0:
                        # skip background/no-op
                        continue
                    if target == orig:
                        continue
                    # call existing remapping helper (will update layer and cache)
                    try:
                        self.remap_segment_id(timestep=t, old_id=orig, new_id=target, propagate_forward=False)
                    except Exception:
                        # fallback: manual replace
                        try:
                            mask = self.segmentation_4d[t] == orig
                            if np.any(mask):
                                self.segmentation_4d[t][mask] = target
                                layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
                                if layer is not None:
                                    layer.data = self.segmentation_4d
                                    try:
                                        layer.refresh()
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                show_info("Remapping applied for current timestep.")
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to apply remaps: {e}")

    def clear_remap_points(self):
        """Clear all points in the `remap_points` layer and remove corresponding UI entries."""
        try:
            # clear napari points layer
            lay = getattr(self, "_remap_points_layer", None)
            if lay is not None:
                try:
                    lay.data = np.empty((0, 3))
                except Exception:
                    try:
                        # fallback to .data assignment as list
                        lay.data = []
                    except Exception:
                        pass

            # remove UI widgets
            try:
                while getattr(self, "_remap_target_widgets", None) and len(self._remap_target_widgets) > 0:
                    w = self._remap_target_widgets.pop()
                    widget = w.get("widget")
                    try:
                        self._remap_entries_container.removeWidget(widget)
                    except Exception:
                        pass
                    try:
                        widget.setParent(None)
                    except Exception:
                        pass
            except Exception:
                pass

            # clear stored ids
            try:
                self._remap_point_original_ids = []
            except Exception:
                pass

            try:
                show_info("Cleared remap points and UI entries.")
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to clear remap points: {e}")

    def debug_segmentation_4d(self, t: int) -> bool:
        """Debug segmentation setup for a specific timestep.
        
        Returns True if segmentation can proceed, False if fatal error found.
        """
        print(f"\n🔍 Debugging timestep {t}")
        fatal_error = False
        
        # 1. Check embeddings exist
        if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
            print(f"❌ embeddings_4d attribute missing")
            return False
        
        entry = self.embeddings_4d.get(int(t))
        if entry is None:
            print(f"❌ No embeddings for timestep {t}")
            return False
        else:
            # Check if embeddings are loaded (have features)
            if isinstance(entry, dict) and "features" in entry:
                print(f"✔ Embeddings found (materialized)")
            elif isinstance(entry, dict) and "path" in entry:
                print(f"✔ Embeddings found (lazy, path: {entry['path']})")
            else:
                print(f"⚠ Embeddings found but in unexpected format: {type(entry)}")
        
        # Check AnnotatorState has embeddings activated
        try:
            from ._state import AnnotatorState
            state = AnnotatorState()
            if state.image_embeddings is None:
                print(f"❌ Embeddings not activated in AnnotatorState")
                return False
            else:
                print(f"✔ Embeddings activated in AnnotatorState")
                
            # Check predictor exists
            if state.predictor is None:
                print(f"❌ No predictor in AnnotatorState")
                return False
            else:
                print(f"✔ Predictor exists")
        except Exception as e:
            print(f"❌ Failed to check AnnotatorState: {e}")
            return False
        
        # 2. Check point prompts exist
        if not hasattr(self, "point_prompts_4d"):
            print(f"❌ point_prompts_4d attribute missing")
            return False
        
        pts = self.point_prompts_4d.get(int(t))
        if pts is None or (hasattr(pts, 'size') and pts.size == 0):
            print(f"❌ No point prompts in timestep {t}")
            return False
        else:
            print(f"✔ Point prompts found: {len(pts)} points")
            # Show first few points
            if len(pts) > 0:
                print(f"  First point: {pts[0]}")
        
        # 4. Check image dimensions
        if self.image_4d is None:
            print(f"❌ image_4d is None")
            return False
        
        try:
            image3d = self.image_4d[int(t)]
            shape = image3d.shape
            
            if len(shape) == 3:
                print(f"✔ Image dimensions OK: {shape} (Z, Y, X)")
            elif len(shape) == 2:
                print(f"⚠ Image is 2D: {shape} (Y, X) - may need 2D segmentation")
            elif len(shape) == 4:
                print(f"❌ Image has wrong dimensions: {shape} (should be 3D, got 4D)")
                return False
            else:
                print(f"❌ Unexpected image dimensions: {shape}")
                return False
                
            # Check image is not empty
            if image3d.size == 0:
                print(f"❌ Image is empty")
                return False
                
            # Check image has reasonable values
            img_min, img_max = image3d.min(), image3d.max()
            print(f"  Image value range: [{img_min}, {img_max}]")
            if img_min == img_max:
                print(f"⚠ Image has constant value - may produce poor segmentation")
                
        except Exception as e:
            print(f"❌ Failed to check image: {e}")
            return False
        
        # 5. Check current_object_4d exists
        if self.current_object_4d is None:
            print(f"⚠ current_object_4d is None (will be created)")
        else:
            print(f"✔ current_object_4d exists with shape {self.current_object_4d.shape}")
        
        print(f"✅ All checks passed - ready for segmentation\n")
        return True

    def segment_all_timesteps(self):
        """Run manual segmentation for all timesteps that have point prompts.

        Reuses the same routine as the manual single-timestep/volume segmentation and writes
        each result into `self.current_object_4d[t]` (T, Z, Y, X).
        """
        if self.image_4d is None or self.n_timesteps is None:
            return

        # Ensure container and 4D layer exist
        if self.current_object_4d is None:
            self.current_object_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                if "current_object_4d" in self._viewer.layers:
                    self._viewer.layers["current_object_4d"].data = self.current_object_4d
                else:
                    self._viewer.add_labels(self.current_object_4d, name="current_object_4d")
            except Exception:
                pass

        # Mapping of per-timestep prompts
        prompt_map = getattr(self, "point_prompts_4d", None) or {}

        # Get point prompts layer reference
        point_layer = self._viewer.layers["point_prompts"] if "point_prompts" in self._viewer.layers else None

        # Remember original timestep to restore at the end
        original_t = getattr(self, "current_timestep", 0)
        original_pts = prompt_map.get(original_t, np.empty((0, 3)))

        # Disconnect point prompts event listener during batch segmentation to prevent interference
        if point_layer is not None and hasattr(point_layer, 'events') and hasattr(point_layer.events, 'data'):
            try:
                if hasattr(self, '_point_prompt_connection'):
                    point_layer.events.data.disconnect(self._point_prompt_connection)
            except Exception:
                pass

        # Access volumetric segmentation widget if available
        try:
            state = AnnotatorState()
            seg_widget = state.widgets.get("segment_nd") if getattr(state, "widgets", None) else None
        except Exception:
            seg_widget = None

        for t in range(int(self.n_timesteps)):
            pts = prompt_map.get(t, np.empty((0, 3)))
            pts_arr = np.asarray(pts) if pts is not None else np.empty((0, 3))
            if pts_arr.size == 0:
                print(f"⏭️  Skipping timestep {t} - no point prompts")
                continue

            # Update current timestep internally without triggering full UI updates
            self.current_timestep = int(t)

            # Run debug checks before attempting segmentation
            if not self.debug_segmentation_4d(t):
                print(f"⏭️  Skipping timestep {t} due to failed checks\n")
                continue

            # Activate embeddings/predictor for this timestep
            # For batch segmentation, we need to ensure embeddings are fully loaded (not async)
            try:
                # Check if we have a lazy entry that needs loading
                entry = self.embeddings_4d.get(int(t)) if hasattr(self, "embeddings_4d") else None
                if entry is not None and isinstance(entry, dict) and "path" in entry and "features" not in entry:
                    # Load synchronously during batch processing
                    try:
                        self._load_embedding_for_timestep(int(t))
                        # Give a moment for the embedding to be activated
                        import time
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Failed to load embeddings for timestep {t}: {e}")
                        continue
                
                # Now ensure embeddings are active
                if getattr(self, "timestep_embedding_manager", None) is not None:
                    try:
                        self.timestep_embedding_manager.on_timestep_changed(int(t))
                    except Exception:
                        pass
                self._ensure_embeddings_active_for_t(int(t))
                
                # Verify embeddings are actually loaded in AnnotatorState
                from ._state import AnnotatorState
                state = AnnotatorState()
                if state.image_embeddings is None:
                    print(f"Warning: Embeddings not activated for timestep {t}, skipping")
                    continue
                    
                # Ensure image metadata is set for segmentation
                if state.image_shape is None:
                    try:
                        image3d = self.image_4d[int(t)]
                        state.image_shape = tuple(image3d.shape)
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"Failed to activate embeddings for timestep {t}: {e}")
                continue

            # Manually set point prompts for this timestep without triggering callbacks
            try:
                if point_layer is not None:
                    point_layer.data = pts_arr
                elif "point_prompts" not in self._viewer.layers:
                    point_layer = self._viewer.add_points(pts_arr, name="point_prompts", size=10,
                                                          face_color="green", edge_color="black",
                                                          blending="translucent")
                    point_layer.face_color_cycle = ["limegreen", "red"]
                    
                    # Setup cursor handling for point prompts layer
                    try:
                        canvas = self._viewer.window.qt_viewer.canvas.native
                        
                        def on_mouse_enter(event):
                            """Set crosshair cursor when entering viewer in add mode"""
                            try:
                                if point_layer.mode == 'add':
                                    canvas.setCursor(Qt.CrossCursor)
                            except Exception:
                                pass
                        
                        def on_mouse_leave(event):
                            """Reset cursor to normal when leaving the viewer"""
                            try:
                                canvas.setCursor(Qt.ArrowCursor)
                            except Exception:
                                pass
                        
                        # Connect to canvas enter/leave events
                        try:
                            original_enter = canvas.enterEvent
                            canvas.enterEvent = lambda event: (on_mouse_enter(event), original_enter(event) if callable(original_enter) else None)
                            canvas.leaveEvent = lambda event: on_mouse_leave(event)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

            # Run manual volumetric segmentation
            try:
                print(f"Attempting segmentation for timestep {t} with {len(pts_arr)} point prompts")
                
                # Use the prompt_segmentation utility directly for 3D segmentation
                from . import util as sam_util
                
                # Get the 3D image for this timestep
                image3d = self.image_4d[int(t)]
                
                # Convert point prompts to the format expected by prompt_segmentation
                if len(pts_arr) > 0:
                    # Point prompts are in (Z, Y, X) format
                    points = pts_arr[:, :3]  # Z, Y, X coordinates
                    labels = np.ones(len(points), dtype=int)  # All positive prompts by default
                    
                    # Get the shape for segmentation (Z, Y, X)
                    shape = image3d.shape
                    
                    # For 3D segmentation, we need to segment from the slice containing the point
                    # Get the Z coordinate of the first point
                    z_slice = int(points[0, 0])
                    
                    # Get 2D points (Y, X) for this slice
                    points_2d = points[:, 1:3]  # Just Y, X
                    shape_2d = shape[1:]  # Just Y, X shape
                    
                    # Perform 2D segmentation on the initial slice
                    from ._state import AnnotatorState
                    state = AnnotatorState()
                    
                    print(f"  Segmenting slice {z_slice} with points at {points_2d}")
                    
                    # Segment the initial slice using 2D segmentation
                    seg_2d = sam_util.prompt_segmentation(
                        state.predictor,
                        points_2d,
                        labels,
                        boxes=np.array([]),
                        masks=None,
                        shape=shape_2d,
                        multiple_box_prompts=False,
                        image_embeddings=state.image_embeddings,
                        i=z_slice,  # Specify the slice index for 3D embeddings
                    )
                    
                    if seg_2d is not None and seg_2d.max() > 0:
                        print(f"  Initial 2D segmentation successful, extending to 3D volume")
                        
                        # Create 3D segmentation array
                        seg = np.zeros(shape, dtype=np.uint32)
                        seg[z_slice] = seg_2d
                        
                        # Now extend the segmentation to neighboring slices using segment_mask_in_volume
                        from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
                        
                        segmented_slices = np.array([z_slice])
                        
                        try:
                            seg, (z_min, z_max) = segment_mask_in_volume(
                                seg,
                                state.predictor,
                                state.image_embeddings,
                                segmented_slices,
                                stop_lower=False,
                                stop_upper=False,
                                iou_threshold=0.5,
                                projection="mask",
                                verbose=False,
                            )
                            print(f"  Extended segmentation from slice {z_min} to {z_max}")
                        except Exception as e:
                            print(f"  Note: Could not extend to full volume (using single slice only): {e}")
                            # Keep the single slice segmentation
                    else:
                        seg = None
                    
                    if seg is not None:
                        print(f"Segmentation successful for timestep {t}, shape: {seg.shape}")
                        
                        # 6. Check if SAM returned empty mask
                        mask_max = seg.max()
                        mask_min = seg.min()
                        mask_nonzero = np.count_nonzero(seg)
                        
                        print(f"  Mask stats: min={mask_min}, max={mask_max}, nonzero_pixels={mask_nonzero}")
                        
                        if mask_max == 0:
                            print(f"❌ SAM produced empty mask for timestep {t}")
                            print(f"  Possible causes:")
                            print(f"    - Point prompts are outside image bounds")
                            print(f"    - Wrong embeddings loaded (mismatch with image)")
                            print(f"    - Image has very low contrast")
                            print(f"    - Predictor not properly initialized")
                            continue
                        
                        # Store directly into current_object_4d and update the layer
                        self.current_object_4d[int(t)] = seg.astype(np.uint32)
                        
                        # Update the napari layer
                        try:
                            lay = self._viewer.layers["current_object_4d"] if "current_object_4d" in self._viewer.layers else None
                            if lay is not None:
                                lay.refresh()
                                print(f"✅ Segmentation stored and layer refreshed for timestep {t}")
                            else:
                                print(f"⚠ current_object_4d layer not found in viewer")
                        except Exception as e:
                            print(f"Failed to refresh layer: {e}")
                    else:
                        print(f"Segmentation returned None for timestep {t}")
                else:
                    print(f"No point prompts for timestep {t}, skipping")
                    
            except Exception as e:
                print(f"Segmentation failed for timestep {t}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Restore original timestep and its point prompts
        try:
            self.current_timestep = int(original_t)
            if point_layer is not None:
                point_layer.data = np.asarray(original_pts) if original_pts is not None else np.empty((0, 3))
        except Exception:
            pass

        # Reconnect point prompts event listener
        if point_layer is not None and hasattr(point_layer, 'events') and hasattr(point_layer.events, 'data'):
            try:
                if hasattr(self, '_point_prompt_connection'):
                    point_layer.events.data.connect(self._point_prompt_connection)
            except Exception:
                pass

        try:
            show_info("Finished segmenting all timesteps with point prompts.")
        except Exception:
            pass

    def commit_all_timesteps(self):
        """Transfer all non-empty `current_object_4d[t]` into `committed_objects_4d` using one new global ID."""
        layer = self._viewer.layers["committed_objects_4d"] if "committed_objects_4d" in self._viewer.layers else None
        if layer is None:
            if self.segmentation_4d is None and self.image_4d is not None:
                self.segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                layer = self._viewer.add_labels(self.segmentation_4d, name="committed_objects_4d")
            except Exception:
                return

        # Sync local reference
        try:
            self.segmentation_4d = layer.data
        except Exception:
            pass

        # Global max across all timesteps for uniqueness
        try:
            global_max = int(self.segmentation_4d.max()) if self.segmentation_4d is not None else 0
        except Exception:
            global_max = 0
        new_id = int(global_max) + 1

        for t in range(int(self.n_timesteps)):
            if self.current_object_4d is None:
                break
            try:
                seg_t = np.asarray(self.current_object_4d[int(t)])
            except Exception:
                seg_t = None
            if seg_t is None or seg_t.size == 0 or not np.any(seg_t != 0):
                continue

            mask = seg_t != 0
            try:
                if hasattr(layer, "events") and hasattr(layer.events, "data"):
                    with layer.events.data.blocker():
                        layer.data[int(t)][mask] = new_id
                else:
                    layer.data[int(t)][mask] = new_id
            except Exception:
                try:
                    arr = np.asarray(layer.data[int(t)])
                    arr[mask] = new_id
                    layer.data[int(t)] = arr
                except Exception:
                    pass

            # Update local cache
            try:
                self.segmentation_4d = layer.data
                if self._segmentation_cache is None:
                    self._segmentation_cache = [None] * self.n_timesteps
                self._segmentation_cache[int(t)] = np.asarray(layer.data[int(t)]).copy()
            except Exception:
                pass

        try:
            layer.refresh()
        except Exception:
            pass

        # Cleanup: clear all point prompts and current object masks after committing
        # Clear stored per-timestep prompts
        try:
            self.point_prompts_4d = {}
        except Exception:
            pass
        # Clear napari points layer
        try:
            if "point_prompts" in self._viewer.layers:
                self._viewer.layers["point_prompts"].data = np.empty((0, 3))
        except Exception:
            pass

        # Clear the 4D current object layer content
        try:
            if self.current_object_4d is not None:
                self.current_object_4d[...] = 0
            if "current_object_4d" in self._viewer.layers:
                lay_cur = self._viewer.layers["current_object_4d"]
                if hasattr(lay_cur, "events") and hasattr(lay_cur.events, "data"):
                    with lay_cur.events.data.blocker():
                        lay_cur.data = self.current_object_4d
                else:
                    lay_cur.data = self.current_object_4d
                try:
                    lay_cur.refresh()
                except Exception:
                    pass
        except Exception:
            pass

        # Optional: also clear 3D current_object layer if present
        try:
            if "current_object" in self._viewer.layers:
                co3d = self._viewer.layers["current_object"].data
                zeros3d = np.zeros_like(co3d, dtype=np.uint32)
                self._viewer.layers["current_object"].data = zeros3d
                try:
                    self._viewer.layers["current_object"].refresh()
                except Exception:
                    pass
        except Exception:
            pass
